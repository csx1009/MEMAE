# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import numpy as np
import torch
from torch import nn

from timm.models.vision_transformer import Block

from .util.helpers import PatchEmbed, get_3d_sincos_pos_embed


class MaskedAutoencoderViT_3D(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size: tuple, patch_size: tuple, in_chans,
                 embed_dim=256 * 3, depth=24, num_heads=16,
                 decoder_embed_dim=128 * 3, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # check sizes and calc patch numbers 
        self.img_size = np.array(img_size)
        self.patch_size = np.array(patch_size)
        assert not (self.img_size % self.patch_size).any(), f'''
            img_size modulo patch_size error'''
        self.num_patches = (self.img_size // self.patch_size).astype(int)
        total_num_patches = np.prod(self.num_patches)
        self.in_chans = in_chans
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, total_num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, total_num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, np.prod(patch_size) * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.num_patches,
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_3d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            self.num_patches,
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """ 
        imgs: (N, C, H, W, D)
        x: (N, L, patch_size**3 * C)
        """
        p = self.patch_size  # = (p_h, p_w, p_d)
        c = self.in_chans
        h, w, d = self.num_patches

        x = imgs.reshape(shape=(imgs.shape[0], c, h, p[0], w, p[1], d, p[2]))
        x = torch.einsum('nchpwqdo->nhwdpqoc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, np.prod(p) * c))
        return x

    def unpatchify(self, x):
        """ 
        x: (N, L, patch_size**3 * C)
        imgs: (N, C, H, W, D)
        """
        p = self.patch_size
        h, w, d = self.num_patches
        assert h * w * d == x.shape[1], f'''
            size error: ({h}, {w}, {d}) vs {x.shape[1]}'''

        x = x.reshape(shape=(x.shape[0], h, w, d, p[0], p[1], p[2], -1))
        x = torch.einsum('nhwdpqoc->nchpwqdo', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p[0], w * p[1], d * p[2]))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # print("--------forward_encoder--------")
        # embed patches
        x = self.patch_embed(x)
        # print("embed patches",x.shape)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # print("add pos embed w/o cls token", x.shape)
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # print("length -> length * mask_ratio", x.shape)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print("apply Transformer blocks", x.shape)
        # apply Transformer blocks
        for blk in self.blocks:
            # print("blk", x.shape)
            x = blk(x)
        # print("self.blocks", x.shape)
        x = self.norm(x)
        # print("self.norm", x.shape)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # print("--------forward_encoder--------")
        # embed tokens
        x = self.decoder_embed(x)
        # print("embed patches", x.shape)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # print("append mask tokens to sequence", x.shape)
        # add pos embed
        x = x + self.decoder_pos_embed
        # print("add pos embed", x.shape)
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            # print("blk", x.shape)
            x = blk(x)
        # print("apply Transformer blocks", x.shape)
        x = self.decoder_norm(x)
        # print("self.decoder_norm", x.shape)
        # predictor projection
        x = self.decoder_pred(x)
        # print("predictor projection", x.shape)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*p*c]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# use decoder depth=4 from SpartiotemporalMAE
def mae_vit_base_dec384d8b(**kwargs):
    model = MaskedAutoencoderViT_3D(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base = mae_vit_base_dec384d8b  # decoder: 384 dim, 8 blocks
