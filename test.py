# from slices import SliceIterator

import argparse
from pathlib import Path
import sys

import torch
import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
import yaml

# sys.path.append('./../training/')
from mae import model_mamba
from utils.data_get import read_HCP, read_BTC, read_Caffine
from utils.model_deep import model_deep

device = 'cuda:3'


def forward(model, img):
    x = torch.tensor(img)
    # x = x.unsqueeze(dim=0)
    x = x.to(device)
    mask_ratio = 0.75
    loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
    # 预测 latent 五次
    # num_predictions = 5
    # latent_list = []
    # for _ in range(num_predictions):
    #     latent, mask, ids_restore = model.forward_encoder(x.float(), mask_ratio)
    #     latent_list.append(latent)
    #
    # # 对五次预测的 latent 取均值
    # latent_mean = torch.mean(torch.stack(latent_list), dim=0)
    # y = model.forward_decoder(latent_mean, ids_restore)  # [N, L, p*p*p*c]
    # loss = model.forward_loss(x.float(), y, mask)
    print(y.shape)
    print(f"mask_ratio:{mask_ratio}   loss:{loss}")
    y = model.unpatchify(y)
    y = y.detach().cpu()

    mask = mask.detach().cpu()
    mask = mask.unsqueeze(-1).repeat(1, 1, int(np.prod(model.patch_size)))
    mask = model.unpatchify(mask).cpu()
    x = x.cpu()
    im_paste = x * (1 - mask) + y * mask
    im_masked = x * (1 - mask)
    return im_paste, im_masked


# -------------------------------------------------------

# load parameters
class Args:
    def __init__(self, d):
        for key, value in d.items():
            if type(value) == dict:
                value = Args(value)
            setattr(self, key, value)


par_file = '/home/csx/Mamba_MAE/parameter/par.yml'
with open(par_file, 'r') as ff:
    par = ff.read()
    args = yaml.safe_load(par)
model_args = Args(args['model'])

model_args.device = device
model_args.input_size = [128, 128, 50]

# args['data']['train']['label_file'] = label_file


# load model
model = model_mamba.__dict__[model_args.model](
    img_size=model_args.input_size,
    patch_size=[4, 4, 5],
    norm_pix_loss=model_args.norm_pix_loss,
    in_chans=13
)
# model_deep(model)
checkpoint = torch.load("/data/csx/MAE/output_13_1000_445_EP2000/checkpoint-1999.pth", map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=False)
print(msg)
_ = model.to(device)
path = {
    "HCP": "/data/csx/HCP_microstructure/",
    "Caffine": "/data/csx/Caffine/Caffine/",
    "BTC": "/data/csx/BTC/BTC"
}
modality = {
    "MSDKI": ['DI', 'F', 'MSD', 'MSK', 'uFA'],
    "FWDTI": ['AD', 'FA', 'FW', 'MD', 'RD'],
    "AMICO/NODDI": ['fit_FWF', 'fit_NDI', 'fit_ODI']
}
num = 26
# x = read_HCP("HCP", path["HCP"], modality)
# x, _ = read_BTC("BTC", path["BTC"], modality, p='test', num=num)
x = read_Caffine("Caffine", path["Caffine"], modality, p="test", num=10)
print(x.shape)
np.save(f"/data/csx/Caffine/x_Caffine_10.npy", x)
for j in range(x.shape[1]):
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(24, 24))
    for i, ax in enumerate(axes.flat):
        # median_data = median(data[:, :, i, l], mask[:, :, i], k=11)
        ax.imshow(x[0, j, :, :, i * 2], cmap='gray')  # 调整通道顺序并显示图像
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/data/csx/Caffine/plt_10/x_{j}.png")
print(x.shape)
y_list = []
for i in range(5):
    y, mask = forward(model, x)
    y_list.append(y[0, :, :, :, :])
y_list = np.array(y_list)
y = np.mean(y_list, axis=0)
# y, mask = forward(model, x)
print(y.shape)
np.save(f"/data/csx/Caffine/y_Caffine_10.npy", y)
for j in range(y.shape[0]):
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(24, 24))
    for i, ax in enumerate(axes.flat):
        # median_data = median(data[:, :, i, l], mask[:, :, i], k=11)
        ax.imshow(y[j, :, :, i * 2], cmap='gray')  # 调整通道顺序并显示图像
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"/data/csx/Caffine/plt_10/y_{j}.png")
# # for j in range(y.shape[1]):
#     fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(24, 24))
#     for i, ax in enumerate(axes.flat):
#         # median_data = median(data[:, :, i, l], mask[:, :, i], k=11)
#         ax.imshow(mask[0,j, :, :, i*2], cmap='gray')  # 调整通道顺序并显示图像
#         ax.axis('off')
#     plt.tight_layout()
#     plt.savefig("/data/csx/MAE/plt_MMAE/mask_{}.png".format(j))
# print(mask.shape)
