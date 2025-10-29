from os.path import join, exists

import numpy as np
import torch
from dipy.io.image import load_nifti
import torchio as tio
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os


def directories(directory_path):
    # 使用os.listdir获取目录下的所有条目
    entries = os.listdir(directory_path)

    # 过滤出子文件夹
    subdirectories = [entry for entry in entries if os.path.isdir(join(directory_path, entry))]

    # 打印子文件夹列表
    return subdirectories


def load_nii(Path, k, v):
    path = join(Path, k, v + '.nii.gz')
    d, _ = load_nifti(path)
    return d


def read_HCP(k, v, modality):
    path = v
    sub = directories(v)
    # print(len(sub))
    data_HCP = []
    w = 0
    L = 0
    for k in modality:
        L = L + len(modality[k])
    for sub_dir in sub:
        # print(sub_dir)
        path_sub = join(path, sub_dir)
        data_HCP_ = []
        f = True
        for k_m, v_m in modality.items():
            if not exists(join(path_sub, k_m)):
                f = False
        if f:
            for k_m, v_m in modality.items():
                for v_m_path in v_m:
                    data_HCP_.append(load_nii(path_sub, k_m, v_m_path))
        if len(data_HCP_) == L:
            d = np.array(data_HCP_)
            data_HCP.append(d)
        # w = w + 1
        # if w == 10:
        #     break
    return np.array(data_HCP)


def read_Caffine(k, v, modality, p):
    con_list = [
        'sub-021', 'sub-022', 'sub-028', 'sub-029', 'sub-031', 'sub-032', 'sub-033', 'sub-036', 'sub-038',
        'sub-041', 'sub-042', 'sub-044', 'sub-049', 'sub-050', 'sub-051', 'sub-054', 'sub-055', 'sub-059',
        'sub-062', 'sub-065', 'sub-068', 'sub-075', 'sub-077', 'sub-078', 'sub-082', 'sub-084', 'sub-085',
        'sub-086', 'sub-087', 'sub-093', 'sub-094', 'sub-095', 'sub-097', 'sub-098', 'sub-099', 'sub-100',
        'sub-101', 'sub-105', 'sub-106', 'sub-110', 'sub-111', 'sub-112', 'sub-115', 'sub-117', 'sub-118',
        'sub-122', 'sub-123', 'sub-125', 'sub-126', 'sub-129', 'sub-130', 'sub-133', 'sub-134', 'sub-146',
        'sub-147', 'sub-153', 'sub-154', 'sub-156', 'sub-159', 'sub-160'
    ]
    path = v
    sub = directories(v)
    # print(sub)
    data_Caffine = []
    L = 0
    w = 0
    for k in modality:
        L = L + len(modality[k])
    if p == 'train':
        for sub_dir in sub:
            if sub_dir in con_list:
                path_sub = join(path, sub_dir)
                data_Caffine_ = []
                f = True
                for k_m, v_m in modality.items():
                    if not exists(join(path_sub, k_m)):
                        f = False
                if f:
                    for k_m, v_m in modality.items():
                        for v_m_path in v_m:
                            data_Caffine_.append(load_nii(path_sub, k_m, v_m_path))
                if len(data_Caffine_) == L:
                    d = np.array(data_Caffine_)
                    data_Caffine.append(d)
                    w = w + 1
            if w == 1:
                break
    else:
        for sub_dir in sub:
            if sub_dir not in con_list:
                path_sub = join(path, sub_dir)
                data_Caffine_ = []
                for k_m, v_m in modality.items():
                    # print(exists(join(path_sub, k_m)))
                    # print(join(path_sub, k_m))
                    f = True
                    for k_m, v_m in modality.items():
                        if not exists(join(path_sub, k_m)):
                            f = False
                    if f:
                        for k_m, v_m in modality.items():
                            for v_m_path in v_m:
                                data_Caffine_.append(load_nii(path_sub, k_m, v_m_path))
                if len(data_Caffine_) == L:
                    d = np.array(data_Caffine_)
                    data_Caffine.append(d)
    return np.array(data_Caffine)


def read_BTC(k, v, modality, p):
    if p == 'train':
        path_list = ['postop_sub-CON', 'postop_sub-PAT', 'preop_sub-CON']
    else:
        path_list = ['preop_sub-PAT']
    # print(path_list)
    path = v
    data_BTC = []
    L = 0
    for k in modality:
        L = L + len(modality[k])
    for sub_dir in path_list:
        for i in range(1):
            sub_dir_ = sub_dir + '{:02}'.format(i + 3)
            path_sub = join(path, sub_dir_)
            # print(path_sub)
            if exists(path_sub):
                # print(path_sub)
                data_BTC_ = []
                f = True
                for k_m, v_m in modality.items():
                    if not exists(join(path_sub, k_m)):
                        f = False
                if f:
                    for k_m, v_m in modality.items():
                        for v_m_path in v_m:
                            data_BTC_.append(load_nii(path_sub, k_m, v_m_path))
                if len(data_BTC_) == L:
                    d = np.array(data_BTC_)
                    data_BTC.append(d)
    return np.array(data_BTC)


def data_read(Path, modality, p):
    data_HCP = None
    data_Caffine = None
    data_BTC = None
    for k, v in Path.items():
        if k == 'HCP' and p == 'train':
            data_HCP = read_HCP(k, v, modality)
            print(data_HCP.shape)
        elif k == 'Caffine':
            data_Caffine = read_Caffine(k, v, modality, p)
            print(data_Caffine.shape)
        elif k == 'BTC':
            data_BTC = read_BTC(k, v, modality, p)
            print(data_BTC.shape)
    if data_BTC is None:
        return data_HCP
    if data_HCP is not None:
        data = np.concatenate([data_HCP, data_Caffine], axis=0)
        data = np.concatenate([data, data_BTC], axis=0)
    else:
        data = np.concatenate([data_Caffine, data_BTC], axis=0)
    return data


class BrainDataset(Dataset):
    def __init__(self, Path, modality, p='train'):
        self.data = data_read(Path, modality, p)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = 0

        return x, y


# if __name__ == '__main__':
#     path = {
#         "HCP": "/data/csx/HCP_microstructure/",
#         "Caffine": "/data/csx/Caffine/",
#         "BTC": "/data/csx/BTC"
#     }
#     modality = {
#         "MSDKI": ['DI', 'F', 'MSD', 'MSK', 'uFA'],
#         "FWDTI": ['AD', 'FA', 'FW', 'MD', 'RD'],
#         # "AMICO/NODDI": ['fit_FWF', 'fit_NDI', 'fit_ODI']
#     }
#     data = BrainDataset(path, modality, p='train')
#     data_train = DataLoader(data, batch_size=64, shuffle=False, num_workers=40)
#     print(data_train)
#     # data, _ = load_nifti('/data/wtl/BTC/postop_sub-CON02/FWDTI/AD.nii.gz')
#     # # caffine /data/wtl/Caffine/sub-015/FWDTI/AD.nii.gz
#     # # (112, 112, 50)
#     # # BTC /data/wtl/BTC/postop_sub-CON02/FWDTI/AD.nii.gz
#     # # (96,96,60)
#     # # HCP /data/wtl/HCP_microstructure/100206/FWDTI/AD.nii.gz
#     # # (145,174,145)
#     # print(data.shape)
#     # print(np.max(data))
#     # print(np.min(data))
#     #
#     # transform_HCP = tio.Compose([
#     #     tio.Clamp(out_min=0, out_max=150),
#     #     tio.RescaleIntensity(percentiles=(0, 98)),
#     #     tio.CropOrPad(target_shape=(80, 80, 50)),
#     #     tio.Resize(target_shape=(128, 128, 50))])
#     # data = transform_HCP(np.array([data]))
#     # print(np.max(data))
#     # print(np.min(data))
#     #
#     # fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(12, 24))
#     # for i, ax in enumerate(axes.flat):
#     #     # median_data = median(data[:, :, i, l], mask[:, :, i], k=11)
#     #     ax.imshow(data[0, :, :, i], cmap='gray')  # 调整通道顺序并显示图像
#     #     ax.axis('off')
#     # plt.tight_layout()
#     # plt.show()
