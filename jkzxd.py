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
from utils.data_get import read_Caffine
from utils.data_get import read_BTC
from utils.anomaly import BrainAnomalyDetector as BAD
import torchio as tio

device = 'cuda:2'


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
num = '26'
# x, mask = read_BTC("BTC", path["BTC"], modality, p='train', mask=True,num=3)
# x = read_Caffine("Caffine", path["Caffine"], modality, p="train")
#
# bad = BAD(model)
# bad.build_memory_bank(x, mask_dataset=mask, memory_bank_path=f'/data/csx/BTC/memory_bank_diff_m20.pt',
#                       device=device, n=4)
bad = BAD(model, healthy_bank_path='/data/csx/Caffine/memory_bank_diff.pt')
# x, mask = read_BTC("BTC", path["BTC"], modality, p='test', mask=True, num=3)
x = read_Caffine("Caffine", path["Caffine"], modality, p="test", num=10)
np.save(f"/data/csx/Caffine/x_Caffine_10.npy", x)
print(x.shape)
mask = x > 0
mask = mask[:, 0, :, :, :]
x = torch.tensor(x)
# x = x.unsqueeze(dim=0)
x = x.to(device)
# print(x.shape)
d, s = bad.detect_anomaly(x, mask)

d = np.array(d)
np.save(f'/data/csx/Caffine/d_Caffine_10.npy', d)
print(s)
print(d.shape)
