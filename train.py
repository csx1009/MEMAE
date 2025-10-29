import argparse
from pathlib import Path

import yaml

from mae.main_pretrain import main
from mae.util.io_manager import Args, copy_files, get_output_folder

import sequence as seq
from utils.data_get import BrainDataset


def train_model(par_file):
    with open(par_file, 'r') as ff:
        par = ff.read()
        args = yaml.safe_load(par)

    # train_data_args = Args({**args['data']['general'], **args['data']['train']})
    # dataset_train = seq.CustomDataset(train_data_args)
    path = {
        "HCP": "/data/csx/HCP_microstructure/HCP_microstructure/",
        "Caffine": "/data/csx/Caffine/Caffine/",
        "BTC": "/data/csx/BTC/BTC"
    }
    modality = {
        "MSDKI": ['DI', 'F', 'MSD', 'MSK', 'uFA'],
        "FWDTI": ['AD', 'FA', 'FW', 'MD', 'RD'],
        "AMICO/NODDI": ['fit_FWF', 'fit_NDI', 'fit_ODI']
    }
    dataset_train = BrainDataset(path, modality, p='train')

    model_args = Args(args['model'])

    # model_args.output_dir = get_output_folder(
    #     args['output']['output_dir'],
    #     args['output']['sub'],
    # )
    model_args.output_dir = args['output']['output_dir']
    print('output dir: ', model_args.output_dir)
    model_args.log_dir = model_args.output_dir

    # sequence output size = model input size
    model_args.input_size = [128, 128, 50]

    if model_args.output_dir:
        Path(model_args.output_dir).mkdir(parents=True, exist_ok=True)
    # files_to_copy = [
    #     __file__,
    #     seq.__file__,
    #     [par_file, 'parameter.yml'],
    # ]
    # copy_files(files_to_copy, model_args.output_dir)
    main(model_args, dataset_train)


def get_param_files(path):
    path = Path(path)
    if path.is_dir():
        files = sorted([ff for ff in path.iterdir() if ff.suffix == '.yml'])
    elif path.suffix == '.yml':
        files = [path]
    else:
        raise ValueError('non valid parameter files (must be folder or .yml)')
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--pdir', default='/home/csx/Mamba_MAE/parameter/par.yml', type=str, help='parameter directory')
    parser.add_argument('--gpu', default=0, type=int, help='choose gpu [0, 1]')
    # 创建Namespace对象并手动设置参数
    # args = argparse.Namespace(pdir='/home/csx/Manba_MAE/parameter/par.yml', gpu='0')
    # 解析参数
    pargs = parser.parse_args()
    # pargs = parser.parse_args()
    # print(1)
    for paramf in get_param_files(pargs.pdir):
        print('processing: ' + str(paramf))
        # try:
        train_model(paramf)
        # except Exception as e:
        #     print(f'failed: {e}')
