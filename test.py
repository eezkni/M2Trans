import os
import torch
import torch.nn as nn
from datas.utils import create_datasets
import math
import argparse, yaml
import utils
import piq

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import ldr_f2u, Cutout, cut_out, cutmix
import numpy as np
import random
from torchvision.utils import save_image
from models.M2Trans_network import *

parser = argparse.ArgumentParser(description='M2Trans')
## yaml configuration files
parser.add_argument('--config', type=str, default='./configs/M2Trans_x2_test.yml', help = 'pre-config file for training')


if __name__ == '__main__':

    args = parser.parse_args()
    if args.config:
       opt = vars(args)
       yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
       opt.update(yaml_args)

    ## set visibel gpu   
    gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    seed = 33
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)

   
    torch.cuda.set_device(0)

    ## select active gpu devices
    device = None
    if args.gpu_ids is not None and torch.cuda.is_available():
        print('## use cuda & cudnn for acceleration! ##')
        print('## the gpu id is: {}'.format(args.gpu_ids))
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        print('## use cpu for training! ##')
        device = torch.device('cpu')
    torch.set_num_threads(args.threads)

    ## create dataset for validating
    _, valid_dataloaders = create_datasets(args)

    ## load checkpoint
    checkpoint = torch.load(args.model_path)
    ## load network
    model = M2Trans(args)
    # model = utils.import_module('models.{}_network'.format(args.model)).create_model(args)
    model = nn.DataParallel(model).to(device)
    ## load parameters
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    torch.set_grad_enabled(False)
    model.eval()
    with torch.no_grad():
        for valid_dataloader in valid_dataloaders:
            avg_psnr, avg_ssim = 0.0, 0.0
            avg_fsim, avg_gmsd = 0.0, 0.0

            name = valid_dataloader['name']
            loader = valid_dataloader['dataloader']
            name = valid_dataloader['name']
            count = 0

            for lr, hr, img_name in tqdm(loader, ncols=80):
                count += 1
                lr, hr = lr.to(device), hr.to(device)
                sr = model(lr)
                

                assert hr.shape == sr.shape
                ## calculate fsim and gmsd
                fsim_index: torch.Tensor = piq.fsim(hr, sr, data_range=1., reduction='none')
                avg_fsim = avg_fsim + fsim_index.item()

                gmsd_index: torch.Tensor = piq.gmsd(hr, sr, data_range=1., reduction='none')
                avg_gmsd= avg_gmsd + gmsd_index.item()
                
                if args.colors == 3:
                    hr_ycbcr = utils.rgb_to_ycbcr(hr)
                    sr_ycbcr = utils.rgb_to_ycbcr(sr)
                    hr = hr_ycbcr[:, 0:1, :, :]
                    sr = sr_ycbcr[:, 0:1, :, :]


                hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                if args.rgb_range == 1:
                    hr, sr = hr*255., sr*255.
                ## calculate psnr and ssim
                psnr = utils.calc_psnr(sr, hr)       
                ssim = utils.calc_ssim(sr, hr)  
                avg_psnr += psnr
                avg_ssim += ssim

            avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
            avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)
            avg_fsim = round(avg_fsim/len(loader) + 5e-5, 4)
            avg_gmsd = round(avg_gmsd/len(loader) + 5e-5, 4)
            print(f"PSNR:{avg_psnr:.2f},SSIM:{avg_ssim:.4f}\nFSIM:{avg_fsim:.4f},GMSD:{avg_gmsd:.4f}")