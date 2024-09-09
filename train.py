import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
from datas.utils import create_datasets
import math
import argparse, yaml
import utils

from tqdm import tqdm
import sys
import time
import glob
from torch.utils.tensorboard import SummaryWriter
from utils import ldr_f2u, Cutout, cut_out, cutmix
import numpy as np
import cv2
import random
from losses import SemanticLoss
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='M2Trans')
## yaml configuration files
parser.add_argument('--config', type=str, default='./configs/M2Trans_x4.yml', help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')


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

    ## create dataset for training and validating
    train_dataloader, valid_dataloaders = create_datasets(args)

    ## definitions of model
    try:
        model = utils.import_module('models.{}_network'.format(args.model)).create_model(args)
    except Exception:
        raise ValueError('not supported model type! or something')
    model = nn.DataParallel(model).to(device)

    ## definition of loss and optimizer
    loss_l1 = torch.nn.L1Loss()
    # You can change the patches num here.
    loss_clip = SemanticLoss(criterion='l1', N_patches=3) 
    lambda_l1 = args.lambda_l1
    lambda_clip = args.lambda_clip
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.eta_min)

    ## load pretrain
    if args.pretrain is not None:
        print('## load pretrained model: {}! ##'.format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load_state_dict(ckpt['model_state_dict'])

    ## resume training
    start_epoch = 1
    if args.resume is not None:
        ckpt_files = glob.glob(os.path.join(args.resume, 'models', "*.pt"))
        if len(ckpt_files) != 0:
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.replace('.pt','').split('_')[-1]))
            ckpt = torch.load(ckpt_files[-1])
            prev_epoch = ckpt['epoch']

            start_epoch = prev_epoch + 1
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            stat_dict = ckpt['stat_dict']
            ## reset folder and param
            experiment_path = args.resume
            log_name = os.path.join(experiment_path, 'log.txt')
            experiment_model_path = os.path.join(experiment_path, 'models')
            print('## select {}, resume training from epoch {}. ##'.format(ckpt_files[-1], start_epoch))
    else:
        ## auto-generate the output logname
        experiment_name = None
        timestamp = utils.cur_timestamp_str()
        if args.log_name is None:
            experiment_name = '{}-{}-x{}-{}'.format(args.model, 'fp32', args.scale, timestamp)
        else:
            experiment_name = '{}-{}'.format(args.log_name, timestamp)
        experiment_path = os.path.join(args.log_path, experiment_name)
        log_name = os.path.join(experiment_path, 'log.txt')
        stat_dict = utils.get_stat_dict()
        
        ## create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
            
        ## save training paramters
        exp_params = vars(args)
        exp_params_name = os.path.join(experiment_path, 'config.yml')
        with open(exp_params_name, 'w') as exp_params_file:
            yaml.dump(exp_params, exp_params_file, default_flow_style=False)
    
    ## create folder for test results
    experiment_test_path = os.path.join(experiment_path, 'test_results_x' + str(args.scale))
    if not os.path.exists(experiment_test_path):
        os.makedirs(experiment_test_path)
            
    # ## print architecture of model
    sys.stdout = utils.ExperimentLogger(log_name, sys.stdout)
    print(model)
    sys.stdout.flush()

    # 初始化 tensorboard
    writer = SummaryWriter(log_dir=experiment_path)

    # model complexity
    from ptflops import get_model_complexity_info
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    with torch.no_grad():
        flops, params = get_model_complexity_info(model, (3, 384//args.scale, 384//args.scale), as_strings=True, print_per_layer_stat=False, verbose=False)
        print('## Flops: ', flops, ', Params: ', params)

    # load text descriptions

    with open('path/to/your/text_description/file','r', encoding="utf-16") as f:
        cap_tokens = [line.strip() for line in f.readlines()]


    ## start training
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    timer_start = time.time()
    for epoch in tqdm(range(start_epoch, args.epochs+1), desc="Training"):
        epoch_loss = 0.0
        l1_loss = 0.0
        total_clip_loss = 0.0 
        stat_dict['epochs'] = epoch
        model = model.train()
        opt_lr = scheduler.get_last_lr()
        tqdm.write('## =========== {}-training, Epoch: {}, lr: {} ============= ##'.format('fp32', epoch, opt_lr))

        for iter, batch in tqdm(enumerate(train_dataloader), desc="Processing each epoch", total=len(train_dataloader)):
            optimizer.zero_grad()
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            if args.cutmix:
                lr, hr = cutmix(lr, hr, alpha=1.0, n_patch=np.random.randint(1,5), scale=args.scale)
            
            if args.cutout and epoch < (args.epochs * 0.2):
                lr = cut_out(lr, n_holes=np.random.randint(1,10), length=int(0.1*args.patch_size//args.scale))

            sr = model(lr)  # [B,3,384,384]

            #add for clip
            import skimage.color as sc

            # get a batch of tokens
            batch_tokens = []
            token_index = iter*sr.shape[0]
            token_outdex = (iter+1)*sr.shape[0]
            for token_index_i in range(token_index,token_outdex):
                batch_tokens.append(cap_tokens[token_index_i%int((len(cap_tokens)))])
                # batch_tokens.append('[none]')
            
            #add for clip end

            if lambda_l1 > 0:
                l1loss = loss_l1(sr, hr) * lambda_l1
            else:
                l1loss = 0
            if lambda_clip > 0:
                clip_loss = 0.0
                for index in range(sr.size()[0]):
                    clip_loss += loss_clip(sr[index], hr[index], batch_tokens[index]) * lambda_clip
            else:
                clip_loss = 0
            loss = l1loss + clip_loss
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)
            l1_loss += float(l1loss)
            total_clip_loss += float(clip_loss)
            

            # save figures to Tensorboard
            if iter % 200 == 0:
                low_img = lr[0].detach().cpu().squeeze().numpy()
                low_img = ldr_f2u(low_img, minv=0, maxv=args.rgb_range)
                high_img = hr[0].detach().cpu().squeeze().numpy()
                high_img = ldr_f2u(high_img, minv=0, maxv=args.rgb_range)
                sr_img = sr[0].detach().cpu().squeeze().numpy()
                sr_img = ldr_f2u(sr_img, minv=0, maxv=args.rgb_range)

                low_img = np.transpose(low_img,(1,2,0))
                lr_up = cv2.resize(low_img,(high_img.shape[2], high_img.shape[1]),interpolation=cv2.INTER_LINEAR)
                low_img = np.transpose(low_img,(2,0,1))
                lr_up = np.transpose(lr_up,(2,0,1))

                img_comp = np.concatenate((lr_up, sr_img, high_img), axis=2) 
                writer.add_image(f'Train/lr_rec_image', low_img, iter, dataformats='CHW')
                writer.add_image(f'Train/lr_sr_hr_image', img_comp, iter, dataformats='CHW')

            if (iter + 1) % args.log_every == 0:
                cur_steps = (iter+1)*args.batch_size
                total_steps = len(train_dataloader.dataset)
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)

                epoch_width = math.ceil(math.log10(args.epochs))
                cur_epoch = str(epoch).zfill(epoch_width)

                avg_loss = epoch_loss / (iter + 1)
                print(avg_loss)
                avgl1_loss = l1_loss / (iter + 1)
                avgtotal_clip_loss = total_clip_loss / (iter + 1)
                stat_dict['losses'].append((avg_loss) / (iter + 1))

                timer_end = time.time()
                duration = timer_end - timer_start
                timer_start = timer_end
                tqdm.write('Epoch:{}, {}/{}, loss: {:.4f}, L1loss: {:.4f}, CLIPloss: {:.8f} time: {:.3f}'.format(cur_epoch, cur_steps, total_steps, avg_loss, avgl1_loss, avgtotal_clip_loss, duration))

                step = (epoch-1)*len(train_dataloader.dataset) + (iter+1)*args.batch_size
                writer.add_scalar("Train/loss",scalar_value=loss.item(), global_step=step)

        if epoch % args.test_every == 0:
            torch.set_grad_enabled(False)
            test_log = ''
            model = model.eval()
            tqdm.write("## validation ##")
            with torch.no_grad():
                for valid_dataloader in valid_dataloaders:
                    avg_psnr, avg_ssim = 0.0, 0.0
                    name = valid_dataloader['name']
                    loader = valid_dataloader['dataloader']
                    name = valid_dataloader['name']
                    count = 0

                    for lr, hr, img_name in tqdm(loader, ncols=80):
                        count += 1
                        lr, hr = lr.to(device), hr.to(device)
                        sr = model(lr)
                        if args.save_image:
                            if not os.path.exists(os.path.join(experiment_test_path,name)):
                                os.makedirs(os.path.join(experiment_test_path,name))
                            save_image(sr, os.path.join(experiment_test_path,os.path.join(name, img_name[0])))


                        # save figures to Tensorboard
                        if count % 20 == 0:
                            low_img = lr[0].detach().cpu().squeeze().numpy()
                            low_img = ldr_f2u(low_img, minv=0, maxv=args.rgb_range)
                            high_img = hr[0].detach().cpu().squeeze().numpy()
                            high_img = ldr_f2u(high_img, minv=0, maxv=args.rgb_range)
                            sr_img = sr[0].detach().cpu().squeeze().numpy()
                            sr_img = ldr_f2u(sr_img, minv=0, maxv=args.rgb_range)

                            low_img = np.transpose(low_img,(1,2,0))
                            lr_up = cv2.resize(low_img,(high_img.shape[2], high_img.shape[1]),interpolation=cv2.INTER_LINEAR)
                            low_img = np.transpose(low_img,(2,0,1))
                            lr_up = np.transpose(lr_up,(2,0,1))
                            img_comp = np.concatenate((lr_up, sr_img, high_img), axis=2) 
                            writer.add_image(f'Valid_{name}/lr_image', low_img, count, dataformats='CHW')
                            writer.add_image(f'Valid_{name}/lr_sr_hr_image', img_comp, count, dataformats='CHW')

                        # conver to ycbcr
                        if args.colors == 3:
                            hr_ycbcr = utils.rgb_to_ycbcr(hr)
                            sr_ycbcr = utils.rgb_to_ycbcr(sr)
                            hr = hr_ycbcr[:, 0:1, :, :]
                            sr = sr_ycbcr[:, 0:1, :, :]

                        # crop image for evaluation
                        hr = hr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                        sr = sr[:, :, args.scale:-args.scale, args.scale:-args.scale]
                        if args.rgb_range == 1:
                            hr, sr = hr*255., sr*255.
                        # calculate psnr and ssim
                        psnr = utils.calc_psnr(sr, hr)       
                        ssim = utils.calc_ssim(sr, hr)         
                        avg_psnr += psnr
                        avg_ssim += ssim

                    avg_psnr = round(avg_psnr/len(loader) + 5e-3, 2)
                    avg_ssim = round(avg_ssim/len(loader) + 5e-5, 4)

                    # write to Tensorboard
                    writer.add_scalars(f'Valid_{name}/PSNR', {"PSNR": avg_psnr}, epoch)
                    writer.add_scalars(f'Valid_{name}/SSIM', {"SSIM": avg_ssim}, epoch)

                    stat_dict[name]['psnrs'].append(avg_psnr)
                    stat_dict[name]['ssims'].append(avg_ssim)
                    if stat_dict[name]['best_psnr']['value'] < avg_psnr:
                        stat_dict[name]['best_psnr']['value'] = avg_psnr
                        stat_dict[name]['best_psnr']['epoch'] = epoch
                    if stat_dict[name]['best_ssim']['value'] < avg_ssim:
                        stat_dict[name]['best_ssim']['value'] = avg_ssim
                        stat_dict[name]['best_ssim']['epoch'] = epoch

                    test_log += '[{}-X{}], PSNR/SSIM: {:.4f}/{:.4f} (Best: {:.4f}/{:.4f}, Epoch: {}/{})\n'.format(
                        name, args.scale, float(avg_psnr), float(avg_ssim), 
                        stat_dict[name]['best_psnr']['value'], stat_dict[name]['best_ssim']['value'], 
                        stat_dict[name]['best_psnr']['epoch'], stat_dict[name]['best_ssim']['epoch'])

            # print log & flush out
            tqdm.write(test_log)
            sys.stdout.flush()

            # save model
            saved_model_path = os.path.join(experiment_model_path, 'model_x{}_{}.pt'.format(args.scale, epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'stat_dict': stat_dict
            }, saved_model_path)

            torch.set_grad_enabled(True)
            # save stat dict
            ## save training paramters
            stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
        ## update scheduler
        scheduler.step()


# python train.py --config ./configs/M2Trans_x2.yml
# python train.py --config ./configs/M2Trans_x3.yml
# python train.py --config ./configs/M2Trans_x4.yml

