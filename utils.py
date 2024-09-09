import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import datetime
import os
import sys
import cv2
from math import exp
from pytorch_msssim import ssim
import importlib
import random


def rand_bbox(size, lam):
    
    W, H = size[2], size[3]
    # cut_rat = np.sqrt(1. - lam)
    cut_rat = np.power(lam, 1/2)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def _cutmix(data, target, alpha=1.0, n_patch=1, scale=2):
    new_data = data.clone()
    new_target = target.clone()
    
    if np.random.random() < 0.5:
        for i in range(n_patch):
            indices = torch.randperm(data.size(0))

            lam = np.clip(np.random.beta(alpha, alpha), 0.1, 0.3)
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
            
            new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
            
            new_target[:, :, bby1*scale:bby2*scale, bbx1*scale:bbx2*scale] = target[indices, :, bby1*scale:bby2*scale, bbx1*scale:bbx2*scale]
    
    return new_data, new_target


def cutmix(data, target, alpha=1.0, n_patch=1, scale=2):
    new_data = data.clone()
    new_target = target.clone()
    
    if new_data.size(0) > 1:
        d1, d2 = new_data.chunk(2, dim=0)
        t1, t2 = new_target.chunk(2, dim=0)
        
        d1, t1 = _cutmix(d1, t1, alpha=alpha, n_patch=n_patch, scale=scale)
        d2, t2 = _cutmix(d2, t2, alpha=alpha, n_patch=n_patch, scale=scale)
        
        new_data = torch.cat([d1, d2], dim=0)
        new_target = torch.cat([t1, t2], dim=0)
    
    else:
        new_data, new_target = _cutmix(new_data, new_target, alpha=alpha, n_patch=n_patch, scale=scale)
    
    return new_data, new_target


def _cut_out(img, n_holes, length):
    b, c, h, w = img.size()
    mask = np.ones((h, w), np.float32)

    if random.random() < 0.5:
        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img).to(img.device, dtype=img.dtype)
        img = img * mask

    return img


def cut_out(img, n_holes, length):
    
    if img.size(0) > 1:
        i1, i2 = img.chunk(2, dim=0)
        
        i1 = _cut_out(i1, n_holes=n_holes, length=length)
        i2 = _cut_out(i2, n_holes=n_holes, length=length)
        
        img = torch.cat([i1, i2], dim=0)
    
    else:
        img = _cut_out(img, n_holes=n_holes, length=length)
    
    return img


def ldr_f2u(x, minv=-1.0, maxv=1.0):
    '''
    from float to uint8
    '''
    x = 255 * (x - minv) / (maxv - minv)
    # x = (x - minv) / (maxv - minv)
    x = x.astype('uint8')
    return x


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    image = image / 255. ## image in range (0, 1)
    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 65.481 * r + 128.553 * g + 24.966 * b + 16.0
    cb: torch.Tensor = -37.797 * r + -74.203 * g + 112.0 * b + 128.0
    cr: torch.Tensor = 112.0 * r + -93.786 * g + -18.214 * b + 128.0

    return torch.stack((y, cb, cr), -3)


def prepare_qat(model):
    ## fuse model
    model.module.fuse_model()
    ## qconfig and qat-preparation & per-channel quantization
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    # model.qconfig = torch.quantization.QConfig(
    #     activation=torch.quantization.FakeQuantize.with_args(
    #         observer=torch.quantization.MinMaxObserver, 
    #         quant_min=-128,
    #         quant_max=127,
    #         qscheme=torch.per_tensor_symmetric,
    #         dtype=torch.qint8,
    #         reduce_range=False),
    #     weight=torch.quantization.FakeQuantize.with_args(
    #         observer=torch.quantization.MinMaxObserver, 
    #         quant_min=-128, 
    #         quant_max=+127, 
    #         dtype=torch.qint8, 
    #         qscheme=torch.per_tensor_symmetric, 
    #         reduce_range=False)
    # )
    model = torch.quantization.prepare_qat(model, inplace=True)
    return model


def import_module(name):
    return importlib.import_module(name)


def calc_psnr(sr, hr):
    sr, hr = sr.double(), hr.double()
    diff = (sr - hr) / 255.00
    mse  = diff.pow(2).mean()
    psnr = -10 * math.log10(mse)                    
    return float(psnr)


def Gaussian_noise_layer(input_layer, std):
    noise = std * torch.randn_like(input_layer).to(input_layer)
    return input_layer + noise.detach()


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def calc_ssim(sr, hr):
    ssim_val = ssim(sr, hr, size_average=True)
    return float(ssim_val)
    

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor


def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content


class ExperimentLogger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_stat_dict():
    stat_dict = {
        'epochs': 0,
        'losses': [],
        'ema_loss': 0.0,
        'CCA-US': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'US-CASE': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        },
        'US1K_23': {
            'psnrs': [],
            'ssims': [],
            'best_psnr': {
                'value': 0.0,
                'epoch': 0
            },
            'best_ssim': {
                'value': 0.0,
                'epoch': 0
            }
        }
    }
    return stat_dict


if __name__ == '__main__':
    timestamp = cur_timestamp_str()
    print(timestamp)