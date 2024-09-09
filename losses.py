import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch.fft as FFT
import torch
import math
from piq import SSIMLoss, MultiScaleSSIMLoss, VIFLoss
from math import pi
import warnings
import cv2
from PIL import Image
from einops import rearrange
from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor


class SemanticLoss(nn.Module):
    def __init__(self, criterion='l1', N_patches=3):
        super(SemanticLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.medmodel = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.medmodel.from_pretrained()
        self.medmodel.to(self.device)
        self.processor = MedCLIPProcessor()
    
        self.N_patches = N_patches

    def createNRandompatches(self, img1, img2, N, patch_size, clipsize=224):
        myw = img1.size()[2]
        myh = img1.size()[3]
        patches1 = []
        patches2 = []

        for i in range(N):
            xcoord = int(torch.randint(myw - patch_size, ()))
            ycoord = int(torch.randint(myh - patch_size, ()))
            patches1.append(img1[:, :, xcoord:xcoord+patch_size, ycoord:ycoord+patch_size])
            patches2.append(img2[:, :, xcoord:xcoord+patch_size, ycoord:ycoord+patch_size])
        return patches1, patches2

    def __call__(self, x, y, batch_tokens):
        
        
        x = x.unsqueeze(0).to(self.device)
        y = y.unsqueeze(0).to(self.device)
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        

        #The resize operation on tensor.
        patches_x = [torch.nn.functional.interpolate(x, mode='bicubic', size=(224, 224), align_corners=True)]  
        patches_y = [torch.nn.functional.interpolate(y, mode='bicubic', size=(224, 224), align_corners=True)]

        ## patch based clip loss
        if self.N_patches > 1:
            patches_x2, patches_y2 = self.createNRandompatches(x, y, self.N_patches-1, 224)
            patches_x += patches_x2
            patches_y += patches_y2
        
        loss = torch.cuda.FloatTensor(1).fill_(0)
        with torch.no_grad():
            outputs = self.processor.tokenizer(text=[batch_tokens],return_tensors="pt",padding=True)
            text_features = self.medmodel.encode_text(outputs['token_type_ids'],outputs['attention_mask'])
            patch_factor = (1.0 / float(self.N_patches))
            for i in range(len(patches_x)):
                x_clip = self.medmodel.encode_image(patches_x[i])
                y_clip = self.medmodel.encode_image(patches_y[i]) #shape [1,512]
    
            x_clip /= x_clip.norm(dim=-1,keepdim=True)
            y_clip /= y_clip.norm(dim=-1,keepdim=True)

            text_features /= text_features.norm(dim=-1,keepdim=True)

            logits_per_image1 = x_clip @ text_features.T
            logits_per_image2 = y_clip @ text_features.T

            loss += torch.abs(logits_per_image1[0]-logits_per_image2[0])*patch_factor

        return loss




class WL1(torch.nn.Module):
    """Weighted L1 loss."""
    def __init__(self, ):
        super(WL1, self).__init__()

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff = torch.mean(torch.abs(diff), dim=[1,2,3], keepdim=True)
        weight = torch.softmax(torch.log(diff.detach()), dim=0)
        # diff = torch.mean(torch.abs(diff), dim=[1,2,3], keepdim=True)
        loss = torch.sum(weight * torch.abs(diff), dim=[0,1,2,3])

        return loss
    

class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, patch_size=0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)
        self.ps = patch_size

    def forward(self, pred, target):
        
        if self.ps > 0:
            B, C, H, W = pred.size()

            grid_height, grid_width = H // self.ps, W//self.ps
            pred_patch = rearrange(
                pred, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw", 
                gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps) 
            
            target_patch = rearrange(
                target, "n c (gh bh) (gw bw) -> n (c gh gw) bh bw", 
                gh=grid_height, gw=grid_width, bh=self.ps, bw=self.ps) 
            
            pred_fft = torch.fft.rfft2(pred_patch, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target_patch, dim=(-2, -1))

            pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        
        else:
            pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target, dim=(-2, -1))

            pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)
    
    
class PSNR:
    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, max_value):
        mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
        PSNR = 20 * np.log10(max_value / np.sqrt(mse))
        return PSNR


class VGG19bn_relu(torch.nn.Module):
    def __init__(self):
        super(VGG19bn_relu, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg19_bn(weights='VGG19_BN_Weights.IMAGENET1K_V1')
        cnn = cnn.to(self.device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(3):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(3, 6):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(6, 10):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(10, 13):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(13, 17):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(17, 20):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(20, 23):
           self.relu3_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(26, 30):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(30, 33):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(33, 36):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(36, 39):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(39, 43):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(43, 46):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(46, 49):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(49, 52):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class PerceptualLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], resize=False, criterion='l1'):
        super(PerceptualLoss, self).__init__()
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'sl1':
            self.criterion = nn.SmoothL1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(criterion))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.add_module('vgg', VGG19_relu())
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        self.weights = weights
        self.resize = resize
        self.transformer = torch.nn.functional.interpolate

    def att(self, in_feat):
        return torch.sigmoid(torch.mean(in_feat, 1).unsqueeze(1))

    def __call__(self, x, y):
        if self.resize:
            x = self.transformer(x, mode='bicubic', size=(224, 224), align_corners=True)
            y = self.transformer(y, mode='bicubic', size=(224, 224), align_corners=True)
        
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x)) / self.std.to(x)
        y = (y - self.mean.to(y)) / self.std.to(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0.0
        if False:
            loss += self.weights[0]*self.criterion(x_vgg['relu1_1']*self.att(y_vgg['relu1_1']), y_vgg['relu1_1']*self.att(y_vgg['relu1_1']))
            loss += self.weights[1]*self.criterion(x_vgg['relu2_1']*self.att(y_vgg['relu2_1']), y_vgg['relu2_1']*self.att(y_vgg['relu2_1']))
            loss += self.weights[2]*self.criterion(x_vgg['relu3_1']*self.att(y_vgg['relu3_1']), y_vgg['relu3_1']*self.att(y_vgg['relu3_1']))
            loss += self.weights[3]*self.criterion(x_vgg['relu4_1']*self.att(y_vgg['relu4_1']), y_vgg['relu4_1']*self.att(y_vgg['relu4_1']))
            loss += self.weights[4]*self.criterion(x_vgg['relu5_1']*self.att(y_vgg['relu5_1']), y_vgg['relu5_1']*self.att(y_vgg['relu5_1']))
        else:
            loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
            loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
            loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
            loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
            loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return loss


# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):

    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError(f"Input images should have the same dimensions, but got {X.shape} and {Y.shape}.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError(f"Input images should have the same dtype, but got {X.type()} and {Y.type()}.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (
        2 ** 4
    ), "Image size should be larger than %d due to the 4 downsamplings in ms-ssim" % ((win_size - 1) * (2 ** 4))

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = X.new_tensor(weights)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
        weighted=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIMLoss, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.weighted = weighted

    def forward(self, X, Y):
        if self.weighted:
            diff = torch.add(X, -Y)
            weight = torch.mean(torch.abs(diff), dim=[1,2,3], keepdim=True)
            weight = torch.softmax(weight.detach(), dim=0)
            out = ssim(
                        X,
                        Y,
                        data_range=self.data_range,
                        size_average=self.size_average,
                        win=self.win,
                        K=self.K,
                        nonnegative_ssim=self.nonnegative_ssim,
                    )
            out = (1 - out) * weight
            return out.sum()
        else:
            out = ssim(
                X,
                Y,
                data_range=self.data_range,
                size_average=self.size_average,
                win=self.win,
                K=self.K,
                nonnegative_ssim=self.nonnegative_ssim,
            )
            return (1.0 - out)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        weights=None,
        K=(0.01, 0.03),
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )


class EASLoss(nn.Module):
    ''' edge aware smoothness loss '''
    def __init__(self):
        super(EASLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def gradient_xy(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        gy = img[:,:,:,:-1] - img[:,:,:,1:]
        return gx, gy
    
    def forward(self, pred, gt):
        pred_grad_x, pred_grad_y = self.gradient_xy(pred)
        gt_grad_x, gt_grad_y = self.gradient_xy(gt)

        weights_x = torch.exp(-torch.mean(torch.abs(gt_grad_x), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(gt_grad_y), 1, keepdim=True))

        smoothness_x = torch.abs(pred_grad_x) * weights_x
        smoothness_y = torch.abs(pred_grad_y) * weights_y

        loss = (torch.mean(smoothness_x) + torch.mean(smoothness_y))
        
        # loss = self.criterion(pred_grad_x, gt_grad_x) + self.criterion(pred_grad_y, gt_grad_y)

        return loss


class MultiscaleEASLoss(nn.Module):
    ''' multiscale edge aware smoothness loss '''
    def __init__(self, scales=1):
        super(MultiscaleEASLoss, self).__init__()
        self.scales = scales
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        # self.downsample = Interpolate(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.weights = [1.0, 1.0/2, 1.0/4, 1.0/8, 1.0/8]
        self.weights = self.weights[:scales]

    def gradient_xy(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        gy = img[:,:,:,:-1] - img[:,:,:,1:]
        return gx, gy
    
    def forward(self, pred, gt):
        loss = 0
        if self.scales > 1:
            for i in range(len(self.weights)):
                pred_grad_x, pred_grad_y = self.gradient_xy(pred)
                gt_grad_x, gt_grad_y = self.gradient_xy(gt)
                weights_x = torch.exp(-torch.mean(torch.abs(gt_grad_x), 1, keepdim=True))
                weights_y = torch.exp(-torch.mean(torch.abs(gt_grad_y), 1, keepdim=True))
                smoothness_x = torch.abs(pred_grad_x) * weights_x
                smoothness_y = torch.abs(pred_grad_y) * weights_y
                loss += (torch.mean(smoothness_x) + torch.mean(smoothness_y)) * self.weights[i]
                if i != len(self.weights) - 1:
                    pred = self.downsample(pred)
                    gt = self.downsample(gt)
        else:
            pred_grad_x, pred_grad_y = self.gradient_xy(pred)
            gt_grad_x, gt_grad_y = self.gradient_xy(gt)
            weights_x = torch.exp(-torch.mean(torch.abs(gt_grad_x), 1, keepdim=True))
            weights_y = torch.exp(-torch.mean(torch.abs(gt_grad_y), 1, keepdim=True))
            smoothness_x = torch.abs(pred_grad_x) * weights_x
            smoothness_y = torch.abs(pred_grad_y) * weights_y
            loss = (torch.mean(smoothness_x) + torch.mean(smoothness_y))
        return loss


class MultiscaleGradientLoss(nn.Module):
    ''' multiscale edge aware smoothness loss '''
    def __init__(self, scales=1, rec_loss_type='l1'):
        super(MultiscaleGradientLoss, self).__init__()
        self.scales = scales
        if rec_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif rec_loss_type == 'sl1':
            self.criterion = nn.SmoothL1Loss()
        elif rec_loss_type == 'l2':
            self.criterion = nn.MSELoss()
        elif rec_loss_type == 'cbl1':
            self.criterion = L1_Charbonnier_loss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(rec_loss_type))
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        self.weights = [1.0, 1.0/2, 1.0/4, 1.0/8, 1.0/8]
        self.weights = self.weights[:scales]
        # self.fx = torch.Tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]).expand(1,3,3,3)
        # self.fy = torch.Tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]).expand(1,3,3,3)

    def gradient_xy(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        gy = img[:,:,:,:-1] - img[:,:,:,1:]
        # padding = nn.ReflectionPad2d(1)
        # gx = padding(F.conv2d(img, self.fx.to(img), padding=0))  
        # gy = padding(F.conv2d(img, self.fy.to(img), padding=1))
        return gx, gy
    
    def forward(self, pred, gt):
        loss = 0
        if self.scales > 1:
            for i in range(len(self.weights)):
                pred_grad_x, pred_grad_y = self.gradient_xy(pred)
                gt_grad_x, gt_grad_y = self.gradient_xy(gt)
                loss = (self.criterion(pred_grad_x, gt_grad_x) + self.criterion(pred_grad_y, gt_grad_y)) * self.weights[i]
                if i != len(self.weights) - 1:
                    pred = self.downsample(pred)
                    gt = self.downsample(gt)
        else:
            pred_grad_x, pred_grad_y = self.gradient_xy(pred)
            gt_grad_x, gt_grad_y = self.gradient_xy(gt)
            loss = self.criterion(pred_grad_x, gt_grad_x) + self.criterion(pred_grad_y, gt_grad_y)
        return loss


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # if no margin assigned, use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor-pos, 2, dim=1).view(-1)
            an_dist = torch.norm(anchor-neg, 2, dim=1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners,
                          recompute_scale_factor=True)
        return out


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        loss = torch.mean(torch.sqrt( diff * diff + self.eps))
        # loss = torch.mean(torch.sqrt(torch.clamp(diff * diff, min=self.eps)))
        return loss


class Info_Weighted_L1_loss(nn.Module):
    def __init__(self, win=None, win_size=11, win_sigma=1.5):
        super(Info_Weighted_L1_loss, self).__init__()
        self.eps = 1e-8
        self.win = win
        self.win_size = win_size
        self.win_sigma = win_sigma

    def fspecial_gauss_1d(self, size, sigma):
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.unsqueeze(0).unsqueeze(0)

    def gaussian_filter(self, input, win):
        assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
        if len(input.shape) == 4:
            conv = F.conv2d
        elif len(input.shape) == 5:
            conv = F.conv3d
        else:
            raise NotImplementedError(input.shape)
        C = input.shape[1]
        out = input
        padding = nn.ReflectionPad2d(self.win_size//2)
        out = padding(out)
        for i, s in enumerate(input.shape[2:]):
            if s >= win.shape[-1]:
                out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
            else:
                warnings.warn(f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}")
        return out

    def forward(self, pred, gt):
        # X = pred.clone()
        Y = gt.clone().detach()
        if Y.shape[1] > 1:
            # X = torch.unsqueeze((X[:,0]*0.299 + X[:,1]*0.587 + X[:,2]*0.114), dim=1)
            Y = torch.unsqueeze((Y[:,0]*0.299 + Y[:,1]*0.587 + Y[:,2]*0.114), dim=1)
        if self.win is None:
            win = self.fspecial_gauss_1d(self.win_size, self.win_sigma)
            win = win.repeat([Y.shape[1]] + [1] * (len(Y.shape) - 1))
            win = win.to(Y.device, dtype=Y.dtype)
        # mu1 = self.gaussian_filter(X, win)
        mu2 = self.gaussian_filter(Y, win)
        # mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        # sigma1_sq = self.gaussian_filter(X * X, win) - mu1_sq
        sigma2_sq = self.gaussian_filter(Y * Y, win) - mu2_sq
        # weight, _ = torch.max(torch.cat([sigma1_sq, sigma2_sq], dim=1), dim=1, keepdim=True)
        # weight = torch.sigmoid(torch.abs(sigma2_sq))
        weight = torch.abs(sigma2_sq)
        weight = weight / weight.sum()

        weighted_diff = torch.abs(pred - gt) * weight
        loss = torch.sum(weighted_diff) / (Y.shape[0]*Y.shape[1])
        return loss


class MultiscaleRecLoss(nn.Module):
    def __init__(self, scales=1, rec_loss_type='l2'):
        super(MultiscaleRecLoss, self).__init__()
        self.scales = scales
        if rec_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif rec_loss_type == 'sl1':
            self.criterion = nn.SmoothL1Loss()
        elif rec_loss_type == 'l2':
            self.criterion = nn.MSELoss()
        elif rec_loss_type == 'cbl1':
            self.criterion = L1_Charbonnier_loss()
        elif rec_loss_type == 'iwl1':
            self.criterion = Info_Weighted_L1_loss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(rec_loss_type))
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        # self.downsample = Interpolate(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.weights = [1.0, 1.0/2, 1.0/4, 1.0/8, 1.0/8]
        self.weights = self.weights[:scales]

    def forward(self, pred, gt, mask=None):
        loss = 0
        if self.scales > 1:
            if mask is not None:
                for i in range(len(self.weights)):
                    loss += self.weights[i] * (self.criterion(pred*mask, gt*mask))
                    if i != len(self.weights) - 1:
                        pred = self.downsample(pred)
                        gt = self.downsample(gt)
                        mask = self.downsample(mask)
            else:
                for i in range(len(self.weights)):
                    loss += self.weights[i] * self.criterion(pred, gt)
                    if i != len(self.weights) - 1:
                        pred = self.downsample(pred)
                        gt = self.downsample(gt)
        else:
            if mask is not None:
                loss = self.criterion(pred*mask, gt*mask)
            else:
                loss = self.criterion(pred, gt)
        return loss


class SpectralConvergenceLoss(nn.Module):
    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()
        
    def forward(self, amp_pred, amp_gt):
        x = torch.norm(amp_gt - amp_pred, p='fro')
        y = torch.norm(amp_gt, p='fro')
        return x / y 


class LogFFTAmplitudeLoss(nn.Module):
    def __init__(self):
        super(LogFFTAmplitudeLoss, self).__init__()
        self.eps = 1e-6
        
    def forward(self, amp_pred, amp_gt):
        # log_amp_pred = torch.log(amp_pred + self.eps)
        # log_amp_gt = torch.log(amp_gt + self.eps)
        log_amp_pred = torch.log(torch.clamp(amp_pred, min=self.eps))
        log_amp_gt = torch.log(torch.clamp(amp_gt, min=self.eps))
        outputs = F.l1_loss(log_amp_pred, log_amp_gt)
        return outputs
        

# Frequency loss
class MultiscaleFFTloss(torch.nn.Module):
    def __init__(self, scales=1):
        super(MultiscaleFFTloss, self).__init__()
        self.scales = scales
        self.eps = 1e-8
        self.criterion = torch.nn.L1Loss(reduction='mean')
        self.spectral_convergenge_loss = SpectralConvergenceLoss()
        self.log_fft_amplitude_loss = LogFFTAmplitudeLoss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        self.weights = [1.0, 1.0/2, 1.0/4, 1.0/8, 1.0/8]
        self.weights = self.weights[:scales]
    
    def extract_amp_phase(self, fft_im):
        ### fft_im: size should be bx3xhxwx2  for old version torch.rfft
        # fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
        # fft_amp = torch.sqrt(torch.clamp(fft_amp, min=self.eps))
        # fft_pha = torch.atan2(fft_im[:,:,:,:,1], torch.clamp(fft_im[:,:,:,:,0], min=self.eps))
        # fft_amp = torch.sqrt(fft_amp)
        # fft_pha = torch.atan2(fft_im[:,:,:,:,1], fft_im[:,:,:,:,0])
        ### fft_im: size should be bx3xhxw  for new version torch.fft.rfft
        ## some problem of the pytorch definition torch.abs() and torch.angle()
        # fft_amp = torch.abs(fft_im)
        # fft_pha = torch.angle(fft_im)
        fft_amp = fft_im.abs()
        fft_pha = fft_im.angle()
        ## my method of calculating amplitude and phase
        # fft_amp = torch.sqrt((fft_im.real ** 2 + fft_im.imag ** 2 + self.eps)) 
        # fft_pha = fft_im.imag.atan2(fft_im.real + self.eps) 
        # fft_im.real[fft_im.real == 0] = self.eps
        # fft_amp = torch.sqrt((fft_im.real ** 2 + fft_im.imag ** 2)) 
        # fft_pha = fft_im.imag.atan2(fft_im.real) 
        # fft_amp = torch.sqrt(torch.clamp(fft_im.real ** 2 + fft_im.imag ** 2, min=self.eps))
        # fft_pha = fft_im.imag.atan2(torch.clamp(fft_im.real, min=self.eps))
        return fft_amp, fft_pha

    def forward(self, pred, gt):
        loss = 0
        if self.scales > 1:
            for i in range(len(self.weights)):
                ### output: size is bx3xhxw  for new version torch.fft.rfft
                fft_pred = FFT.fft2(pred)
                fft_gt = FFT.fft2(gt)

                ### output: size is bx3xhxwx2  for old version torch.rfft
                # fft_pred = torch.rfft(pred, signal_ndim=2, onesided=False, normalized=True)
                # fft_gt = torch.rfft(gt, signal_ndim=2, onesided=False, normalized=True)

                amp_pred, pha_pred = self.extract_amp_phase(fft_pred)
                amp_gt, pha_gt = self.extract_amp_phase(fft_gt)
                loss += 0.5 * self.weights[i] * (self.criterion(amp_pred, amp_gt) + self.criterion(pha_pred, pha_gt))
                # loss += self.weights[i] * (self.criterion(pha_pred, pha_gt) + self.spectral_convergenge_loss(amp_pred, amp_gt) + self.log_fft_amplitude_loss(amp_pred, amp_gt)) / 3.0

                if i != len(self.weights) - 1:
                    pred = self.downsample(pred)
                    gt = self.downsample(gt)
        else:
            ### for old version torch.rfft
            # fft_pred = torch.rfft(pred, signal_ndim=2, onesided=False, normalized=True)
            # fft_gt = torch.rfft(gt, signal_ndim=2, onesided=False, normalized=True)

            ### for new version torch.fft.rfft
            fft_pred = FFT.fft2(pred)
            fft_gt = FFT.fft2(gt)

            # print('fft pred size: ', fft_pred.size())
            # a = fft_pred
            # print(a.mean())
            # print(a, a.real, a.imag)

            # loss += self.criterion(fft_pred, fft_gt)
            # loss += self.criterion(fft_pred.real, fft_gt.real) + self.criterion(fft_pred.imag, fft_gt.imag)

            amp_pred, pha_pred = self.extract_amp_phase(fft_pred)
            amp_gt, pha_gt = self.extract_amp_phase(fft_gt)
            # loss += self.criterion(amp_pred, amp_gt)
            # print(amp_pred.mean())
            # print(amp_gt.mean())
            # print(pha_pred.mean())
            # print(pha_gt.mean())
            loss += self.criterion(amp_pred)
            # loss += 0.5 * (self.criterion(amp_pred, amp_gt) + self.criterion(pha_pred, pha_gt))
            # loss += 0.5 * (self.log_fft_amplitude_loss(amp_pred, amp_gt) + self.criterion(amp_pred, amp_gt))
            # loss += 0.5 * (self.spectral_convergenge_loss(amp_pred, amp_gt) + self.log_fft_amplitude_loss(amp_pred, amp_gt))
            # loss += (self.criterion(pha_pred, pha_gt) + self.spectral_convergenge_loss(amp_pred, amp_gt) + self.log_fft_amplitude_loss(amp_pred, amp_gt)) / 3.0
        return loss


class FFTLossqqqq(torch.nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.eps = 1e-8
        self.criterion = torch.nn.L1Loss(reduction='mean')
        self.size_average = True
    
    def extract_amp_phase(self, fft_im):
        ### fft_im: size should be bx3xhxw  for new version torch.fft.rfft
        ## some problem of the pytorch definition torch.abs() and torch.angle()
        fft_amp = torch.abs(fft_im)
        fft_pha = torch.angle(fft_im)
        # fft_amp = fft_im.abs()
        # fft_pha = fft_im.angle()
        ## my method of calculating amplitude and phase
        # fft_amp = torch.sqrt((fft_im.real ** 2 + fft_im.imag ** 2 + self.eps)) 
        # fft_pha = fft_im.imag.atan2(fft_im.real + self.eps) 
        # fft_im.real[fft_im.real == 0] = self.eps
        # fft_amp = torch.sqrt((fft_im.real ** 2 + fft_im.imag ** 2)) 
        # fft_pha = fft_im.imag.atan2(fft_im.real) 
        # fft_amp = torch.sqrt(torch.clamp(fft_im.real ** 2 + fft_im.imag ** 2, min=self.eps))
        # fft_pha = fft_im.imag.atan2(torch.clamp(fft_im.real, min=self.eps))
        return fft_amp, fft_pha

    def forward(self, pred, gt):
        if True:
            loss = 0

            ### for new version torch.fft.rfft
            fft_pred = FFT.fft2(pred, norm="ortho")
            fft_gt = FFT.fft2(gt, norm="ortho")

            # loss += self.criterion(fft_pred, fft_gt)
            # loss += self.criterion(fft_pred.real, fft_gt.real) + self.criterion(fft_pred.imag, fft_gt.imag)

            amp_pred, pha_pred = self.extract_amp_phase(fft_pred)
            amp_gt, pha_gt = self.extract_amp_phase(fft_gt)

            loss += self.criterion(amp_pred, amp_gt)
            loss += self.criterion(pha_pred, pha_gt)
            # loss += 0.5 * (self.criterion(amp_pred, amp_gt) + self.criterion(pha_pred, pha_gt))
            return loss
        else:
            L =  (pred - gt)
            L_fft = FFT.rfft2(L, norm="forward")
            L_fft = L_fft.abs() + L_fft.angle()
            return torch.mean(L_fft) if self.size_average else torch.sum(L_fft)


def mse_fft(input, target, size_average=True):
    L = (input - target)
    L_fft = torch.rfft(L, 2, onesided=False, normalized=True).to(self.device)
    L_fft = L_fft ** 2
    return torch.mean(L_fft) if size_average else torch.sum(L_fft)


class AngularLoss(nn.Module):
    def __init__(self, shrink=True, eps=1e-6):
        super(AngularLoss, self).__init__()
        self.eps = eps
        self.shrink = shrink

    def forward(self, pred, gt):
        cossim = torch.clamp(torch.sum(pred * gt, dim=1) / (torch.norm(pred, dim=1) * torch.norm(gt, dim=1) + 1e-9), -1, 1.)
        if self.shrink:
            angle = torch.acos(cossim * (1-self.eps))
        else:
            angle = torch.acos(cossim)
        
        angle = angle * 180 / math.pi
        error = torch.mean(angle)
        return error


class UQI(nn.Module):
    def __init__(self):
        super(UQI, self).__init__()
    
    def forward(self, pred, gt):
        E_pred = torch.mean(pred, dim=(1,2))
        E_pred2 = torch.mean(pred * pred, dim=(1,2))
        E_gt = torch.mean(gt, dim=(1,2))
        E_gt2 = torch.mean(gt * gt, dim=(1,2))
        E_predgt = torch.mean(pred * gt, dim=(1,2))

        var_pred = E_pred2 - E_pred * E_pred
        var_gt = E_gt2 - E_gt * E_gt
        cov_predgt = E_predgt - E_pred * E_gt

        return 1 - torch.mean(4 * cov_predgt * E_pred * E_gt / (var_pred + var_gt) / (E_pred**2 + E_gt**2))


def DCLoss(img, patch_size=5):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
    dc = maxpool(1-img[:, None, :, :, :])
    
    # target = torch.FloatTensor(dc.shape).zero_().cuda(opt.gpu_ids[0])
    target = torch.zeros_like(dc)
     
    loss = nn.L1Loss(reduction='sum')(dc, target)
    return -loss


def BCLoss(img, patch_size):
    """
    calculating bright channel of image, the image shape is of N*C*W*H
    """
    patch_size = 15
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
    bc = maxpool(img[:, None, :, :, :])
    
    target = torch.ones_like(bc)
    loss = nn.L1Loss(reduction='sum')(bc, target)
    return loss


class BrightChannelLoss(nn.Module):
    def __init__(self, kernel_size=15):
        super(BrightChannelLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)

    def forward(self, x):
        # x : (B, 3, H, W), in [0, 1]
        H, W = x.size()[2], x.size()[3]

        # Miaximum among three channels
        x, _ = x.max(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        x = self.unfold(x)  # (B, k*k, H*W)
        x = x.unsqueeze(1)  # (B, 1, k*k, H*W)

        # Maximum in (k, k) patch
        bright_map, _ = x.max(dim=2, keepdim=False)  # (B, 1, H*W)
        x = bright_map.view(-1, 1, H, W)

        return x.clamp(min=0.0, max=0.1)

    def __call__(self, fake, real):
        real_map = self.forward(real)
        fake_map = self.forward(fake)
        return self.loss(real_map, fake_map)


class DarkChannelLoss(nn.Module):
    def __init__(self, kernel_size=15):
        super(DarkChannelLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.kernel_size = kernel_size
        self.pad_size = (self.kernel_size - 1) // 2
        self.unfold = nn.Unfold(self.kernel_size)

    def forward(self, x):
        # x : (B, 3, H, W), in [0, 1]
        H, W = x.size()[2], x.size()[3]

        # Minimum among three channels
        x, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W)
        x = nn.ReflectionPad2d(self.pad_size)(x)  # (B, 1, H+2p, W+2p)
        x = self.unfold(x)  # (B, k*k, H*W)
        x = x.unsqueeze(1)  # (B, 1, k*k, H*W)

        # Minimum in (k, k) patch
        dark_map, _ = x.min(dim=2, keepdim=False)  # (B, 1, H*W)
        x = dark_map.view(-1, 1, H, W)

        # Count Zeros
        #y0 = torch.zeros_like(x)
        #y1 = torch.ones_like(x)
        #x = torch.where(x < 0.1, y0, y1)
        #x = torch.sum(x)
        #x = int(H * W - x)
        return x.clamp(min=0.0, max=0.1)

    def __call__(self, real, fake):
        real_map = self.forward(real)
        fake_map = self.forward(fake)
        return self.loss(real_map, fake_map)


class DLoss(nn.Module):
    def __init__(self, opt):
        super(DLoss, self).__init__()
        self.opt = opt
        self.criterionGAN = GANLoss(opt.GAN_type, tensor=torch.cuda.FloatTensor)

    def discriminate(self, netD, fake_image, real_image):
        fake_and_real_img = torch.cat([fake_image, real_image], dim=0)
        discriminator_out = netD(fake_and_real_img)
        fake_feats, fake_preds, real_feats, real_preds = divide_pred(discriminator_out)

        return fake_feats, fake_preds, real_feats, real_preds

    def forward(self, pred_G, gt, netD):
        _, pred_fake, _, pred_real = self.discriminate(netD, pred_G, gt)
        loss_gan  = self.criterionGAN(pred_real, pred_fake, target_is_real=None, for_discriminator=True) * self.opt.lambda_gan

        return loss_gan


# Take the prediction of fake and real images from the combined batch
def divide_pred(pred):
    fake_feats = []
    fake_preds = []
    real_feats = []
    real_preds = []
    for p in pred[0]:
        fake_feats.append(p[:p.size(0)//2])
        real_feats.append(p[p.size(0)//2:])
    for p in pred[1]:
        fake_preds.append(p[:p.size(0)//2])
        real_preds.append(p[p.size(0)//2:])

        return fake_feats, fake_preds, real_feats, real_preds


class GANFeatLoss(nn.Module):
    def __init__(self, criterion='l1', opt=None):
        super(GANFeatLoss, self).__init__()
        self.opt = opt
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError('Unexpected criterion type {}'.format(criterion))
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    def forward(self, feat_fake, feat_real, label=None):
        loss = self.FloatTensor(1).fill_(0)
        num_D = len(feat_fake)
        for j in range(num_D):
            loss += self.criterion(feat_fake[j], feat_real[j].detach()) / num_D
        
        return loss


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        elif gan_mode == 'rahinge':
            pass
        elif gan_mode == 'rals':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        if self.gan_mode == 'original': # cross entropy loss
            if for_real:
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(real_preds, target_tensor)
                return loss
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(fake_preds, target_tensor)
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'ls':   
            if for_real:  
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                return F.mse_loss(real_preds, target_tensor)
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                return F.mse_loss(fake_preds, target_tensor)
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'hinge':
            if for_real:
                loss = -torch.mean(real_preds)
                return loss
            elif for_fake:
                loss = -torch.mean(fake_preds)
                return loss
            elif for_discriminator:
                loss_fake = torch.mean(torch.nn.ReLU()(1.0 + fake_preds))
                loss_real = torch.mean(torch.nn.ReLU()(1.0 - real_preds))
                loss = (loss_fake + loss_real) / 2.
                return loss
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")
        elif self.gan_mode == 'rahinge':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = (torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff))) / 1.
                return loss
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = (torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))) / 1.
                return loss
        elif self.gan_mode == 'rals':
            if for_discriminator:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = (torch.mean((r_f_diff - 1) ** 2) + torch.mean((f_r_diff + 1) ** 2)) / 1.
                return loss
            else:
                ## difference between real and fake
                r_f_diff = real_preds - torch.mean(fake_preds)
                ## difference between fake and real
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = (torch.mean((r_f_diff + 1) ** 2) + torch.mean((f_r_diff - 1) ** 2)) / 1.
                return loss
        else:
            # wgan
            if for_real:
                if target_is_real:
                    return -real_preds.mean()
                else:
                    return real_preds.mean()
            elif for_fake:
                if target_is_real:
                    return -fake_preds.mean()
                else:
                    return fake_preds.mean()
            else:
                raise NotImplementedError("nither for real_preds nor for fake_preds")

    def __call__(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        ## computing loss is a bit complicated because |input| may not be
        ## a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(real_preds, list):
            loss = 0
            for (pred_real_i, pred_fake_i) in zip(real_preds, fake_preds):
                if isinstance(pred_real_i, list):
                    pred_real_i = pred_real_i[-1]
                if isinstance(pred_fake_i, list):
                    pred_fake_i = pred_fake_i[-1]

                loss_tensor = self.loss(pred_real_i, pred_fake_i, target_is_real, for_real, for_fake, for_discriminator)

                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(real_preds)
        else:
            return self.loss(real_preds, target_is_real, for_discriminator)