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
