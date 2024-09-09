# from model import common
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import torch.fft as fft
import math
import torch.nn.init as init

def create_model(args):
    return M2Trans(args)


class M2Trans(nn.Module):
    def __init__(self, args):
        super(M2Trans, self).__init__()

        # Params
        n_feats = args.n_feats  ### 64 ###
        self.scale = args.scale
        self.window_sizes = [8, 16, 32]
        self.rgb_range = args.rgb_range
        self.n_blocks = args.n_blocks

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std,-1)
        self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)


        self.head = nn.Conv2d(args.colors, n_feats, kernel_size=3, bias=True, stride=1, padding=1, padding_mode='reflect')

        self.body = nn.ModuleList(
            [CFTM(nf=n_feats, block_size=8, halo_size=1, norm=True) for i in range(args.n_blocks)]
        )
        
        if self.scale == 4:
            self.tail= nn.Sequential(
                nn.Conv2d(n_feats*1, n_feats*4, kernel_size=1, bias=True, stride=1, padding=0, padding_mode='reflect'),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(n_feats, n_feats*4, kernel_size=1, bias=True, stride=1, padding=0, padding_mode='reflect'),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(n_feats, 3, kernel_size=3, bias=False, stride=1, padding=1, padding_mode='reflect'),
            )
        else:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats*1, n_feats*self.scale*self.scale, kernel_size=1, bias=True, stride=1, padding=0, padding_mode='reflect'),
                nn.PixelShuffle(self.scale),
                nn.GELU(),
                nn.Conv2d(n_feats, 3, kernel_size=3, bias=False, stride=1, padding=1, padding_mode='reflect'),
            )
            
    def forward(self, x):
        
        H, W = (x.shape[2], x.shape[3])
        x = self.check_image_size(x) 
        
        res = self.head(x) 
        
        x = res 

        for blkid in range(self.n_blocks):
            x = self.body[blkid](x)  
        
        x = res + x 

        x = self.tail(x)  

        x = torch.clamp(x, min=0.0, max=self.rgb_range) 

        return x[:, :, 0:H*self.scale, 0:W*self.scale] 

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0] 
        for i in range(1, len(self.window_sizes)): 
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i]) 
        mod_pad_h = (wsize - h % wsize) % wsize 
        mod_pad_w = (wsize - w % wsize) % wsize 
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')  
        return x
    
    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

class CFTM(nn.Module):
    def __init__(self, nf, block_size=8, halo_size=2, norm=True):
        super(CFTM, self).__init__()
        self.is_norm = norm

        self.attn1 = TBlock(nf//4, block_size=8, halo_size=1, num_heads=1, bias=False)
        self.attn2 = TBlock(nf*1, block_size=8, halo_size=1, num_heads=1, bias=False)
        self.attn3 = TBlock(nf*4, block_size=8, halo_size=1, num_heads=1, bias=False)
        self.attn4 = TBlock(nf*4, block_size=8, halo_size=1, num_heads=1, bias=False)
        
        self.feed_forward = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.norm = nn.InstanceNorm2d(nf)
        
        self.down = DWT()
        self.up = IWT()

    def forward(self, x):
        if self.is_norm:
            
            x1 = self.norm(x)
            
            x1, x2, x3, x4 = torch.chunk(x1, 4, dim=1)
            
            x1 = self.attn1(x1) + x1
            
            x2 = (x2 + x1) / 2.0
            x2r = x2
            x2 = self.down(x2)
            x2 = self.attn2(x2) 
            x2 = self.up(x2) + x2r

            x3 = (x3 + x2) / 2.0
            x3r = x3
            x3 = self.down(x3)
            x3 = self.down(x3)
            x3 = self.attn3(x3)
            x3 = self.up(x3)
            x3 = self.up(x3) + x3r

            x4 = (x4 + x3) / 2.0
            x4r = x4
            x4 = self.down(x4)
            x4 = self.down(x4)
            x4 = self.attn4(x4)
            x4 = self.up(x4)
            x4 = self.up(x4) + x4r

            xc = torch.cat([x1, x2, x3, x4], dim=1)
            x = self.feed_forward(xc) + x
            
        else:
            x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
            
            x1 = self.attn1(x1) + x1
            
            x2 = (x2 + x1) / 2.0
            x2r = x2
            x2 = self.down(x2)
            x2 = self.attn2(x2) 
            x2 = self.up(x2) + x2r

            x3 = (x3 + x2) / 2.0
            x3r = x3
            x3 = self.down(x3)
            x3 = self.down(x3)
            x3 = self.attn3(x3)
            x3 = self.up(x3)
            x3 = self.up(x3) + x3r

            x4 = (x4 + x3) / 2.0
            x4r = x4
            x4 = self.down(x4)
            x4 = self.down(x4)
            x4 = self.attn4(x4)
            x4 = self.up(x4)
            x4 = self.up(x4) + x4r
            
            xc = torch.cat([x1, x2, x3, x4], dim=1)
            x = self.feed_forward(xc) + x

        return x

class DWT(torch.nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def dwt_init(self, x):
        x_LL = (1.0 / 2) * (x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2])
        x_HL = (1.0 / 2) * (-x[:, :, 0::2, 0::2] - x[:, :, 1::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2])
        x_LH = (1.0 / 2) * (-x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2])
        x_HH = (1.0 / 2) * (x[:, :, 0::2, 0::2] - x[:, :, 1::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2])

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

    def forward(self, x):
        return self.dwt_init(x)

class IWT(torch.nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def iwt_init(self, x):
        r = 2
        iB, iC, iH, iW = x.size()
        oB, oC, oH, oW = iB, int(iC / (r ** 2)), r * iH, r * iW
        h = torch.zeros([oB, oC, oH, oW]).float().cuda()

        h[:, :, 0::2, 0::2] = (1.0 / 2) * (
                    x[:, 0:oC, :, :] - x[:, oC:oC * 2, :, :] - x[:, oC * 2:oC * 3, :, :] + x[:, oC * 3:oC * 4, :, :])
        h[:, :, 1::2, 0::2] = (1.0 / 2) * (
                    x[:, 0:oC, :, :] - x[:, oC:oC * 2, :, :] + x[:, oC * 2:oC * 3, :, :] - x[:, oC * 3:oC * 4, :, :])
        h[:, :, 0::2, 1::2] = (1.0 / 2) * (
                    x[:, 0:oC, :, :] + x[:, oC:oC * 2, :, :] - x[:, oC * 2:oC * 3, :, :] - x[:, oC * 3:oC * 4, :, :])
        h[:, :, 1::2, 1::2] = (1.0 / 2) * (
                    x[:, 0:oC, :, :] + x[:, oC:oC * 2, :, :] + x[:, oC * 2:oC * 3, :, :] + x[:, oC * 3:oC * 4, :, :])

        return h

    def forward(self, x):
        return self.iwt_init(x)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=True):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=True)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=True)
        
        self.act = nn.GELU()

    def forward(self, x):
        
        x = self.project_in(x)
        
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        
        x = self.act(x1) * x2
        
        x = self.project_out(x)
        
        return x


class TBlock(nn.Module):
    def __init__(self, ch, block_size=8, halo_size=3, num_heads=4, bias=False, sr=1):
        super(TBlock, self).__init__()
        self.block_size = block_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.head_ch = ch // num_heads
        assert ch % (num_heads*2) == 0, "ch should be divided by # heads"
        #注意186行需要(ch/num_head)%2==0

        # relative positional embedding: row and column embedding each with dimension 1/2 head_ch
        self.rel_h = nn.Parameter(torch.randn(1, block_size+2*halo_size, 1, self.head_ch//2), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(1, 1, block_size+2*halo_size, self.head_ch//2), requires_grad=True)

        self.qkv_conv = nn.Conv2d(ch, ch*3, kernel_size=1, bias=bias)
        
        self.sr = sr
        if sr > 1:
            self.sampler = nn.MaxPool2d(kernel_size=sr, stride=sr, padding=0)
            self.LocalProp = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch, bias=True, padding_mode='reflect')

        self.reset_parameters()

    def forward(self, x):
        # add channel reduction here to maintain the later concate operation
               
        if self.sr > 1:
            x = self.sampler(x)
        
        # pad feature maps to multiples of window size
        B, C, H, W = x.size()
        pad_l = pad_t = 0
        pad_r = (self.block_size - W % self.block_size) % self.block_size
        pad_b = (self.block_size - H % self.block_size) % self.block_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='reflect')

        b, c, h, w, block, halo, heads = *x.shape, self.block_size, self.halo_size, self.num_heads
        assert h % block == 0 and w % block == 0, 'feature map dimensions must be divisible by the block size'
        
        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = rearrange(q, 'b c (h k1) (w k2) -> (b h w) (k1 k2) c', k1=block, k2=block)
        q = q * (self.head_ch ** -0.5)

        k = F.unfold(k, kernel_size=block+halo*2, stride=block, padding=halo)
        k = rearrange(k, 'b (c a) l -> (b l) a c', c=c)

        v = F.unfold(v, kernel_size=block+halo*2, stride=block, padding=halo)
        v = rearrange(v, 'b (c a) l -> (b l) a c', c=c)

        # b*#blocks*#heads, flattened_vector, head_ch
        q, v = map(lambda i: rearrange(i, 'b a (h d) -> (b h) a d', h=heads), (q, v))
        # positional embedding
        k = rearrange(k, 'b (k1 k2) (h d) -> (b h) k1 k2 d', k1=block+2*halo, h=heads)
        k_h, k_w = k.split(self.head_ch//2, dim=-1)
        k = torch.cat([k_h+self.rel_h, k_w+self.rel_w], dim=-1)
        k = rearrange(k, 'b k1 k2 d -> b (k1 k2) d')

        # b*#blocks*#heads, flattened_query, flattened_neighborhood
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        attn = F.softmax(sim, dim=-1)
        # b*#blocks*#heads, flattened_query, head_ch
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h w n) (k1 k2) d -> b (n d) (h k1) (w k2)', b=b, h=(h//block), w=(w//block), k1=block, k2=block)
        
        if self.sr > 1:
            out = F.interpolate(out, scale_factor=(self.sr, self.sr), mode='bilinear', align_corners=False)
            out = self.LocalProp(out)

        if pad_r > 0 or pad_b > 0:
            return out[:,:,0:H,0:W]
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.qkv_conv.weight, mode='fan_out', nonlinearity='relu')
        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
        
        
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


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
