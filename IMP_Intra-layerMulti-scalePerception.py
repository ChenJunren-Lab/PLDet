'''
Intra-layer Multi-scale Perception (IMP) 

[from] PLDet: Intra-layer Multi-scale Perception and Local Space Attention for Pulmonary Lesion Detection in CT Images (BSPC 2026)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # patch_top_left  = x[...,  ::2,  ::2]
        # patch_bot_left  = x[..., 1::2,  ::2]
        # patch_top_right = x[...,  ::2, 1::2]
        # patch_bot_right = x[..., 1::2, 1::2]
        # x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)) # self.conv(x)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)
    

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate, groups=1, bias=False, act="silu"):
        super().__init__()
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

#! CJR：上采样模块
class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bicubic")
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv1(self.upsample(x))
    
class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, shortcut=False, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)

        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

        
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class IMP(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=1.5, depthwise=False, act="silu",):

        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        
        self.conv2_d1 = DilatedConv(in_channels, hidden_channels, dilation_rate=1)
        self.conv2_d2 = DilatedConv(in_channels, hidden_channels, dilation_rate=2)
        self.conv2_d5 = DilatedConv(in_channels, hidden_channels, dilation_rate=5)

        self.conv3  = BaseConv(4 * hidden_channels, out_channels, 1, stride=1, act=act)

        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)


    def forward(self, x):

        x_1 = self.conv1(x) 

        x_2_d1 = self.conv2_d1(x)
        x_2_d2 = self.conv2_d2(x)
        x_2_d5 = self.conv2_d5(x)
        
        x_2_add = x_2_d1 + x_2_d2 + x_2_d5


        x_1_bottleneck = self.m(x_1 + x_2_add)
        

        x = torch.cat((x_1_bottleneck + x_1, x_2_d1, x_2_d2, x_2_d5), dim=1)

        x = self.conv3(x)
        return x
    

# Example usage
if __name__ == "__main__":
    model = IMP(in_channels=32, out_channels=32)
    input_tensor = torch.randn(1, 32, 128, 128)  # Batch size of 1, 32 channels, 128x128 image size
    output = model(input_tensor)
    print(output.shape)  # Should be the same as input shape: (1, 32, 128, 128)