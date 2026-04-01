'''
Local Spatial Attention (LSA)

[from] PLDet: Intra-layer Multi-scale Perception and Local Space Attention for Pulmonary Lesion Detection in CT Images (BSPC 2026)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class LSA(nn.Module): 

    def __init__(self, in_channels, out_channels, bias=False, shortcut=True, k_size=7, alpha=0.7):
        super(LSA, self).__init__()
        
        self.alpha = alpha
        
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.max_pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, None))
        
        # Depthwise Separable Convolution
        self.pool_h_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=(k_size, 1), stride=1, groups=in_channels, padding=((k_size - 1) // 2, 0), bias=bias) # bias=False 可选
        # self.pool_h_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, groups=1, padding=0)
        self.pool_w_depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=(1, k_size), stride=1, groups=in_channels, padding=(0, (k_size - 1) // 2), bias=bias) # bias=False 可选
        # self.pool_w_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, groups=1, padding=0)
        
        self.use_add = shortcut and in_channels == out_channels
    def forward(self, x):

        x_h_avg = self.avg_pool_h(x)
        x_w_avg = self.avg_pool_w(x)
        x_h_max = self.max_pool_h(x)
        x_w_max = self.max_pool_w(x)
        
        x_h = self.alpha*x_h_avg + (1-self.alpha)*x_h_max
        x_w = self.alpha*x_w_avg + (1-self.alpha)*x_w_max
        a_h = self.pool_h_depthwise(x_h).sigmoid()
        a_w = self.pool_w_depthwise(x_w).sigmoid()

        # a_h = self.pool_h_depthwise(x_h_avg).sigmoid()
        # a_w = self.pool_w_depthwise(x_w_avg).sigmoid()

        out = x * a_w * a_h
        if self.use_add:
            out += x

        return out
    

# Example usage
if __name__ == "__main__":
    model = LSA(in_channels=32, out_channels=32)
    input_tensor = torch.randn(1, 32, 128, 128)  # Batch size of 1, 32 channels, 128x128 image size
    output = model(input_tensor)
    print(output.shape)  # Should be the same as input shape: (1, 32, 128, 128)