# -*- coding: utf-8 -*-
"""
Created on Tue May 31 09:36:58 2022

@author: Naive
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
#from swinT import SwinT
#import torch.utils.checkpoint as checkpoint
#from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#import numpy as np


class CNNmodel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.num_classes = num_classes
        self.conv = nn.Sequential(
            #nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            #SwinT(64,64,input_resolution=[64,64],num_heads=4,window_size=8,downsample=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            #nn.SELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            #nn.AdaptiveAvgPool2d(64),
            #
            #SwinT(64,128,input_resolution=[64,64],num_heads=16,window_size=8,downsample=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            #nn.SELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.SELU(inplace=True),
            nn.ReLU(inplace=True),
            #winT(128,128,input_resolution=[32,32],num_heads=16,window_size=8),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            #nn.SELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.SELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, ),
        )
        self.psfconv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            #nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            #nn.AvgPool2d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(96384, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            
            #nn.SELU(inplace=True),
            #nn.SELU(inplace=True),
            #nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(512, self.num_classes),
        )

    def forward(self, x, psf):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        psf = self.psfconv(psf)
        psf=torch.flatten(psf,1)
        x=torch.cat((x,psf),dim=1)
        x = self.fc1(x)
        return x
