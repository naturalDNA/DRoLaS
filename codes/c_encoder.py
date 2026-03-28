import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

class Swish(nn.Module):
    """
    ### Swish 激活函数
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class Land_Encoder(nn.Module):
    def __init__(self,n_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(10,n_channels,(3,3),(2,2),(1,1))
        self.act1 = Swish()
        self.norm1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels,2*n_channels,(3,3),(2,2),(1,1))
        self.act2 = Swish()
        self.norm2 = nn.BatchNorm2d(2*n_channels)
        self.conv3 = nn.Conv2d(2*n_channels,n_channels,(3,3),(2,2),(1,1))
        self.act3 = Swish()
        self.norm3 = nn.BatchNorm2d(n_channels)

    def forward(self,x):
        x = x.to(torch.long)
        b,c,w,h= x.shape
        x = x.view(b,w,h)
        x = nn.functional.one_hot(x,num_classes=10).float().permute(0,3,1,2)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.act3(self.norm3(self.conv3(x)))
        return x

class c_residule(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3),padding=(1,1))
        self.act1 = Swish()
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3),padding=(1,1))
        self.act2 = Swish()
        self.norm2 = nn.BatchNorm2d(out_channels)
        if in_channels!=out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (3, 3),padding=(1,1))
        else:
            self.shortcut = nn.Identity()

    def forward(self,x):
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


land_encoder = Land_Encoder(64//2)

