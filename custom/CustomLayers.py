# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/8 14:04
@Project : Pixel2PixelHD
@File : CustomLayers.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch
from torch import nn


class Activation(nn.Module):
    def __init__(self,
                 name,
                 **kwargs):
        super(Activation, self).__init__()
        if name is None:
            self.activation = nn.Identity()
        elif name == "softmax":
            self.activation = nn.Softmax(dim=-1)
        elif name == "sigmoid":
            self.activation = torch.sigmoid
        elif name == "tanh":
            self.activation = torch.tanh
        elif name == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif name == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=.2,
                                           inplace=False)
        else:
            raise NameError('Activation got {}'.format(name))

    def forward(self, x):

        x = self.activation(x)

        return x


class ConvBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple = (3, 3),
                 stride: tuple = (1, 1),
                 padding: tuple = (0, 0),
                 padding_mode: str = "reflect",
                 drop_rate: float = 0.,
                 normalize=nn.InstanceNorm2d,
                 upsample: bool = False,
                 activate="relu"):

        sequentials = []
        if upsample: sequentials.append(nn.Upsample(scale_factor=2.,
                                                    mode="bilinear",
                                                    align_corners=True))
        sequentials.append(nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     stride=stride,
                                     padding_mode=padding_mode))

        if normalize is not None:
            sequentials.append(normalize(num_features=out_channels))
        sequentials.append(Activation(name=activate))
        sequentials.append(nn.Dropout(p=drop_rate))

        super(ConvBlock, self).__init__(*sequentials)
        self.apply(self.init_params)

    @torch.no_grad()
    def init_params(self, module):

        if isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0., std=.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, input):
        x = super(ConvBlock, self).forward(input)

        return x


class ResBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: tuple = (3, 3),
                 padding: tuple = (1, 1),
                 padding_mode: str = "reflect",
                 use_drop: bool = True,
                 drop_rate: float = 0.,
                 activate: bool = False):
        super(ResBlock, self).__init__()
        self.activate = activate

        self.init_block = ConvBlock(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=kernel_size,
                                    stride=(1, 1), padding=padding,
                                    padding_mode=padding_mode)

        self.final_block = ConvBlock(in_channels=channels,
                                     out_channels=channels,
                                     kernel_size=kernel_size,
                                     stride=(1, 1), padding=padding,
                                     padding_mode=padding_mode,
                                     activate=None)

        if use_drop:
            self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, input):

        x = self.init_block(input)
        if hasattr(self, "dropout"):
            x = self.dropout(x)

        x = self.final_block(x)

        x = x + input

        if self.activate:
            x = torch.relu(x)

        return x
