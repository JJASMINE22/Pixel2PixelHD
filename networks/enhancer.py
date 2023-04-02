# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/21 14:41
@Project : Pixel2PixelHD
@File : enhancer.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch
import numpy as np
from torch import nn
from custom.CustomLayers import ConvBlock, ResBlock


class Enhancer(nn.Sequential):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 basic_c: int,
                 layers_num: int,
                 activate: str = "tanh"):

        sequentials = []
        init_conv = ConvBlock(in_channels=in_c,
                                   out_channels=basic_c,
                                   kernel_size=(7, 7),
                                   padding=(3, 3))

        final_conv = ConvBlock(in_channels=basic_c,
                                    out_channels=out_c,
                                    kernel_size=(7, 7),
                                    padding=(3, 3),
                                    normalize=None,
                                    activate=activate)

        down_layers = nn.ModuleList([ConvBlock(in_channels=basic_c * (2 ** i),
                                                    out_channels=basic_c * (2 ** (i + 1)),
                                                    kernel_size=(3, 3),
                                                    stride=(2, 2),
                                                    padding=(1, 1),
                                                    padding_mode="zeros") for i in range(layers_num)])

        up_layers = nn.ModuleList([ConvBlock(in_channels=basic_c * (2 ** (layers_num - i)),
                                                  out_channels=basic_c * (2 ** (layers_num - i - 1)),
                                                  kernel_size=(3, 3),
                                                  padding=(1, 1),
                                                  padding_mode="zeros",
                                                  upsample=True) for i in range(layers_num)])
        sequentials.extend([init_conv, *down_layers, *up_layers, final_conv])

        super(Enhancer, self).__init__(*sequentials)
        self.out_c = out_c

    def forward(self, inputs):
        input, inst = inputs
        x = super(Enhancer, self).forward(input)

        # instance-wise average pooling
        x_mean = x.clone()
        inst_set = torch.unique(inst.int())
        for i in inst_set:
            for b in range(input.size(0)):
                indices = torch.where(torch.eq(inst[b], i))
                for j in range(self.out_c):
                    x_ins = x[b, j, indices[0], indices[1]]
                    mean_feat = torch.mean(x_ins).expand_as(x_ins)
                    x_mean[b, j, indices[0], indices[1]] = mean_feat
        return x_mean
