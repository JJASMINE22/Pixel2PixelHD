# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/8 14:05
@Project : Pixel2PixelHD
@File : discriminator.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch
from torch import nn
from custom.CustomLayers import ConvBlock


class EncodeLayer(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 basic_c: int,
                 layers_num: int,
                 activate: str = None):
        super(EncodeLayer, self).__init__()

        max_c = basic_c * 2 ** layers_num
        sequentials = nn.ModuleList()
        sequentials.add_module(module=ConvBlock(in_channels=in_c,
                                                out_channels=basic_c,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                padding_mode="zeros",
                                                normalize=nn.BatchNorm2d,
                                                activate="leaky_relu"), name="0")
        for i in range(1, layers_num):
            in_c = basic_c
            basic_c = min(2 * in_c, max_c)
            sequentials.add_module(module=ConvBlock(in_channels=in_c,
                                                    out_channels=basic_c,
                                                    kernel_size=(4, 4),
                                                    stride=(2, 2),
                                                    padding=(1, 1),
                                                    padding_mode="zeros",
                                                    normalize=nn.BatchNorm2d,
                                                    activate="leaky_relu"), name="{:d}".format(i))

        sequentials.add_module(module=ConvBlock(in_channels=basic_c,
                                                out_channels=min(basic_c * 2, max_c),
                                                kernel_size=(4, 4),
                                                stride=(1, 1),
                                                padding=(2, 2),
                                                padding_mode="zeros",
                                                normalize=nn.BatchNorm2d,
                                                activate="leaky_relu"), name="{:d}".format(layers_num))

        sequentials.add_module(module=ConvBlock(in_channels=min(basic_c * 2, max_c),
                                                out_channels=out_c,
                                                kernel_size=(4, 4),
                                                stride=(1, 1),
                                                padding=(1, 1),
                                                padding_mode="zeros",
                                                normalize=None,
                                                activate=activate), name="{:d}".format(layers_num + 1))

        setattr(self, "sequentials", sequentials)

    def forward(self, x):

        features = []
        for module in self.sequentials:
            x = module(x)
            features.append(x)

        return features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 basic_c: int,
                 layers_num: int,
                 blocks_num: int,
                 activate: str = "sigmoid",
                 get_feats: bool = False):
        super(MultiScaleDiscriminator, self).__init__()

        # self.init_conv = ConvBlock(in_channels=in_c,
        #                            out_channels=basic_c,
        #                            kernel_size=(3, 3),
        #                            stride=(1, 1),
        #                            padding=(1, 1),
        #                            padding_mode="zeros",
        #                            normalize=nn.BatchNorm2d,
        #                            activate="leaky_relu")

        for i in range(blocks_num):
            basic_module = EncodeLayer(in_c=in_c,
                                       out_c=out_c,
                                       basic_c=basic_c,
                                       layers_num=layers_num,
                                       activate=activate)
            setattr(self, "block_{:d}".format(i), basic_module)

        self.avg_pool = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1))

        self.get_feats = get_feats
        self.blocks_num = blocks_num

    def forward(self, x):

        # x = self.init_conv(x)

        features = []
        for i in range(self.blocks_num):
            block = getattr(self, "block_{:d}".format(i))
            if self.get_feats:
                features.append(block(x))
            else:
                features.append(block(x)[-1])
            x = self.avg_pool(x)

        return features

