# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/8 14:05
@Project : Pixel2PixelHD
@File : generator.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch
from torch import nn
from custom.CustomLayers import ConvBlock, ResBlock


class Encoder(nn.Module):
    def __init__(self,
                 basic_c: int,
                 cut_offs: int,
                 layers_num: int):
        """
        :argument: Prepared for Unet architecture,
        separate Encoder from Generator
        :param basic_c: Basic embedded channels
        :param layers_num: Number of layers to be downsampled
        """
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([ConvBlock(in_channels=basic_c * (2 ** i)
        if i < cut_offs else (2 ** cut_offs) * basic_c,
                                               out_channels=basic_c * (2 ** (i + 1))
                                               if i < cut_offs else (2 ** cut_offs) * basic_c,
                                               kernel_size=(3, 3),
                                               stride=(2, 2),
                                               padding=(1, 1),
                                               drop_rate=0. if i < cut_offs else .3) for i in range(layers_num)])

    def forward(self, x):
        feats = []
        feats.append(x)
        for layer in self.layers:
            x = layer(x)
            feats.append(x)

        return feats


class Decoder(nn.Module):
    def __init__(self,
                 basic_c: int,
                 cut_offs: int,
                 layers_num: int):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([ConvBlock(in_channels=basic_c * 2 ** cut_offs
        if i < layers_num - cut_offs else basic_c * 2 ** (layers_num - i),
                                               out_channels=basic_c * 2 ** cut_offs
                                               if i < layers_num - cut_offs else basic_c * 2 ** (layers_num - i - 1),
                                               kernel_size=(3, 3),
                                               padding=(1, 1),
                                               upsample=True,
                                               drop_rate=0. if i >= layers_num - cut_offs else .3)
                                     for i in range(layers_num)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class UnetDecoder(nn.Module):
    def __init__(self,
                 basic_c: int,
                 cut_offs: int,
                 layers_num: int):
        """
        :argument: Prepared for Unet architecture,
        separate Decoder from Generator
        :param basic_c: Basic embedded channels
        :param layers_num: Number of layers to be upsampled
        """
        super(UnetDecoder, self).__init__()
        self.layers_num = layers_num
        self.cut_offs = cut_offs

        self.former_layers = nn.ModuleList([ConvBlock(in_channels=basic_c * 2 ** cut_offs
        if not i else basic_c * 2 ** (cut_offs + 1),
                                                      out_channels=basic_c * 2 ** cut_offs,
                                                      kernel_size=(3, 3),
                                                      padding=(1, 1),
                                                      upsample=True,
                                                      drop_rate=.3) for i in range(layers_num - cut_offs)])

        self.latter_layers = nn.ModuleList([ConvBlock(in_channels=basic_c * 2 ** (cut_offs - i + 1),
                                                      out_channels=basic_c * (2 ** (cut_offs - i - 1)),
                                                      kernel_size=(3, 3),
                                                      padding=(1, 1),
                                                      upsample=True) for i in range(cut_offs)])

    def forward(self, x, enc_outputs):

        for i, layer in enumerate(self.former_layers):
            x = layer(x)
            x = torch.cat([enc_outputs[-i - 1], x], dim=1)

        for j, layer in enumerate(self.latter_layers):
            x = layer(x)
            x = torch.cat([enc_outputs[self.cut_offs - j - 1], x], dim=1)

        return x


class Generator(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 basic_c: int,
                 cutoffs: int,
                 layers_num: int,
                 blocks_num: int,
                 activate: str = "tanh"):
        super(Generator, self).__init__()

        self.init_conv = ConvBlock(in_channels=in_c,
                                   out_channels=basic_c,
                                   kernel_size=(7, 7),
                                   padding=(3, 3))
        self.final_conv = ConvBlock(in_channels=basic_c,
                                    out_channels=out_c,
                                    kernel_size=(7, 7),
                                    padding=(3, 3),
                                    normalize=None,
                                    activate=activate)

        self.encoder = Encoder(basic_c, cutoffs, layers_num)
        self.decoder = Decoder(basic_c, cutoffs, layers_num)
        self.blocks = nn.ModuleList([ResBlock(channels=basic_c * 2 ** cutoffs,
                                              drop_rate=.5)
                                     for i in range(blocks_num)])

    def forward(self, input):
        x = self.init_conv(input)
        x = self.encoder(x)[-1]
        for block in self.blocks:
            x = block(x)
        x = self.decoder(x)
        x = self.final_conv(x)

        return x


class UnetGenerator(nn.Module):
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 basic_c: int,
                 cutoffs: int,
                 layers_num: int,
                 blocks_num: int,
                 activate: str = None):
        super(UnetGenerator, self).__init__()
        assert cutoffs < layers_num

        self.init_conv = ConvBlock(in_channels=in_c,
                                   out_channels=basic_c,
                                   kernel_size=(7, 7),
                                   padding=(3, 3))
        self.final_conv = ConvBlock(in_channels=basic_c * 2,
                                    out_channels=out_c,
                                    kernel_size=(7, 7),
                                    padding=(3, 3),
                                    normalize=None,
                                    activate=activate)

        self.encoder = Encoder(basic_c, cutoffs, layers_num)
        self.decoder = UnetDecoder(basic_c, cutoffs, layers_num)
        self.blocks = nn.ModuleList([ResBlock(channels=basic_c * 2 ** cutoffs,
                                              drop_rate=.5)
                                     for i in range(blocks_num)])

    def forward(self, input):
        x = self.init_conv(input)
        enc_outputs = self.encoder(x)
        x = enc_outputs[-1]
        for block in self.blocks:
            x = block(x)
        x = self.decoder(x, enc_outputs[:-1])
        x = self.final_conv(x)

        return x
