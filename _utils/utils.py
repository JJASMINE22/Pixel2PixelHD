# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/16 13:50
@Project : Pixel2PixelHD
@File : utils.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import cv2
import torch
import random
import numpy as np
from torch import nn
from torch.nn import functional as F


class RandomRotation90(nn.Module):
    def __init__(self,
                 p: float):
        super(RandomRotation90, self).__init__()
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            x = torch.rot90(x, k=2, dims=[2, 3])
        return x


def get_edges(instances):
    edge = torch.zeros_like(instances)
    edge[:, :, :, 1:] = torch.logical_or(edge[:, :, :, 1:],
                                         torch.not_equal(instances[:, :, :, 1:], instances[:, :, :, :-1]))
    edge[:, :, :, :-1] = torch.logical_or(edge[:, :, :, :-1],
                                          torch.not_equal(instances[:, :, :, 1:], instances[:, :, :, :-1]))
    edge[:, :, 1:, :] = torch.logical_or(edge[:, :, 1:, :],
                                         torch.not_equal(instances[:, :, 1:, :], instances[:, :, :-1, :]))
    edge[:, :, :-1, :] = torch.logical_or(edge[:, :, :-1, :],
                                          torch.not_equal(instances[:, :, 1:, :], instances[:, :, :-1, :]))

    return edge.float()


class DataWarp:
    def __init__(self,
                 scale_size: tuple,
                 image_size: tuple,
                 padding_size: tuple):
        self.scale_size = scale_size
        self.image_size = image_size
        self.padding_size = padding_size

    def __call__(self, image, gridx_offset, gridy_offset):
        # assert image.shape.__len__() == 3

        image = cv2.resize(image, self.image_size)
        # 3 Dimension
        image = torch.tensor(image)
        image = F.pad(image, pad=(0, 0, *self.padding_size, *self.padding_size)).numpy()

        gridx = np.linspace(self.padding_size[0], self.image_size[0] + self.padding_size[0], self.scale_size[0])
        gridy = np.linspace(self.padding_size[1], self.image_size[1] + self.padding_size[1], self.scale_size[1])

        gridy, gridx = np.meshgrid(gridy, gridx, indexing='ij')
        gridx += gridx_offset
        gridy += gridy_offset

        gridx = cv2.resize(gridx, (self.image_size[0] + 4 * self.padding_size[0],
                                   self.image_size[1] + 4 * self.padding_size[1]))
        gridy = cv2.resize(gridy, (self.image_size[0] + 4 * self.padding_size[0],
                                   self.image_size[1] + 4 * self.padding_size[1]))
        gridx = np.clip(gridx, self.padding_size[0], self.image_size[0] + self.padding_size[0] - 1)
        gridy = np.clip(gridy, self.padding_size[1], self.image_size[1] + self.padding_size[1] - 1)

        interp_gridx = gridx[2 * self.padding_size[1]: self.image_size[1] + 2 * self.padding_size[1],
                       2 * self.padding_size[0]: self.image_size[0] + 2 * self.padding_size[0]].astype('float32')
        interp_gridy = gridy[2 * self.padding_size[1]: self.image_size[1] + 2 * self.padding_size[1],
                       2 * self.padding_size[0]: self.image_size[0] + 2 * self.padding_size[0]].astype('float32')

        warped_image = cv2.remap(image, interp_gridx, interp_gridy, cv2.INTER_LINEAR)

        return warped_image
