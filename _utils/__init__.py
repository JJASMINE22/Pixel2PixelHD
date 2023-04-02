# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/16 15:20
@Project : Pixel2PixelHD
@File : __init__.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch
from torchvision import transforms
from _utils.utils import RandomRotation90

compose = transforms.Compose([
    RandomRotation90(p=.5),
    transforms.RandomVerticalFlip(p=.5),
    transforms.RandomHorizontalFlip(p=.5)
])
