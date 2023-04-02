# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/13 17:51
@Project : Pixel2PixelHD
@File : loss_utils.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch
from torch import nn

l1loss = nn.L1Loss(reduction="mean")
bceloss = nn.BCELoss(reduction="mean")