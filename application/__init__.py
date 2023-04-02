# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/13 16:56
@Project : Pixel2PixelHD
@File : __init__.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import timm
import torch
from torch import nn
from configure import *
from configure import config as cfg

model = timm.create_model(model_name=backbone, pretrained=True,
                          features_only=True, out_indices=out_indices).to(cfg.device)
