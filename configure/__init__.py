# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/13 17:02
@Project : Pixel2PixelHD
@File : __init__.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch

# ===data generator===
fill_per = .3
batch_size = 8
scale_size = (13, 7)
image_size = (512, 256)
padding_size = (16, 8)
transform_ratio = 1.

# ===model===
use_enhancer = False
in_c = 4 if use_enhancer else 1
out_c = 3
basic_c = 64
classes_num = 39
get_disFeats = True
backbone = "vgg19"
out_indices = (1, 2, 3, 4, 5)
lambda_feat = 10.

# ===enhancer===
e_basic_c = 16
e_layers_num = 4

# ===discriminator==
D_blocks = 3
D_layers = 3

# ===generator==
G_blocks = 9
G_layers = 4
cutoffs = 3

# === train===
Epoches = 150
resume_train = True
lr_rates = {'gen_lr': 3e-4,
            'dis_lr': 2e-4}
betas = {'beta1': .5,
         'beta2': 0.999}
per_sample_interval = 100
