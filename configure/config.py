# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/8 14:05
@Project : Pixel2PixelHD
@File : config.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch

# ===generate===
anno_train_root = "训练集标签根目录"
anno_valid_root = "测试/验证集标签根目录"
image_train_root = "训练集数据根目录"
image_valid_root = "测试/验证集数据根目录"
# ===model===
gen_ckpt_path = './torch_models/generator'
dis_ckpt_path = './torch_models/discriminator'
eha_ckpt_path = './torch_models/enhancer'
# ===prediction===
sample_path = "./result/Batch{}.jpg"
# ===device===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
