# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/20 10:19
@Project : Pixel2PixelHD
@File : train.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch
import numpy as np
from pixel2pixel import Pixel2Pixel
from _utils.generate import Generator
from configure import *
from configure import config as cfg

if __name__ == '__main__':

    pixel = Pixel2Pixel(in_c=in_c,
                        out_c=out_c,
                        basic_c=basic_c,
                        e_basic_c=e_basic_c,
                        cutoffs=cutoffs,
                        classes_num=classes_num,
                        g_layers_num=G_layers,
                        g_blocks_num=G_blocks,
                        d_layers_num=D_layers,
                        d_blocks_num=D_blocks,
                        e_layers_num=e_layers_num,
                        get_disFeats=get_disFeats,
                        betas=betas,
                        lr_rates=lr_rates,
                        resume_train=resume_train,
                        gen_ckpt_path=cfg.gen_ckpt_path + "/Epoch089_gen_loss47.54_gen_acc0.47.pth.tar",
                        dis_ckpt_path=cfg.dis_ckpt_path + "/Epoch089_dis_loss0.19_dis_acc98.00.pth.tar",
                        eha_ckpt_path=cfg.eha_ckpt_path + "/",
                        lambda_feat=lambda_feat,
                        use_enhancer=use_enhancer,
                        device=cfg.device)

    data_gen = Generator(anno_train_root=cfg.anno_train_root,
                         anno_valid_root=cfg.anno_valid_root,
                         image_train_root=cfg.image_train_root,
                         image_valid_root=cfg.image_valid_root,
                         classes_num=classes_num,
                         fill_per=fill_per,
                         batch_size=batch_size,
                         scale_size=scale_size,
                         image_size=image_size,
                         padding_size=padding_size,
                         transform_ratio=transform_ratio)

    train_gen = data_gen.generate(training=True)
    valid_gen = data_gen.generate(training=False)

    for epoch in range(Epoches):

        for i in range(data_gen.get_train_len()):
            labels, instances, images = next(train_gen)
            pixel.train(labels, instances, images)
            print(i)
            if not (i + 1) % per_sample_interval:
                pixel.generate_sample(labels, instances, images, i + 1)

        torch.save({'gen_state_dict': pixel.generator.state_dict(),
                    'gen_loss': pixel.train_gen_loss / data_gen.get_train_len(),
                    'gen_acc': pixel.train_gen_acc / data_gen.get_train_len() * 100},
                   cfg.gen_ckpt_path + '/Epoch{:0>3d}_gen_loss{:.2f}_gen_acc{:.2f}.pth.tar'.format(
                       epoch + 1,
                       pixel.train_gen_loss / data_gen.get_train_len(),
                       pixel.train_gen_acc / data_gen.get_train_len() * 100))

        torch.save({'dis_state_dict': pixel.discriminator.state_dict(),
                    'dis_loss': pixel.train_dis_loss / data_gen.get_train_len(),
                    'dis_acc': pixel.train_dis_acc / data_gen.get_train_len() * 100},
                   cfg.dis_ckpt_path + '/Epoch{:0>3d}_dis_loss{:.2f}_dis_acc{:.2f}.pth.tar'.format(
                       epoch + 1,
                       pixel.train_dis_loss / data_gen.get_train_len(),
                       pixel.train_dis_acc / data_gen.get_train_len() * 100))

        if use_enhancer:
            torch.save({'eha_state_dict': pixel.enhancer.state_dict()},
                       cfg.eha_ckpt_path + '/Epoch{:0>3d}.pth.tar'.format(
                           epoch + 1))

        print(f'Epoch: {epoch + 1}\n'
              f'train_dis_loss: {pixel.train_dis_loss / data_gen.get_train_len()}\n'
              f'train_gen_loss: {pixel.train_gen_loss / data_gen.get_train_len()}\n'
              f'train_gen_acc: {pixel.train_gen_acc / data_gen.get_train_len() * 100}\n'
              f'train_dis_acc: {pixel.train_dis_acc / data_gen.get_train_len() * 100}\n')

        for i in range(data_gen.get_valid_len()):
            labels, instances, images = next(valid_gen)
            pixel.validate(labels, instances, images)

        print(f'Epoch: {epoch + 1}\n'
              f'valid_dis_loss: {pixel.valid_dis_loss / data_gen.get_valid_len()}\n'
              f'valid_gen_loss: {pixel.valid_gen_loss / data_gen.get_valid_len()}\n'
              f'valid_gen_acc: {pixel.valid_gen_acc / data_gen.get_valid_len() * 100}\n'
              f'valid_dis_acc: {pixel.valid_dis_acc / data_gen.get_valid_len() * 100}\n')

        pixel.train_gen_loss, pixel.valid_gen_loss = 0, 0
        pixel.train_dis_loss, pixel.valid_dis_loss = 0, 0
        pixel.train_gen_acc, pixel.valid_gen_acc = 0, 0
        pixel.train_dis_acc, pixel.valid_dis_acc = 0, 0
