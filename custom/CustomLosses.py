# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/8 14:04
@Project : Pixel2PixelHD
@File : CustomLosses.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import torch
from application import model
from custom.loss_utils import *
from configure import D_blocks, D_layers, lambda_feat


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()

        self.model = model
        self.weights = [1.0 / 32, 1.0 / 16,
                        1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, fake, real):
        fake_logits = self.model(fake)
        real_logits = self.model(real)

        loss = 0.
        for i, (fake_logit, real_logit) in enumerate(zip(fake_logits,
                                                         real_logits)):
            loss += self.weights[i] * l1loss(fake_logit, real_logit.detach())
        return loss


class DisFeatLoss(nn.Module):
    def __init__(self,
                 blocks_num: int,
                 layers_num: int):
        super(DisFeatLoss, self).__init__()
        self.blocks_num = blocks_num
        self.layers_num = layers_num
        self.f_weights = 4. / (layers_num + 1)
        self.d_weights = 1. / blocks_num

    def forward(self, fake_logits, real_logits):
        loss = 0.
        for fake_logit, real_logit in zip(fake_logits, real_logits):
            for f_logit, r_logit in zip(fake_logit, real_logit):
                loss += self.f_weights * self.d_weights * l1loss(f_logit, r_logit.detach()) * lambda_feat
        return loss


class GANLoss(nn.Module):
    def __init__(self,
                 fake_label: torch.Tensor = torch.tensor(0.),
                 real_label: torch.Tensor = torch.tensor(1.)):
        super(GANLoss, self).__init__()
        self.fake_label = fake_label
        self.real_label = real_label

    def gen_loss(self, fake_logit, real_logit):
        if fake_logit is not None:
            fake_target = torch.ones_like(fake_logit) * self.fake_label
            real_target = torch.ones_like(real_logit) * self.real_label

            fake_loss = bceloss(fake_logit, fake_target)
            real_loss = bceloss(real_logit, real_target)

            return fake_loss, real_loss
        else:
            real_target = torch.ones_like(real_logit) * self.real_label

            real_loss = bceloss(real_logit, real_target)

            return real_loss

    def forward(self, fake_logits, real_logits):
        if isinstance(real_logits[0], list):
            loss = 0.
            if fake_logits is not None:
                for fake_logit, real_logit in zip(fake_logits, real_logits):
                    f_logit = fake_logit[-1]
                    r_logit = real_logit[-1]

                    fake_loss, real_loss = self.gen_loss(f_logit, r_logit)

                    loss += (fake_loss + real_loss) / 2.
            else:
                for real_logit in real_logits:
                    r_logit = real_logit[-1]
                    real_loss = self.gen_loss(None, r_logit)
                    loss += real_loss

        else:
            if fake_logits is not None:
                fake_logit = fake_logits[-1]
                real_logit = real_logits[-1]

                fake_loss, real_loss = self.gen_loss(fake_logit, real_logit)

                loss = fake_loss + real_loss
                loss /= 2.
            else:
                real_logit = real_logits[-1]

                loss = self.gen_loss(None, real_logit)

        return loss

