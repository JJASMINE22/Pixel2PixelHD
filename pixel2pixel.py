'''
@Time : 2023/3/8 14:04
@Project : Pixel2PixelHD
@File : pixel2pixel.py
@IDE : PyCharm
@Author : XinYi Huang
@Email : m13541280433@163.com
'''

import torch
import random
import numpy as np
from PIL import Image
from networks.enhancer import Enhancer
from networks.generator import Generator
from networks.discriminator import MultiScaleDiscriminator
from _utils.utils import get_edges
from custom.CustomLosses import *
from configure import config as cfg


class Pixel2Pixel:
    def __init__(self,
                 in_c: int,
                 out_c: int,
                 basic_c: int,
                 e_basic_c: int,
                 cutoffs: int,
                 classes_num: int,
                 g_layers_num: int,
                 g_blocks_num: int,
                 d_layers_num: int,
                 d_blocks_num: int,
                 e_layers_num: int,
                 get_disFeats: bool,
                 betas: dict,
                 lr_rates: dict,
                 resume_train: bool,
                 gen_ckpt_path: str,
                 dis_ckpt_path: str,
                 eha_ckpt_path: str,
                 lambda_feat: float,
                 use_enhancer: bool,
                 device):
        assert lr_rates.__len__() == 2
        assert all(map(lambda i: isinstance(betas[i], float), betas))
        assert all(map(lambda i: isinstance(lr_rates[i], float), lr_rates))

        self.device = device
        self.use_eha = use_enhancer
        self.lambda_feat = lambda_feat
        self.get_disFeat = get_disFeats
        beta1 = betas['beta1']
        beta2 = betas['beta2']
        gen_learning_rate = lr_rates['gen_lr']
        dis_learning_rate = lr_rates['dis_lr']

        if use_enhancer:
            self.enhancer = Enhancer(in_c=out_c,
                                     out_c=out_c,
                                     basic_c=e_basic_c,
                                     layers_num=e_layers_num).to(device)

        self.generator = Generator(in_c=in_c + classes_num,
                                   out_c=out_c,
                                   basic_c=basic_c,
                                   cutoffs=cutoffs,
                                   layers_num=g_layers_num,
                                   blocks_num=g_blocks_num).to(device)

        self.discriminator = MultiScaleDiscriminator(in_c=out_c + classes_num + 1,
                                                     out_c=1,
                                                     basic_c=basic_c,
                                                     layers_num=d_layers_num,
                                                     blocks_num=d_blocks_num,
                                                     get_feats=get_disFeats).to(device)

        if resume_train:
            try:
                gen_ckpt = torch.load(gen_ckpt_path)
                dis_ckpt = torch.load(dis_ckpt_path)
                self.generator.load_state_dict(gen_ckpt['gen_state_dict'])
                self.discriminator.load_state_dict(dis_ckpt['dis_state_dict'])
                print("generator successfully loaded, gen loss is {:.3f}".format(gen_ckpt['gen_loss']))
                print("discriminator successfully loaded, dis loss is {:.3f}".format(dis_ckpt['dis_loss']))
                if use_enhancer:
                    eha_ckpt = torch.load(eha_ckpt_path)
                    self.enhancer.load_state_dict(eha_ckpt['eha_state_dict'])
                    print("enhancer successfully loaded")
            except FileNotFoundError:
                raise ("please enter the right params path, get dis path {:s}, gen path {:s}".format(
                    dis_ckpt_path, gen_ckpt_path))

        self.vgg_loss_fn = VGGLoss()
        self.gan_loss_fn = GANLoss()
        self.dis_loss_fn = DisFeatLoss(d_blocks_num, d_layers_num)

        if use_enhancer:
            generator_params = [*self.enhancer.parameters(), *self.generator.parameters()]
        else:
            generator_params = self.generator.parameters()
        self.g_optimizer = torch.optim.Adam(params=generator_params,
                                            lr=gen_learning_rate, betas=(beta1, beta2))
        self.d_optimizer = torch.optim.Adam(params=self.discriminator.parameters(),
                                            lr=dis_learning_rate, betas=(beta1, beta2))

        self.train_gen_loss, self.valid_gen_loss = 0, 0
        self.train_dis_loss, self.valid_dis_loss = 0, 0
        self.train_gen_acc, self.valid_gen_acc = 0, 0
        self.train_dis_acc, self.valid_dis_acc = 0, 0

    def discriminate(self, images, labels):
        logits = self.discriminator(torch.cat([labels, images], dim=1))

        return logits

    def train(self, labels, instances, targets):
        if self.device:
            labels = labels.to(self.device)
            instances = instances.to(self.device)
            targets = targets.to(self.device)

        edge_maps = get_edges(instances)
        input_labels = torch.cat([labels, edge_maps], dim=1)

        # Discriminator
        self.d_optimizer.zero_grad()

        if self.use_eha:
            inst_maps = torch.argmax(labels, dim=1)
            feat_maps = self.enhancer([targets, inst_maps])
            fake_images = self.generator(torch.cat([input_labels, feat_maps], dim=1))
        else:
            fake_images = self.generator(input_labels)
        fake_logits = self.discriminate(fake_images.detach(), input_labels)
        real_logits = self.discriminate(targets, input_labels)
        loss_D = self.gan_loss_fn(fake_logits, real_logits)
        loss_D.backward(retain_graph=True)
        self.d_optimizer.step()

        self.train_dis_loss += loss_D.data.item()

        if isinstance(fake_logits[0], list):
            self.train_dis_acc += torch.tensor([torch.cat([torch.le(fake_logit[-1].squeeze(1), .5),
                                                           torch.gt(real_logit[-1].squeeze(1), .5)], dim=0).sum() /
                                                (2 * torch.prod(torch.tensor(fake_logit[-1].size())))
                                                for fake_logit, real_logit in
                                                zip(fake_logits, real_logits)]).mean().detach().cpu().numpy()
        else:
            self.train_dis_acc += torch.cat([torch.le(fake_logits[-1].squeeze(1), .5),
                                             torch.gt(real_logits[-1].squeeze(1), .5)], dim=0).sum() / \
                                  (2 * torch.prod(torch.tensor(fake_logits[-1].size()))).detach().cpu().numpy()

        # Generator
        self.g_optimizer.zero_grad()

        fake_logits = self.discriminate(fake_images, input_labels)
        real_logits = self.discriminate(targets, input_labels)
        loss_G = self.gan_loss_fn(None, fake_logits)
        vgg_loss = self.vgg_loss_fn(fake_images, targets) * self.lambda_feat
        if self.get_disFeat:
            feat_loss = self.dis_loss_fn(fake_logits, real_logits)
        else:
            feat_loss = 0.
        loss_G += (vgg_loss + feat_loss)
        loss_G.backward(retain_graph=False)
        self.g_optimizer.step()

        self.train_gen_loss += loss_G.data.item()

        if isinstance(fake_logits[0], list):
            self.train_gen_acc += torch.tensor([torch.gt(fake_logit[-1].squeeze(1), .5).sum() /
                                                (torch.prod(torch.tensor(fake_logit[-1].size())))
                                                for fake_logit in fake_logits]).mean().detach().cpu().numpy()
        else:
            self.train_gen_acc += torch.gt(fake_logits[-1].squeeze(1), .5).sum() / \
                                  (torch.prod(torch.tensor(fake_logits[-1].size()))).mean().detach().cpu().numpy()

    def validate(self, labels, instances, targets):
        if self.device:
            labels = labels.to(self.device)
            instances = instances.to(self.device)
            targets = targets.to(self.device)

        edge_maps = get_edges(instances)
        input_labels = torch.cat([labels, edge_maps], dim=1)

        # Discriminator

        if self.use_eha:
            inst_maps = torch.argmax(labels, dim=1)
            feat_maps = self.enhancer([targets, inst_maps])
            fake_images = self.generator(torch.cat([input_labels, feat_maps], dim=1))
        else:
            fake_images = self.generator(input_labels)
        fake_logits = self.discriminate(fake_images.detach(), input_labels)
        real_logits = self.discriminate(targets, input_labels)
        loss_D = self.gan_loss_fn(fake_logits, real_logits)

        self.valid_dis_loss += loss_D.data.item()

        if isinstance(fake_logits[0], list):
            self.valid_dis_acc += torch.tensor([torch.cat([torch.le(fake_logit[-1].squeeze(1), .5),
                                                           torch.gt(real_logit[-1].squeeze(1), .5)], dim=0).sum() /
                                                (2 * torch.prod(torch.tensor(fake_logit[-1].size())))
                                                for fake_logit, real_logit in
                                                zip(fake_logits, real_logits)]).mean().detach().cpu().numpy()
        else:
            self.valid_dis_acc += torch.cat([torch.le(fake_logits[-1].squeeze(1), .5),
                                             torch.gt(real_logits[-1].squeeze(1), .5)], dim=0).sum() / \
                                  (2 * torch.prod(torch.tensor(fake_logits[-1].size()))).detach().cpu().numpy()

        # Generator

        fake_logits = self.discriminate(fake_images, input_labels)
        real_logits = self.discriminate(targets, input_labels)
        loss_G = self.gan_loss_fn(None, fake_logits)
        vgg_loss = self.vgg_loss_fn(fake_images, targets) * self.lambda_feat
        if self.get_disFeat:
            feat_loss = self.dis_loss_fn(fake_logits, real_logits)
        else:
            feat_loss = 0.
        loss_G += (vgg_loss + feat_loss)

        self.valid_gen_loss += loss_G.data.item()

        if isinstance(fake_logits[0], list):
            self.valid_gen_acc += torch.tensor([torch.gt(fake_logit[-1].squeeze(1), .5).sum() /
                                                (torch.prod(torch.tensor(fake_logit[-1].size())))
                                                for fake_logit in fake_logits]).mean().detach().cpu().numpy()
        else:
            self.valid_gen_acc += torch.gt(fake_logits[-1].squeeze(1), .5).sum() / \
                                  (torch.prod(torch.tensor(fake_logits[-1].size()))).mean().detach().cpu().numpy()

    def generate_sample(self, labels, instances, targets, batch):
        if self.device:
            labels = labels.to(self.device)
            instances = instances.to(self.device)
            targets = targets.to(self.device)

        edge_maps = get_edges(instances)
        input_labels = torch.cat([labels, edge_maps], dim=1)

        if labels.size(0) < 8:
            random_index = 0
        else: random_index = random.randint(0, 7)
        label = labels[random_index][None]
        input_label = input_labels[random_index][None]
        real_image = targets[random_index]

        if self.use_eha:
            inst_map = torch.argmax(label, dim=1)
            feat_map = self.enhancer([real_image[None], inst_map])
            fake_image = self.generator(torch.cat([input_label, feat_map], dim=1))
        else:
            fake_image = self.generator(input_label)

        fake_image = fake_image.cpu().detach().numpy()
        fake_image = np.clip(fake_image[0], -1., 1.)
        fake_image = (fake_image + 1) * 127.5
        real_image = (real_image.cpu().numpy() + 1) * 127.5
        fake_image = np.transpose(fake_image, [1, 2, 0])
        real_image = np.transpose(real_image, [1, 2, 0])

        image = np.concatenate([fake_image, real_image], axis=1)

        image = Image.fromarray(image.astype('uint8'))
        image.save(cfg.sample_path.format(batch), quality=95, subsampling=0)
