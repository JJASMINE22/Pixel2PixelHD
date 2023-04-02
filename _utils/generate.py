# -*- coding: UTF-8 -*-
'''
@Time : 2023/3/21 8:56
@Project : Pixel2PixelHD
@File : generate.py
@IDE : PyCharm 
@Author : XinYi Huang
@Email : m13541280433@163.com
'''
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image
from _utils import compose
from _utils.utils import DataWarp

import time

class Generator:
    def __init__(self,
                 anno_train_root: str,
                 anno_valid_root: str,
                 image_train_root: str,
                 image_valid_root: str,
                 fill_per: float,
                 batch_size: int,
                 scale_size: tuple,
                 image_size: tuple,
                 padding_size: tuple,
                 classes_num: int,
                 transform_ratio: float):
        self.fill_per = fill_per
        self.scale_size = scale_size
        self.image_size = image_size
        self.batch_size = batch_size
        self.classes_num = classes_num
        self.transform_ratio = transform_ratio
        self.anno_train_root = anno_train_root
        self.anno_valid_root = anno_valid_root
        self.image_train_root = image_train_root
        self.image_valid_root = image_valid_root

        self.data_warp = DataWarp(scale_size=scale_size,
                                  image_size=image_size,
                                  padding_size=padding_size)
        self.get_file_info()

    def get_file_info(self):

        self.train_images_paths = []
        for i, (root, _, files) in enumerate(os.walk(self.image_train_root)):
            if not i:
                continue
            else:
                for file in files:
                    file_path = os.path.join(root, file)
                    self.train_images_paths.append(file_path)

        self.valid_images_paths = []
        for i, (root, _, files) in enumerate(os.walk(self.image_valid_root)):
            if not i:
                continue
            else:
                for file in files:
                    file_path = os.path.join(root, file)
                    self.valid_images_paths.append(file_path)

        self.train_instances_paths = []
        self.train_labels_paths = []
        for i, (root, _, files) in enumerate(os.walk(self.anno_train_root)):
            if not i:
                continue
            else:
                for file in files:
                    if file.split("_")[-1] == "labelIds.png":
                        file_path = os.path.join(root, file)
                        self.train_labels_paths.append(file_path)
                    elif file.split("_")[-1] == "instanceIds.png":
                        file_path = os.path.join(root, file)
                        self.train_instances_paths.append(file_path)

        self.valid_instances_paths = []
        self.valid_labels_paths = []
        for i, (root, _, files) in enumerate(os.walk(self.anno_valid_root)):
            if not i:
                continue
            else:
                for file in files:
                    if file.split("_")[-1] == "labelIds.png":
                        file_path = os.path.join(root, file)
                        self.valid_labels_paths.append(file_path)
                    elif file.split("_")[-1] == "instanceIds.png":
                        file_path = os.path.join(root, file)
                        self.valid_instances_paths.append(file_path)

        if self.fill_per:
            fill_numbers = int(self.train_images_paths.__len__() * self.fill_per)
            fill_indices = np.random.choice(np.arange(self.train_images_paths.__len__()),
                                            size=(fill_numbers,), replace=False)
            train_image_fill_paths = np.array(self.train_images_paths)[fill_indices].tolist()
            train_label_fill_paths = np.array(self.train_labels_paths)[fill_indices].tolist()
            train_instance_fill_paths = np.array(self.train_instances_paths)[fill_indices].tolist()

            self.train_images_paths.extend(train_image_fill_paths)
            self.train_labels_paths.extend(train_label_fill_paths)
            self.train_instances_paths.extend(train_instance_fill_paths)

    def get_train_len(self):

        train_files_len = self.train_images_paths.__len__()
        if not train_files_len % self.batch_size:
            return train_files_len // self.batch_size
        else:
            return train_files_len // self.batch_size + 1

    def get_valid_len(self):

        valid_files_len = self.valid_images_paths.__len__()
        if not valid_files_len % self.batch_size:
            return valid_files_len // self.batch_size
        else:
            return valid_files_len // self.batch_size + 1

    def preprocess(self, file_path):

        image = Image.open(file_path)
        image = image.resize(self.image_size)
        image = np.array(image).astype("float")

        return image

    def image_preprocess(self, file_path):

        image = Image.open(file_path)
        image = image.resize(self.image_size)
        image = np.array(image).astype("float")
        image = image / 127.5 - 1.
        image = np.clip(image, -1., 1.)

        return image

    def generate(self, training=True):

        while True:

            if training:
                images_paths = np.array(self.train_images_paths.copy())
                labels_paths = np.array(self.train_labels_paths.copy())
                instances_paths = np.array(self.train_instances_paths.copy())
            else:
                images_paths = np.array(self.valid_images_paths.copy())
                labels_paths = np.array(self.valid_labels_paths.copy())
                instances_paths = np.array(self.valid_instances_paths.copy())

            random_index = [*range(images_paths.__len__())]
            random.shuffle(random_index)

            images_paths = images_paths[random_index]
            labels_paths = labels_paths[random_index]
            instances_paths = instances_paths[random_index]

            images, labels, instances = [], [], []
            for i, (image_path, label_path, instance_path) in enumerate(zip(images_paths,
                                                                            labels_paths,
                                                                            instances_paths)):
                label = self.preprocess(label_path)
                instance = self.preprocess(instance_path)
                label = np.tile(label[..., None], (1, 1, 3))
                instance = np.tile(instance[..., None], (1, 1, 3))
                image = self.image_preprocess(image_path)

                if random.random() > self.transform_ratio:
                    gridx_offset = np.random.normal(size=(self.scale_size[1], self.scale_size[0]), scale=3)
                    gridy_offset = np.random.normal(size=(self.scale_size[1], self.scale_size[0]), scale=3)
                    label = self.data_warp(label, gridx_offset, gridy_offset)
                    instance = self.data_warp(instance, gridx_offset, gridy_offset)
                    image = self.data_warp(image, gridx_offset, gridy_offset)
                label = torch.tensor(np.transpose(label, (2, 0, 1))[None])
                instance = torch.tensor(np.transpose(instance, (2, 0, 1))[None])
                image = torch.tensor(np.transpose(image, (2, 0, 1))[None])
                if random.random() > self.transform_ratio:
                    images_ = torch.cat([label, instance, image], dim=0)
                    images_ = compose(images_)
                    label, instance, image = torch.split(images_, split_size_or_sections=1, dim=0)

                label = label[:, 0].long()
                label = torch.eye(self.classes_num)[label]
                label = torch.permute(label, (0, 3, 1, 2))

                labels.append(label)
                instances.append(instance[:, 0:1])
                images.append(image)

                if images.__len__() == self.batch_size or i == images_paths.__len__() - 1:
                    if images.__len__() > 1:
                        anno_labels = torch.cat(labels.copy(), dim=0)
                        anno_instances = torch.cat(instances.copy(), dim=0)
                        anno_images = torch.cat(images.copy(), dim=0)
                    else:
                        anno_labels = labels.copy()[0]
                        anno_instances = instances.copy()[0]
                        anno_images = images.copy()[0]

                    images.clear()
                    instances.clear()
                    labels.clear()

                    yield anno_labels.float(), anno_instances.float(), anno_images.float()
