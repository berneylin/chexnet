#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Author: Weijie Lin
# @Email : berneylin@gmail.com
# @Date  : 2019-01-19

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from configs import *
from torch.utils.data import DataLoader


class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, transform=None):
        """
        初始化PyTorch的Dataset类以使用dataloader类
        :param path_to_img_dir: 存储ChestX-ray14 Dataset的图片文件的路径
        :param path_to_dataset_file: 存储`test.txt`, `train.txt`等文件的路径（训练集、测试集划分及图片所属的标签）
        :param transform: 对图片要做的transform处理
        """
        self.list_image_paths = []
        self.list_image_labels = []
        self.transform = transform
        with open(path_to_dataset_file, "r") as file_descriptor:
            lines = file_descriptor.readlines()
            for line in lines:
                line_items = line.split()
                image_path = os.path.join(path_to_img_dir, line_items[0])  # line_items为图片的相对路径信息
                image_label = line_items[1:]  # 从第二个开始，都为标签信息
                image_label = [int(i) for i in image_label]  # 通过list生成向量
                self.list_image_paths.append(image_path)
                self.list_image_labels.append(image_label)

    def __getitem__(self, index):
        """

        :param index: get item 时提供的索引index数值
        :return:
            imageData: 图片数据
            imageLabel: 标签信息
        """
        image_path = self.list_image_paths[index]
        image_data = Image.open(image_path).convert('RGB')
        image_label = torch.FloatTensor(self.list_image_labels[index])

        if self.transform:
            image_data = self.transform(image_data)

        return image_data, image_label

    def __len__(self):
        """

        :return:
            len: 数据集的总长度
        """
        return len(self.list_image_paths)


def get_train_dataloader(batch_size, shuffle, num_workers, transform_seq):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                     path_to_dataset_file=PATH_TO_TRAIN_FILE, transform=transform_seq)
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train


def get_validation_dataloader(batch_size, shuffle, num_workers, transform_seq):
    dataset_validation = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                          path_to_dataset_file=PATH_TO_VAL_FILE, transform=transform_seq)
    data_loader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_validation


def get_test_dataloader(batch_size, shuffle, num_workers, transform_seq):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR,
                                    path_to_dataset_file=PATH_TO_TEST_FILE, transform=transform_seq)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                   shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test