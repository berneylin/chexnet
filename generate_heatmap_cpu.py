#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Author: Weijie Lin
# @Email : berneylin@gmail.com
# @Date  : 2019-01-20

import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from densenet_models import DenseNet121, DenseNet169, DenseNet201
from configs import *


class HeatMapGenerator:
    def __init__(self):
        self.nn_architecture = NN_ARCHITECTURE
        self.pre_checkpoint = PRE_TRAINED_WEIGHTS_FOR_HEAT_MAP
        self.num_classes = NUM_CLASSES
        self.trans_crop = TRANS_CROP
        self.nn_is_pre_trained = True

        # initialize network model with pre-trained weights
        self.network_model = None
        try:
            if self.nn_architecture == 'DENSE-NET-121':
                self.network_model = DenseNet121(self.num_classes, self.nn_is_pre_trained)
            elif self.nn_architecture == 'DENSE-NET-169':
                self.network_model = DenseNet169(self.num_classes, self.nn_is_pre_trained)
            elif self.nn_architecture == 'DENSE-NET-201':
                self.network_model = DenseNet201(self.num_classes, self.nn_is_pre_trained)
            else:
                raise ValueError
        except ValueError:
            print("ValueError: CheXNet only supports nn_architecture == `DENSE-NET-***`(*** equals 121/169/201"
                  ". Not", self.nn_architecture)

        self.network_model = torch.nn.DataParallel(self.network_model)
        model_checkpoint = torch.load(self.pre_checkpoint, map_location='cpu')
        self.network_model.load_state_dict(model_checkpoint['state_dict'])

        # only need convolution layers' weights to generate heat map
        self.cam_model = self.network_model.module.dense_net_121.features
        self.cam_model.eval()  # set network as evaluation model without BN & Dropout
        self.network_model.eval()  # set network as evaluation model without BN & Dropout

        # CPU runtime, annotate below line to disable cuda
        # self.network_model.cuda()

        # select the last convolution layers' weights as weights of CAM
        self.weights = list(self.cam_model.parameters())[-2]

        # initialize the images transform sequence
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transform_list = list()
        transform_list.append(transforms.Resize(self.trans_crop))
        transform_list.append(transforms.ToTensor())
        transform_list.append(normalize)
        self.transform_sequence = transforms.Compose(transform_list)

    def generator(self, path_to_raw_image, path_to_output_image, trans_crop=None):
        if trans_crop is None:
            trans_crop = self.trans_crop

        # load image, transform, convert
        image_data = Image.open(path_to_raw_image).convert('RGB')
        image_data = self.transform_sequence(image_data)
        image_data = image_data.unsqueeze_(0)

        var_image = torch.autograd.Variable(image_data)

        with torch.autograd.no_grad():
            var_output = self.cam_model(var_image)
            var_prediction = self.network_model(var_image)

        # output predicted result
        predict_probability = np.max(var_prediction.data.numpy())
        predict_label = CLASS_NAMES[np.argmax(var_prediction.data.numpy())]
        print(predict_label, predict_probability)

        # start generating heat map
        heat_map = None
        for i in range(len(self.weights)):
            tmp_map = var_output[0, i, :, :]
            if i == 0:
                heat_map = self.weights[i] * tmp_map
            else:
                heat_map += self.weights[i] * tmp_map

        # convert torch tensor to numpy nd-array
        heat_map = heat_map.data.numpy()

        # blend raw image and heat map
        raw_img = cv2.imread(path_to_raw_image, 1)
        raw_img = cv2.resize(raw_img, (trans_crop, trans_crop))

        cam = heat_map / np.max(heat_map)
        cam = cv2.resize(cam, (trans_crop, trans_crop))
        heat_map = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

        output_img = heat_map * 0.5 + raw_img

        # save heat map image
        cv2.imwrite(path_to_output_image, output_img)


if __name__ == '__main__':
    path_to_raw_image = './heat_map/raw.png'
    path_to_output_image = './heat_map/out.png'

    heat_map_generator = HeatMapGenerator()
    heat_map_generator.generator(path_to_raw_image, path_to_output_image)

