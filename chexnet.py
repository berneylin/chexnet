#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Author: Weijie Lin
# @Email : berneylin@gmail.com
# @Date  : 2019-01-19

from configs import *
import numpy as np
import time
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics.ranking import roc_auc_score
from densenet_models import *
from dataset_generator import *


class CheXNet:
    def __init__(self, mode='train', checkpoint=None):
        # Data Member Declarations:
        # ---- path_to_images_dir - path to the directory that contains images
        # ---- path_to_train_file - path to the file that contains image paths and label pairs (training set)
        # ---- path_to_validation_file - path to the file that contains image path and label pairs (validation set)
        # ---- nn_architecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
        # ---- nn_is_pre_trained - if True, uses pre-trained version of the network (pre-trained on ImageNet)
        # ---- num_classes - number of output classes
        # ---- batch_size - batch size
        # ---- trans_resize - size of the image to scale down to (not used in current implementation)
        # ---- trans_crop - size of the cropped image
        # ---- launch_time_stamp - date/time, used to assign unique name for the checkpoint file
        # ---- pre_checkpoint - if not None loads the model and continues training

        try:
            if mode == 'train':
                self.hyper_params_dict = TRAIN_DICT
            elif mode == 'test':
                if checkpoint is None:
                    raise ValueError(1)
                else:
                    self.hyper_params_dict = TEST_DICT
            else:
                raise ValueError(0)
        except ValueError(0):
            print("ValueError: CheXNet only supports mode == `train` or `test`. Not", str(mode))
        except ValueError(1):
            print("ValueError: Test mode needs specify pre trained weights dict file path. ")

        self.path_to_images_dir = PATH_TO_IMAGES_DIR
        self.path_to_train_file = PATH_TO_TRAIN_FILE
        self.path_to_validation_file = PATH_TO_VAL_FILE
        self.path_to_test_file = PATH_TO_TEST_FILE
        self.nn_architecture = NN_ARCHITECTURE
        self.nn_is_pre_trained = NN_IS_PRE_TRAINED
        self.num_classes = NUM_CLASSES

        self.trans_resize = TRANS_RESIZE
        self.trans_crop = TRANS_CROP
        self.launch_time_stamp = time.strftime("%H%M%S") + '-' + time.strftime("%d%m%Y")
        self.pre_checkpoint = checkpoint

    @staticmethod
    def get_current_time():
        timestamp_time = time.strftime("%H:%M:%S")
        timestamp_date = time.strftime("%Y/%m/%d")
        current_time = timestamp_date + '-' + timestamp_time  # get start timestamp
        return current_time

    def train(self):
        # Define Network Architecture
        network_model = None
        try:
            if self.nn_architecture == 'DENSE-NET-121':
                network_model = DenseNet121(self.num_classes, self.nn_is_pre_trained).cuda()
            elif self.nn_architecture == 'DENSE-NET-169':
                network_model = DenseNet169(self.num_classes, self.nn_is_pre_trained).cuda()
            elif self.nn_architecture == 'DENSE-NET-201':
                network_model = DenseNet201(self.num_classes, self.nn_is_pre_trained).cuda()
            else:
                raise ValueError
        except ValueError:
            print("ValueError: CheXNet only supports nn_architecture == `DENSE-NET-***`(*** equals 121/169/201"
                  ". Not", self.nn_architecture)

        network_model = torch.nn.DataParallel(network_model).cuda()  # make model available multi GPU cores training
        torch.backends.cudnn.benchmark = True  # improve train speed slightly
        # normalize data with ImageNet mean and standard deviation
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # compose transform operations
        transform_list = list()
        transform_list.append(transforms.RandomResizedCrop(self.trans_crop))
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        transform_list.append(normalize)
        transform_sequence = transforms.Compose(transform_list)

        # get data loader object
        dataloader_train = get_train_dataloader(batch_size=self.hyper_params_dict['Batch Size'],
                                                shuffle=True, num_workers=32, transform_seq=transform_sequence)
        dataloader_val = get_validation_dataloader(batch_size=self.hyper_params_dict['Batch Size'],
                                                   shuffle=False, num_workers=32, transform_seq=transform_sequence)

        # settings for optimizers and loss
        optimizer = optim.Adam(network_model.parameters(), lr=TRAIN_DICT['Learning Rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
        loss = torch.nn.BCELoss()

        # load pre-trained weights dict
        if self.pre_checkpoint:
            model_checkpoint = torch.load(self.pre_checkpoint)
            network_model.load_state_dict(model_checkpoint['state_dict'])
            optimizer.load_state_dict(model_checkpoint['optimizer'])

        # start training network
        min_loss = 999999999  # initialize min loss value with a large number

        for epoch_index in range(self.hyper_params_dict['Max Epoch']):
            # record time information
            prior_time = time.time()

            # Mini-Batch Training Process
            network_model.train()  # set network as train mode
            with torch.autograd.enable_grad():
                for batch_index, (image, label) in enumerate(dataloader_train):
                    label.cuda()
                    var_image = torch.autograd.Variable(image).cuda()
                    var_label = torch.autograd.Variable(label).cuda()
                    var_output = network_model(var_image)

                    loss_tensor = loss(var_output, var_label)
                    optimizer.zero_grad()
                    loss_tensor.backward()
                    optimizer.step()
                    current_time = self.get_current_time()
                    print("%s Epoch: %d, Step: %d, train loss: %.4f." % (current_time, epoch_index+1,
                                                                   batch_index+1, loss_tensor.item()))

            # Validation Process
            network_model.eval()  # set network as eval mode without BN & Dropout

            with torch.autograd.no_grad():
                loss_val = 0.
                mean_loss_tensor = 0.

                for batch_index, (image, label) in enumerate(dataloader_val):
                    label.cuda()
                    var_image = torch.autograd.Variable(image).cuda()
                    var_label = torch.autograd.Variable(label).cuda()
                    var_output = network_model(var_image)

                    curr_loss_tensor = loss(var_output, var_label)  # the output of loss() is a tensor
                    mean_loss_tensor += curr_loss_tensor  # tensor op.
                    loss_val += curr_loss_tensor.item()  # scalar op.

                    current_time = self.get_current_time()
                    print("%s Epoch: %d, Step: %d, validation loss: %.4f." % (current_time, epoch_index + 1,
                                                                              batch_index + 1, curr_loss_tensor.item()))

            loss_val = loss_val / len(dataloader_val)  # scalar
            mean_loss_tensor = mean_loss_tensor / len(dataloader_val)  # tensor

            # End validation process
            current_time = self.get_current_time()

            scheduler.step(mean_loss_tensor.item())
            spend_time_ms = 1000 * (time.time() - prior_time)

            if loss_val < min_loss:
                min_loss = loss_val
                torch.save({'epoch': epoch_index + 1, 'state_dict': network_model.state_dict(), 'best_loss': min_loss,
                            'optimizer': optimizer.state_dict()}, 'CheXNet-' + self.launch_time_stamp +
                           'loss' + str(min_loss)[0:9] + '.pth.tar')
                print("%s Epoch: %d, Val Loss: %f, Used time: %d ms. Saved!" % (current_time, epoch_index+1,
                                                                                loss_val, spend_time_ms))
            else:
                print("%s Epoch: %d, Val Loss: %f, Used time: %d ms." % (current_time, epoch_index+1,
                                                                         loss_val, spend_time_ms))

    def compute_auroc(self, ground_truth, prediction):
        out_auroc = []
        np_ground_truth = ground_truth.cpu().numpy()
        np_prediction = prediction.cpu().numpy()
        for i in range(self.num_classes):
            # calculate the roc_auc_score of each class
            out_auroc.append(roc_auc_score(np_ground_truth[:, i], np_prediction[:, i]))
        return out_auroc

    def test(self):
        # Define Network Architecture
        network_model = None
        try:
            if self.nn_architecture == 'DENSE-NET-121':
                network_model = DenseNet121(self.num_classes, self.nn_is_pre_trained).cuda()
            elif self.nn_architecture == 'DENSE-NET-169':
                network_model = DenseNet169(self.num_classes, self.nn_is_pre_trained).cuda()
            elif self.nn_architecture == 'DENSE-NET-201':
                network_model = DenseNet201(self.num_classes, self.nn_is_pre_trained).cuda()
            else:
                raise ValueError
        except ValueError:
            print("ValueError: CheXNet only supports nn_architecture == `DENSE-NET-***`(*** equals 121/169/201"
                  ". Not", self.nn_architecture)

        cudnn.benchmark = True  # improve test speed slightly

        network_model = torch.nn.DataParallel(network_model).cuda()
        model_checkpoint = torch.load(self.pre_checkpoint)
        network_model.load_state_dict(model_checkpoint['state_dict'])

        # normalize data with ImageNet mean and standard deviation
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # compose transform operations
        transform_list = list()
        transform_list.append(transforms.Resize(self.trans_resize))
        transform_list.append(transforms.TenCrop(self.trans_crop))
        transform_list.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop)
                                                                           for crop in crops])))
        transform_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_sequence = transforms.Compose(transform_list)

        # get test data loader
        data_loader_test = get_test_dataloader(batch_size=TEST_DICT['Batch Size'], shuffle=False,
                                               num_workers=16, transform_seq=transform_sequence)

        # initialize test output with tensor of type CUDA-float
        output_ground_truth = torch.FloatTensor().cuda()
        output_prediction = torch.FloatTensor().cuda()

        # start testing
        network_model.eval()  # set network as eval mode without BN & Dropout
        with torch.autograd.no_grad():
            for batch_index, (image, label) in enumerate(data_loader_test):
                label = label.cuda()
                output_ground_truth = torch.cat((output_ground_truth, label), 0)
                batch_size, n_crops, num_channels, height, width = image.size()
                var_image = torch.autograd.Variable(image.view(-1, num_channels, height, width).cuda())
                out = network_model(var_image)
                out_mean = out.view(batch_size, n_crops, -1).mean(1)
                output_prediction = torch.cat((output_prediction, out_mean.data), 0)

                current_time = self.get_current_time()  # get end timestamp
                print("%s Test batch index: %d finished." % (current_time, batch_index+1))

            auroc_individual = self.compute_auroc(output_ground_truth, output_prediction)
            auroc_mean = np.array(auroc_individual).mean()

        print('Mean AUROC is %f.' % auroc_mean)

        for i in range(len(auroc_individual)):
            print(CLASS_NAMES[i], auroc_individual[i])
