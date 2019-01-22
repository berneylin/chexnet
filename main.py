#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Author: Weijie Lin
# @Email : berneylin@gmail.com
# @Date  : 2019-01-19

import os
import numpy as np
import time
import sys
from chexnet import CheXNet
from configs import *


def main():
    # start_train()
    start_test()


def start_train():
    print('Start training network model %s.' % NN_ARCHITECTURE)
    chexnet = CheXNet(mode='train', checkpoint=None)
    chexnet.train()


def start_test():
    print('Start testing network model %s.' % NN_ARCHITECTURE)
    chexnet = CheXNet(mode='test', checkpoint='model0122.pth')
    chexnet.test()


if __name__ == '__main__':
    main()
