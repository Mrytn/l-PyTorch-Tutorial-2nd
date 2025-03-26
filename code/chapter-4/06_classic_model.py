# -*- coding:utf-8 -*-
"""
@file name  : 06_classic_model.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-02-13
@brief      : torchvision 经典模型学习
"""
import torch
import torch.nn as nn
from torchvision import models
# 加载 AlexNet 模型。
model_alexnet = models.alexnet()
# 加载 VGG16 模型
model_vgg16 = models.vgg16()
# 加载 GoogLeNet（Inception v1） 模型。
model_googlenet = models.googlenet()
# 加载 ResNet-50 模型
model_resnet50 = models.resnet50()
# 遍历 model_alexnet.modules() 里的 所有层。
for m in model_alexnet.modules():
    if isinstance(m, torch.nn.Conv2d):
        # 如果当前层 m 是 torch.nn.Conv2d（卷积层），就用 Kaiming 正态分布初始化：
        # mode='fan_out'：适用于 前馈网络，保持梯度稳定。
        # nonlinearity='relu'：适用于 ReLU 激活函数，可以防止梯度消失。
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
