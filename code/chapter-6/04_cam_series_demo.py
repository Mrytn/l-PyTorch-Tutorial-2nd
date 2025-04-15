# -*- coding:utf-8 -*-
"""
@file name  : 04_cam_series_demo.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-156
@brief      : https://github.com/jacobgil/pytorch-grad-cam  学习与使用
安装：pip install grad_cam
"""
import cv2
import json
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from matplotlib import pyplot as plt
def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img
def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input
def cam_factory(cam_name_):
    """
    根据字符串动态创建 CAM 对象
    :param cam_name_: str, CAM 方法名称
    :return: 对应的 CAM 实例
    """
    # 例如 cam_factory("GradCAM") 会执行 GradCAM 这个类，并返回其实例。
    # eval 可能导致安全问题（代码执行风险），建议使用 getattr 替代
    # getattr(cam_modules, cam_name_)()
    return eval(cam_name_)


if __name__ == '__main__':
    path_img = "code/chapter-6/both.png"
    output_dir = "./Result"
    # 图片读取
    img = cv2.imread(path_img, 1)  # 读取彩色图片 (H*W*C)
    img_input = img_preprocess(img)  # 预处理图片
    model = resnet50(pretrained=True)  # 加载 ResNet50 预训练模型
    target_layers = [model.layer4[-1]]  # 选取 ResNet50 的最后一层作为目标层
    input_tensor = img_input
    cam_alg_list = "GradCAM,ScoreCAM,GradCAMPlusPlus,XGradCAM,EigenCAM,FullGrad".split(
        ",")
    plt.tight_layout()
    # fig, axs = plt.subplots(2, 3, figsize=(9, 9))
    fig, axs = plt.subplots(2, 3)
    for idx, cam_name in enumerate(cam_alg_list):
        # # 创建 CAM 实例
        cam = cam_factory(cam_name)(model=model, target_layers=target_layers)
        # targets = [e.g ClassifierOutputTarget(281)]
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        # # 计算 CAM 可视化结果
        # targets 通常指的是你希望计算激活图时关注的特定类别（不关注其他类别）。none不关注特定类别
        # # 取出第一张图片的结果
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)  # If targets is None, the highest scoring category
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        # img_norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        #  # 归一化原始图片
        img_norm = img/255.
        # 将 CAM 热力图（grayscale_cam）叠加到归一化后的原始图像（img_norm）上，生成带有热力图的图像。
        # img_norm：归一化后的原始图像（形状为 [H, W, C]，像素值在 0 到 1 之间）。
# grayscale_cam：计算得到的 CAM 热力图（一般为灰度图，形状为 [H, W]，表示每个像素点的关注程度）。
# use_rgb=False：指定是否将生成的热力图渲染为 RGB 格式。因为 grayscale_cam 是灰度图，通常不需要进行颜色转换，所以设置为 False，而是直接使用灰度热力图进行叠加
# grayscale_cam 是一个灰度图，它表示图像中每个像素点的关注程度，但仅仅是一个二维的灰度图（即单通道图像），不包含颜色信息。
# 然而，show_cam_on_image 函数并不是简单地将这个灰度热力图直接叠加到原图上，而是会使用一定的色彩映射来将热力图的数值映射到颜色空间中，从而生成一个“高亮”效果的图像。这种高亮效果表现为在热力图高值区域（即模型特别关注的区域）使用较为鲜艳的颜色（如红色或黄色），而在低值区域则使用较暗的颜色。
        visualization = show_cam_on_image(img_norm, grayscale_cam,
                                          use_rgb=False)
        # 将图像从 BGR 色彩空间转换为 RGB 色彩空间。
        vis_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
        # axs 是一个包含多个子图的 2D 数组（例如 2x3 子图，表示 2 行 3 列的排列）。
# ravel() 将 axs 数组展平为 1D 数组，方便通过索引来访问子图。
# [idx] 是当前子图的索引，imshow(vis_rgb) 将热力图显示在该子图
        im = axs.ravel()[idx].imshow(vis_rgb)
        axs.ravel()[idx].set_title(cam_name)
    plt.show()