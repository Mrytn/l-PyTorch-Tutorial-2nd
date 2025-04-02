# -*- coding:utf-8 -*-
"""
@file name  : 02_conv_kernel_fmap_vis.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-09
@brief      : 卷积核可视化，特征图可视化
"""
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.models as models
if __name__ == "__main__":
    # ----------------------------------- kernel visualization -----------------------------------
    # 可视化卷积层的卷积核
    writer = SummaryWriter(
        comment='kernel', filename_suffix="_test_your_filename_suffix")
    alexnet = models.alexnet(pretrained=True)
    kernel_num = -1
    vis_max = 1
    # 遍历alexnet的module
    for sub_module in alexnet.modules():
        # 非卷积层则跳过
        if isinstance(sub_module, nn.Conv2d):
            # 超过预期可视化的层数，则停止
            kernel_num += 1
            if kernel_num > vis_max:
                break
            kernels = sub_module.weight
            # 获取conv2d层的权重，即卷积核权重(输出通道数,输入通道数,卷积核的高度,宽度)
            c_out, c_int, k_w, k_h = tuple(kernels.shape)
            # 根据卷积核个数进行遍历
            for o_idx in range(c_out):
                # 此处是三维卷积核
                # 一个卷积核是4D的，包括两个通道 c_int, c_out,这里将每一个二维矩阵看作一个最细粒度的卷积核，进行绘制。
                # unsqueeze(1) 在维度 1 处增加一个维度kernel_idx 是形状 (c_in, 1, k_w, k_h) 的张量，相当于 c_in 张单通道的 k_w × k_h 的图片，make_grid 会将它们组合成一个网格。
                kernel_idx = kernels[o_idx, :, :, :].unsqueeze(
                    1)  # make_grid需要 BCHW，这里拓展C维度
                # 通过 make_grid 方法将卷积核转换为图像，并使用 writer.add_image() 将其添加到 TensorBoard 中进行可视化。
                # normalize=True：对卷积核的值进行归一化，使其在 [0,1] 范围内，便于可视化。
                # scale_each=True：对每个卷积核分别归一化，防止某个卷积核值范围过大导致其他卷积核的对比度过低。
                # nrow=c_int：指定网格的列数为 c_int（即输入通道数），这样每行 c_int 个卷积核（此处是二维卷积核），便于对比可视化。
                # 3个通道的卷积核，每个通道的卷积核都是一个11*11的矩阵
                kernel_grid = vutils.make_grid(
                    kernel_idx, normalize=True, scale_each=True, nrow=c_int)
                writer.add_image('{}_Convlayer_split_in_channel'.format(
                    kernel_num), kernel_grid, global_step=o_idx)
            # 对总的卷积核进行可视化
            # -1 让 PyTorch 自动推导 batch 维度，这里是 64
            kernel_all = kernels.view(-1, 3, k_h, k_w)  # 3, h, w
            # 一共有 64 个卷积核，nrow=8，所以总共 8 行 8 列，每个单元是一个 (3, 11, 11) 卷积核。
            kernel_grid = vutils.make_grid(
                kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
            writer.add_image('{}_all'.format(kernel_num),
                             kernel_grid, global_step=42)
            print("{}_convlayer shape:{}".format(
                kernel_num, tuple(kernels.shape)))
    writer.close()
    # ----------------------------------- feature map visualization -----------------------------------
    # 可视化输入图片的特征图
    writer = SummaryWriter(
        comment='fmap_vis', filename_suffix="_test_your_filename_suffix")
    # 数据
    # you can download lena from anywhere. tip: lena(Lena Soderberg, 莱娜·瑟德贝里)
    path_img = r"data\imgs\lena.png"  # your path to image
    # print("img.shape1", path_img.size)
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])
    # # 读取图片并转成 RGB
    img_pil = Image.open(path_img).convert('RGB')
    print("img.shape2", img_pil.size)
    if img_transforms is not None:
        img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw
    print("img.shape3", img_tensor.shape)
    # 模型
    alexnet = models.alexnet(pretrained=True)
    # forward
    convlayer1 = alexnet.features[0]
    # [1, 3, 224, 224]->[1, 64, 55, 55]
    fmap_1 = convlayer1(img_tensor)
    print("img.shape4", fmap_1.shape)
    # 预处理
    # 交换 batch 维度 和 通道维度
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    print("img.shape5", fmap_1.shape)
    fmap_1_grid = vutils.make_grid(
        fmap_1, normalize=True, scale_each=True, nrow=8)
    writer.add_image('feature map in conv1', fmap_1_grid, global_step=322)
    writer.close()
