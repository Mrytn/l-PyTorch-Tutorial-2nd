# -*- coding:utf-8 -*-
"""
@file name  : 04_grad_cam_pp.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-15
@brief      : Grad-CAM++ 演示
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
def load_class_names(p_clsnames):
    """
    加载标签名
    :param p_clsnames:
    :return:
    """
    # r表示以只读打开
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    return class_names
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
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [
                             0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input
def backward_hook(module, grad_in, grad_out):
    # detach() 是 PyTorch 张量的一个方法，其作用是创建一个与原张量共享数据，但不参与计算图构建的新张量
    grad_block.append(grad_out[0].detach())
def farward_hook(module, input, output):
    fmap_block.append(output)
# 显示并保存 CAM
def show_cam_on_image(img, mask, out_dir):
    #  # 生成热力图
    # OpenCV 提供的 伪彩色映射，使用 JET 颜色映射。
    # 红色区域表示高权重，蓝色区域表示低权重。
    heatmap = cv2.applyColorMap(
        np.uint8(255 * mask), cv2.COLORMAP_JET)  # 生成热力图
    heatmap = np.float32(heatmap) / 255  # 归一化
    # img 是原始输入图像（已归一化到 [0,1]）。
# heatmap 是伪彩色 CAM。
# 逐元素相加，得到 叠加后的可视化结果
    cam = heatmap + np.float32(img)  # 叠加原图
    cam = cam / np.max(cam)  # 归一化
    path_cam_img = os.path.join(out_dir, "cam.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))
def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        #  # 若无指定类别，则选择最高概率的类别
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    print("shape1", index.shape)
    # # 变成 (1, 1) 形状
    index = index[np.newaxis, np.newaxis]
    print("shape2", index.shape)
    index = torch.from_numpy(index)
    print("shape3", index.shape)
    # 形状 (1, 1000)，其中 index 位置是 1，其余是 0。
    one_hot = torch.zeros(1, 1000).scatter_(1, index, 1)
    one_hot.requires_grad = True
    print("shape4", ouput_vec.shape)
    class_vec = torch.sum(one_hot * ouput_vec)  # one_hot = 11.8605
    return class_vec
def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 初始化 CAM（H, W）
    weights = np.mean(grads, axis=(1, 2))  # 计算每个通道的全局平均梯度
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 加权求和
    cam = np.maximum(cam, 0)  # ReLU 操作，去除负值
    cam = cv2.resize(cam, (224, 224))  # 调整大小
    cam -= np.min(cam)
    cam /= np.max(cam)  # 归一化
    return cam
if __name__ == '__main__':
    path_img = "code/chapter-6/both.png"
    path_cls_names = "code/chapter-6/imagenet1000.json"
    output_dir = "Result"
    input_size = 224
    # ImageNet 1000 类别名称的 JSON 文件
    classes = load_class_names(path_cls_names)
    # 加载了 预训练 的 ResNet-50 模型，用于分类任务
    resnet_50 = models.resnet50(pretrained=True)
    fmap_block = []  # 存储前向传播的特征图
    grad_block = []  # 存储反向传播的梯度
    # 图片读取
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)
    # 注册hook
    # 存储特征图（网络前向传播时的输出）
    # 最后一个残差块。也就是说，使用负索引 -1 选取了 layer4 中的最后一个子模块。
    resnet_50.layer4[-1].register_forward_hook(farward_hook)
    # 存储梯度（损失函数对特征图的梯度）
    resnet_50.layer4[-1].register_full_backward_hook(backward_hook)
    # forward
    output = resnet_50(img_input)  # 执行 ResNet-50 前向传播
    idx = np.argmax(output.cpu().data.numpy())  # 取概率最大的类别索引
    print("predict: {}".format(classes[idx]))  # 输出预测类别
    # backward
    resnet_50.zero_grad()  # 清空梯度
    class_loss = comp_class_vec(output)  # 计算目标类别的损失
    class_loss.backward()  # 反向传播
    # 提取梯度和特征图
    grads_val = grad_block[0].cpu().data.numpy().squeeze()  # 取出梯度
    grads_val2 = grad_block[0].cpu().data.numpy()  # 取出梯度
    fmap = fmap_block[0].cpu().data.numpy().squeeze()  # 取出特征图
    fmap2 = fmap_block[0].cpu().data.numpy()  # 取出特征图
    print("f1: ", fmap.shape)
    print("f2: ", fmap2.shape)
    print("g1: ", grads_val.shape)
    print("f2: ", grads_val2.shape)
    # # 计算 CAM
    cam = gen_cam(fmap, grads_val)
    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (input_size, input_size))) / 255
    show_cam_on_image(img_show, cam, output_dir)
