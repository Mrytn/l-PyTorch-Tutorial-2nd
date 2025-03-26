# -*- coding:utf-8 -*-
"""
@file name  : 05_hook_for_grad_cam.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-02-12
@brief      : 通过实现Grad-CAM学习module中的forward_hook和full_backward_hook函数
"""
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # self.conv1(x)：通过卷积层 conv1 对输入 x 进行卷积操作。x 是输入图像（假设是 3 通道 RGB 图像），卷积层会提取图像的特征。
        # F.relu(self.conv1(x))：卷积层的输出通过 ReLU 激活函数，增加网络的非线性能力。
        # self.pool1(F.relu(self.conv1(x)))：通过池化层 pool1，对卷积层的输出进行池化。池化操作有助于减少特征图的尺寸，并保留重要特征。这里是最大池化。
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# transform 是一个预处理函数，它会对图像执行一些操作，如转换为张量、调整大小、归一化等。PyTorch 中常见的 transform


def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    # Image.fromarray(np.uint8(img)) 将 numpy 数组转换为 PIL 图像。PIL 是 Python Imaging Library，用于处理图像的各种操作。此处将 img 强制转换为 uint8 类型，这是因为 PIL 库通常处理 uint8 类型的图像数据（0-255的像素值）。
    img = Image.fromarray(np.uint8(img))
    # 应用图像变换
    img = transform(img)
    # unsqueeze(0) 会在张量的第一个维度上插入一个新的维度，将形状从 (C, H, W) 变成 (1, C, H, W)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img, (32, 32))
    # :（第一个维度）：表示选取所有的行，即图像的所有高度。
# :（第二个维度）：表示选取所有的列，即图像的所有宽度。
# ::-1（第三个维度）：表示对颜色通道进行反转。::-1 是 Python 中的切片操作，用来反转数组的顺序。对于三通道的图像（RGB），这将会把红色通道（R）变成蓝色通道（B），绿色通道（G）保持不变，蓝色通道（B）变成红色通道（R）。因此，RGB图像会被转换为 BGR 格式。
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [
                             0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input
# 该 hook 在反向传播过程中被触发，用于获取梯度


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())
# 该 hook 在前向传播时触发，输出的是某一层的特征图


def farward_hook(module, input, output):
    fmap_block.append(output)
# 在原图上显示类激活图（CAM），并保存为图片


def show_cam_on_image(img, mask, out_dir):
    # 将生成的 CAM 掩码（mask）应用一个颜色映射，转化为热力图（通常使用 JET 颜色图）
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    # 将生成的 CAM 掩码（mask）应用一个颜色映射，转化为热力图（通常使用 JET 颜色图）
    heatmap = np.float32(heatmap) / 255
    # 将热力图与原始图像叠加，生成最终的 CAM 图像。
    cam = heatmap + np.float32(img)
    # 对 CAM 图像进行归一化处理，将其像素值调整到 [0, 1]
    cam = cam / np.max(cam)
    path_cam_img = os.path.join(out_dir, "cam.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 将生成的 CAM 图像保存为 cam.jpg
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    # 将原始图像保存为 raw.jpg
    cv2.imwrite(path_raw_img, np.uint8(255 * img))
    print("save img to :{}".format(path_cam_img))
# 该函数用于计算 类别向量，通常是用来生成类激活图时需要的目标类别的梯度。


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    # 这里检查 index 是否为 None 或空值。如果是空值（None 或 False），则从 output_vec 中选择一个类别索引。
    if not index:
        # output_vec 是网络输出的类分数向量，通常是一个包含各个类别概率或分数的张量。
        # cpu() 将数据移动到 CPU 上（如果之前在 GPU 上）。
        # data.numpy() 将张量转换为 NumPy 数组。
        # # np.argmax() 返回数组中最大值的索引，这个索引对应于网络预测的类别。
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        # 如果传入了 index 参数（即不是空值），则将其转换为 NumPy 数组
        index = np.array(index)
    # index[np.newaxis, np.newaxis] 通过 np.newaxis 增加新的轴，将 index 的形状从一维变为二维，形状为 (1, 1)。这样做是为了匹配后续操作中 one_hot 向量的形状（[1, 10]）。
# 假设 index 是一个单一类别的整数（例如 5），经过这一操作后，index 的形状变为 (1, 1)，内容是 [[5]]。
    index = index[np.newaxis, np.newaxis]
    # 将 index 从 NumPy 数组转换为 PyTorch 张量
    index = torch.from_numpy(index)
    # torch.zeros(1, 10) 创建一个大小为 [1, 10] 的全零张量
    # scatter_(1, index, 1) 操作将张量 one_hot 在第1维（即列）第 index 位置赋值为 1
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True
    # 这里计算了目标类别的 "类向量"（class_vec）。具体做法是，将 one_hot 和 output（网络的输出）进行逐元素相乘，再对结果求和。
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605
    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    # 这里创建一个全零的数组 cam，其形状是 feature_map 中的 高度（H） 和 宽度（W），也就是特征图的空间维度。
# feature_map.shape[1:] 代表 feature_map 的第二到最后一个维度，通常是 H 和 W，因此 cam 最终的形状为 [H, W]，是一个与特征图空间维度相同的矩阵。
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
    # axis=(1, 2) 表示对每个通道内的高度和宽度维度求均值。
    weights = np.mean(grads, axis=(1, 2))  #
    # 将当前通道 i 的特征图（feature_map[i, :, :]）乘以该通道的权重 w。
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    # 应用 ReLU（Rectified Linear Unit） 操作。ReLU 将 cam 中的所有负值置为 0，只保留正值。这是因为我们只关心类激活图中的正向贡献，负值通常没有实际意义。
    cam = np.maximum(cam, 0)
    # 使用 OpenCV 的 cv2.resize() 函数将生成的 CAM 图像大小调整为 32x32。通常，我们会将 CAM 图像缩放到与原始输入图像相同的大小，但这里为了演示，可能选择了一个固定的大小 32x32。
    cam = cv2.resize(cam, (32, 32))
    # 将 CAM 中的最小值减去，目的是 将 CAM 图像的最小值归零。这一步确保 CAM 图像的值域从 0 开始。
    cam -= np.min(cam)
    # 将 CAM 中的最大值除以，使得整个图像的值域被缩放到 [0, 1] 之间。这一步的目的是 归一化，确保 CAM 图像的数值处于 0 到 1 之间，方便后续的可视化处理（例如，通过 cv2.applyColorMap() 生成热力图）。
    cam /= np.max(cam)
    return cam

    # 根据 梯度 和 特征图 生成 类激活图（CAM） 的功能
# 通过计算梯度和特征图来生成类激活图（CAM），并将其可视化。
if __name__ == '__main__':
    # 当前脚本所在的目录路径
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = r"data/datasets/"
    # you can download the datasets from
    # https://pan.baidu.com/s/1eDwZchwp6P1Ab9d8Qn6rbA   code：l8qe
    path_img = os.path.join(BASE_DIR, "grad_cam_data",
                            "cam_img", "test_img_8.png")
# pkl 文件就是存储序列化后字节流的文件。
    path_net = os.path.join(BASE_DIR, "grad_cam_data", "net_params_72p.pkl")
    output_dir = os.path.join(
        BASE_DIR, "grad_cam_data", "results", "backward_hook_cam")
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    fmap_block = list()
    grad_block = list()
    # 图片读取；网络加载
    # cv2.imread() 读取图像，img 的形状为 H * W * C（高度、宽度和通道数）。
    # 1：以 彩色图像 读取图像（默认使用 BGR 通道顺序）。
# 0：以 灰度图像 读取图像。
# -1：读取图像时保留原始的 透明通道（Alpha channel），如果图像有的话。
    img = cv2.imread(path_img, 1)  # H*W*C
    # 进行图像预处理，准备输入到网络。
    img_input = img_preprocess(img)
    net = Net()
    # 加载预训练的网络权重
    net.load_state_dict(torch.load(path_net))
    # 注册hook
    # forward_hook 会在 conv2 层的前向传播时调用，用来提取特征图。
    net.conv2.register_forward_hook(farward_hook)
    # backward_hook 会在反向传播时调用，用来提取梯度。
    net.conv2.register_full_backward_hook(backward_hook)
    # forward
    # net(img_input) 进行前向传播，计算网络输出
    output = net(img_input)
    #  获取预测的类别索引（即最可能的类别）。
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))
    # backward
    # 清除之前的梯度。
    net.zero_grad()
    # 计算该类别的损失（即类别向量）。这里的 comp_class_vec 函数应该根据网络的输出向量计算出指定类别的 梯度损失。
    class_loss = comp_class_vec(output)
    # 执行反向传播，计算梯度。
    class_loss.backward()
    # 生成cam
    # 从 grad_block 和 fmap_block 中获取存储的梯度和特征图。
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    # gen_cam(fmap, grads_val) 调用前面定义的 gen_cam() 函数，生成类激活图（CAM）。
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)
    # 保存cam图片
    # 使用 OpenCV 将原图 img 缩放到 32x32 的大小，并归一化到 [0, 1] 范围。
    img_show = np.float32(cv2.resize(img, (32, 32))) / 255
    # 调用 show_cam_on_image(img_show, cam, output_dir) 将生成的 CAM 图和原图叠加，并保存到指定的输出目录
    show_cam_on_image(img_show, cam, output_dir)
