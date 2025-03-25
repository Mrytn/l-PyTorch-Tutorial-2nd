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
    img = cv2.resize(img,(32, 32))
    # :（第一个维度）：表示选取所有的行，即图像的所有高度。
# :（第二个维度）：表示选取所有的列，即图像的所有宽度。
# ::-1（第三个维度）：表示对颜色通道进行反转。::-1 是 Python 中的切片操作，用来反转数组的顺序。对于三通道的图像（RGB），这将会把红色通道（R）变成蓝色通道（B），绿色通道（G）保持不变，蓝色通道（B）变成红色通道（R）。因此，RGB图像会被转换为 BGR 格式。
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
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
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    path_cam_img = os.path.join(out_dir, "cam.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))
    print("save img to :{}".format(path_cam_img))
def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605
    return class_vec
def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
    weights = np.mean(grads, axis=(1, 2))  #
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (32, 32))
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # you can download the datasets from
    # https://pan.baidu.com/s/1eDwZchwp6P1Ab9d8Qn6rbA   code：l8qe
    path_img = os.path.join(BASE_DIR, "grad_cam_data", "cam_img", "test_img_8.png")
    path_net = os.path.join(BASE_DIR, "grad_cam_data", "net_params_72p.pkl")
    output_dir = os.path.join(
        BASE_DIR, "grad_cam_data", "results", "backward_hook_cam")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fmap_block = list()
    grad_block = list()
    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)
    net = Net()
    net.load_state_dict(torch.load(path_net))
    # 注册hook
    net.conv2.register_forward_hook(farward_hook)
    net.conv2.register_full_backward_hook(backward_hook)
    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))
    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()
    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)
    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (32, 32))) / 255
    show_cam_on_image(img_show, cam, output_dir)
