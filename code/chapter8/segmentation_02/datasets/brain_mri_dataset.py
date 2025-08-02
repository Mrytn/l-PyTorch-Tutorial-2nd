# -*- coding:utf-8 -*-
"""
@file name  : brain_mri_dataset.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-03-03
@brief      : brain mri 数据集读取
"""
import os
import random

import pandas as pd
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from skimage.io import imread


# 读取中文路径的图片
def cv_imread(path_file):
    # OpenCV 的标准读取函数 cv2.imread(path) 不支持包含中文或特殊字符的文件路径（尤其在 Windows 上），因为它使用的是 C++ 标准库的文件读取方式。
    # np.fromfile(path_file, dtype=np.uint8)
    # 作用：从指定路径的文件中读取二进制数据（字节流）。
    # 参数：
    # path_file：图像文件的完整路径（可以包含中文）。
    # dtype=np.uint8：以 8 位无符号整数读取，适用于图像数据。
    # 📌 这一部分相当于手动打开图片文件并把内容加载到内存中。
    # cv2.imdecode(...)
    # 作用：将上述字节数组解码成图像（类似于 cv2.imread 的功能）。
    # 参数：
    # 输入：np.ndarray（即字节流）
    # 解码标志位：cv2.IMREAD_UNCHANGED，表示读取图像时保持原始格式，包括 alpha 通道（透明度）等信息。
    cv_img = cv2.imdecode(np.fromfile(path_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return cv_img


class BrainMRIDataset(Dataset):
    def __init__(self, path_csv, transforms_=None):
        # 每一行csv包括
        # id,image_path,mask_path
        # 0,images/img0.png,masks/mask0.png
        # 1,images/img1.png,masks/mask1.png
        self.df = pd.read_csv(path_csv)
        self.transforms = transforms_

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 使用 iloc[idx, 1] 和 iloc[idx, 2] 获取图像路径和掩码路径（分别是第2列和第3列）
        image = cv_imread(self.df.iloc[idx, 1])
        mask = cv_imread(self.df.iloc[idx, 2])
        # 掩码通常是黑白图（背景是 0，前景是 255），此处将其转换为二值图（0 和 1）用于 二分类任务。
        mask[mask == 255] = 1  # 转换为0, 1 二分类标签

        if self.transforms:
            # 使用传入的增强函数对图像和掩码同步变换（保持配对关系）。
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        # mask.long() 是为了转换为 torch.int64 类型（PyTorch 训练时 loss 函数如 CrossEntropyLoss 要求标签为 LongTensor）。
        return image, mask.long()


if __name__ == "__main__":
    root_dir_train = r"../data_train.csv"  # path to your data
    root_dir_valid = r"../data_val.csv"  # path to your data

    train_set = BrainMRIDataset(root_dir_train)
    valid_set = BrainMRIDataset(root_dir_valid)

    train_loader = DataLoader(dataset=train_set, batch_size=2, shuffle=True)
    for i, (inputs, target) in enumerate(train_loader):
        print(i, inputs.shape, inputs, target.shape, target)
