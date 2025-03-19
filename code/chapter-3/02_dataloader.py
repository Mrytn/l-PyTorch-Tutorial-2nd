# -*- coding:utf-8 -*-
"""
@file name  : 02_dataloader.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-22
@brief      : dataloader使用学习
"""
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms


class AntsBeesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        # 由于标签信息是string，需要一个字典转换为模型训练时用的int类型
        self.str_2_int = {"ants": 0, "bees": 1}
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        img = Image.open(path_img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith("jpg"):
                    path_img = os.path.join(root, file)
                    sub_dir = os.path.basename(root)
                    label_int = self.str_2_int[sub_dir]
                    self.img_info.append((path_img, label_int))


if __name__ == "__main__":
    # 链接：https://pan.baidu.com/s/1X11v5XEbdrgdgsVAESCVrA
    # 提取码：4wx1
    root_dir = r"data\datasets\mini-hymenoptera_data\train"
    # =========================== 配合 Dataloader ===================================
    # 均值 (mean) 和标准差 (std) 都有三个值是因为 图片通常是 RGB 3 通道 的数据，每个通道（Red, Green, Blue）都有各自的分布统计特性
    # 标准化：减去 ImageNet 训练集的均值 [0.485, 0.456, 0.406]，再除以标准差 [0.229, 0.224, 0.225]，确保数据分布一致。
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 来自ImageNet数据集统计值
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小到 (224, 224)
        transforms.ToTensor(),          # 转换为 PyTorch Tensor 格式
        normalize                       # 进行标准化
    ])
    # 加载数据集
    train_set = AntsBeesDataset(root_dir, transform=transforms_train)  # 加入transform

# DataLoader 负责将数据集 train_set 按批次 读取，并提供 自动打乱（shuffle） 和 批量加载（batching） 功能：
# batch_size=2：每次取 2 个样本
# batch_size=3：每次取 3 个样本
# shuffle=True：在每个 epoch 开始前，打乱数据顺序
# drop_last=True：如果最后剩余样本不足 batch_size=2，就丢弃最后的不足样本
    train_loader_bs2 = DataLoader(
        dataset=train_set, batch_size=2, shuffle=True)
    train_loader_bs3 = DataLoader(dataset=train_set, batch_size=3, shuffle=True)
    train_loader_bs2_drop = DataLoader(dataset=train_set, batch_size=2, shuffle=True, drop_last=True)

# 逐批遍历 train_loader_bs2，其中：
# inputs.shape = (batch_size, 3, 224, 224)，代表：
# batch_size=2（批次大小）
# 3（RGB 3 通道）
# 224 × 224（图像尺寸）
# target.shape = (batch_size,)，是 batch_size=2 对应的类别标签
# target：Tensor([0, 1])，表示当前批次的两张图片类别，如：
# 0 → ants
# 1 → bees
    for i, (inputs, target) in enumerate(train_loader_bs2):
        print(i, inputs.shape, target.shape, target)
    for i, (inputs, target) in enumerate(train_loader_bs3):
        print(i, inputs.shape, target.shape, target)
    for i, (inputs, target) in enumerate(train_loader_bs2_drop):
        print(i, inputs.shape, target.shape, target)

