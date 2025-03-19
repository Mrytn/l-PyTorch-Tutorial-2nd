# -*- coding:utf-8 -*-
"""
@file name  : 03_sampler_unbalance.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-21
@brief      : WeightedRandomSampler  使用于不均衡数据集
"""
import os
import shutil
import collections
import torch
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision.transforms import transforms
# 创建一个不均衡的 CIFAR-10 训练数据集，即从 CIFAR-10 原始数据集中按照特定比例随机抽取不同类别的样本，并存放到新的目录


def make_fake_data(base_num):
    """
    制作虚拟数据集，
    :return:
    """
    root_dir = r"bigdata\tmp\cifar10\cifar10_train"
    out_dir = r"data\datasets\cifar-unbalance"
    import random
    for i in range(10):
        # 计算 类别 i 需要抽取的样本数量：
        # 类别 0：1 * base_num
        # 类别 1：2 * base_num
        # 类别 9：10 * base_num
        sample_num = (i + 1) * base_num
        # 读取类别 i 的所有样本
        sub_dir = os.path.join(root_dir, str(i))
        # 获取该类别所有图片的文件名列表
        path_imgs = os.listdir(sub_dir)
        # 打乱 图片顺序，保证抽取样本是随机的，而不是按照原始顺序
        random.shuffle(path_imgs)
        out_sub_dir = os.path.join(out_dir, str(i))
        # 如果目标目录不存在，则创建该文件夹。
        if not os.path.exists(out_sub_dir):
            os.makedirs(out_sub_dir)
            # 遍历 sample_num 计算出的目标数量，逐个抽取文件
        for j in range(sample_num):
            file_name = path_imgs[j]
            path_img = os.path.join(sub_dir, file_name)
            # 复制文件到目标目录
            shutil.copy(path_img, out_sub_dir)
    print("done")

# 该类 CifarDataset 是 torch.utils.data.Dataset 的子类，用于加载 CIFAR-10 数据集，并且支持数据增强（transform）


class CifarDataset(Dataset):
    # names：类别名称列表，对应 CIFAR-10 数据集的 10 个类别。
    # cls_num：类别总数，即 len(names) = 10。
    names = ('plane', 'car', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck')
    cls_num = len(names)

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []      # 定义list用于存储样本路径、标签
        self._get_img_info()

    def __getitem__(self, index):
        path_img, label = self.img_info[index]
        # 使用 PIL 读取图片，并转为 RGB 格式（确保三通道）。
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))   # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        for root, dirs, _ in os.walk(self.root_dir):
            # 遍历类别
            for sub_dir in dirs:
                # # 获取子目录下的所有图片文件
                img_names = os.listdir(os.path.join(root, sub_dir))
                # # 仅保留 PNG 格式图片
                img_names = list(
                    filter(lambda x: x.endswith('.png'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    # # 获取图片的绝对路径
                    path_img = os.path.abspath(
                        os.path.join(root, sub_dir, img_name))
                    # # 目录名即类别索引
                    label = int(sub_dir)
                    self.img_info.append((path_img, int(label)))
        random.shuffle(self.img_info)   # 将数据顺序打乱


if __name__ == "__main__":
    # make_fake_data()
    # 链接：https://pan.baidu.com/s/1ST85f8qgyKQucvKCBKbzug
    # 提取码：vf4j
    root_dir = r"data\datasets\cifar-unbalance"
    # 标准化，均值和标准差对应于 CIFAR-10 数据集的 RGB 通道统计值
    normalize = transforms.Normalize(
        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    transforms_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        normalize
    ])
    train_data = CifarDataset(root_dir=root_dir, transform=transforms_train)
    # 第一步：计算各类别的采样权重
    # 计算每个类的样本数量
    # 提取所有样本的类别标签。
    train_targets = [sample[1] for sample in train_data.img_info]
    # 统计每个类别的样本数量。
    label_counter = collections.Counter(train_targets)
    # 按类别 ID 排序后，获取每个类别的样本数量。
    class_sample_counts = [label_counter[k]
                           for k in sorted(label_counter)]  # 需要特别注意，此list的顺序！
    # 计算权重，利用倒数即可
    # 计算类别的采样权重，类别样本数越少，权重越高。
# 目的是让模型在训练时，能更均衡地学习所有类别。
    weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # weights = 12345. / torch.tensor(class_sample_counts, dtype=torch.float)
    # 第二步：生成每个样本的采样权重
    samples_weights = weights[train_targets]
    # 第三步：实例化WeightedRandomSampler
    # 用于基于权重的随机采样
    # 允许采样时重复选择相同样本，以确保训练过程中类别均衡
    sampler_w = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True)
    # 配置dataloader
    # 采用 WeightedRandomSampler 进行均衡采样。
    train_loader_sampler = DataLoader(
        dataset=train_data, batch_size=16, sampler=sampler_w)
    # 采用普通随机采样。
    train_loader = DataLoader(dataset=train_data, batch_size=16)

    def show_sample(loader):
        for epoch in range(10):
            label_count = []
            # 遍历 DataLoader，统计每轮训练中不同类别的样本数。
            # loader 是一个 DataLoader 对象，它会按批次（batch_size）返回数据。
            # enumerate(loader) 用于遍历 DataLoader，i 表示批次索引。
            # inputs 是当前批次的输入数据（图像张量）。
            # target 是当前批次的类别标签（张量）。
            for i, (inputs, target) in enumerate(loader):
                # target.tolist() 将 Tensor 转换为 Python 列表
                # label_count.extend(...) 将这个列表追加到 label_count
                label_count.extend(target.tolist())
                # collections.Counter(label_count) 统计 label_count 里每个类别的数量，并返回一个 Counter 字典
            print(collections.Counter(label_count))
    show_sample(train_loader)
    print("\n接下来运用sampler\n")
    show_sample(train_loader_sampler)
