# -*- coding:utf-8 -*-
"""
@file name  : 04_dataset_api.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-01-21
@brief      : dataset api 使用： concat\sub_set\random_split\
"""
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from PIL import Image
from torchvision.transforms import transforms
# os：用于操作文件和文件夹。
# pandas：用于处理 CSV 文件的数据。
# torch.utils.data：
# Dataset：用于定义自定义数据集类。
# DataLoader：用于加载数据集。
# ConcatDataset：用于合并多个数据集。
# Subset：用于从数据集中提取子集。
# random_split：用于随机拆分数据集。
# PIL.Image：用于读取和处理图片。
# torchvision.transforms：用于对图像进行变换（如归一化、翻转等）

# 该类用于加载数据集 1，其图片路径和标签信息存储在 txt 文件中。
class COVID19Dataset(Dataset):
    def __init__(self, root_dir, txt_path, transform=None):
        """
        获取数据集的路径、预处理的方法
        """
        # root_dir：数据集的根目录。
        # txt_path：存储图像文件名及其标签的 txt 文件路径。
        # transform：对图像进行变换（例如归一化）。
        # self.img_info：用于存储 (图片路径, 标签) 的列表。
        self.root_dir = root_dir
        self.txt_path = txt_path
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        # 其中 i.split()[0] 获取图片文件名，i.split()[2] 获取标签。
        path_img, label = self.img_info[index]
        # 读取 index 处的图片并转换为灰度模式（'L'）
        img = Image.open(path_img).convert('L')
        # 进行预处理（如果 transform 不为空
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path, label
        :return:
        """
        # 读取txt，解析txt
        with open(self.txt_path, "r") as f:
            txt_data = f.read().strip()
            txt_data = txt_data.split("\n")
        # i.split()表示用空格分割字符串，返回一个列表。
        self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
                         for i in txt_data]

# 该类用于加载数据集 3，图像路径和标签存储在 CSV 文件 中，并且训练集和验证集在同一文件中
class COVID19Dataset_3(Dataset):
    """
    对应数据集形式-3： 数据的划分及标签在csv中
    """

    def __init__(self, root_dir, path_csv, mode, transform=None):
        """
        获取数据集的路径、预处理的方法。由于数据划分体现在同一份文件中，因此需要设计 train/valid模式
        :param root_dir:
        :param path_csv:
        :param mode: str, train/valid
        :param transform:
        """
        self.root_dir = root_dir
        self.path_csv = path_csv
        self.mode = mode
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        # 由于标签信息是string，需要一个字典转换为模型训练时用的int类型
        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        # Image.open(path_img): 打开图像文件
# Image 是 PIL（Pillow）的一个模块，提供了图像处理的基本功能。
# open(path_img) 用于打开指定路径 path_img 处的图片文件，并返回一个 Image 对象。
# convert('L') 方法用于将图片转换为 灰度模式（Grayscale）。
# 'L' 模式表示每个像素点的颜色范围从 0（黑色）到 255（白色），中间值表示不同的灰度级别
        img = Image.open(path_img).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path, label
        :return:
        """
        df = pd.read_csv(self.path_csv)
        # 这行代码的作用是从 DataFrame 中删除不符合指定 mode 的行
        # df["set-type"] != self.mode
        # 这部分代码的作用是筛选出所有 set-type 不等于 self.mode 的行，返回一个布尔
        # df[df["set-type"] != self.mode].index
        # 这一步提取上一步中 True 的行索引
        # df.drop([1, 3], inplace=True) 这一步从 df 中删除索引 1 和 3 对应的行。
        # inplace=True 使修改直接作用于 df，不会返回新的 DataFrame，而是直接修改原来的 df。
        # df[df["set-type"] != self.mode]：使用布尔索引筛选出列 "set-type" 的值不等于 self.mode 的所有行。
# df[df["set-type"] != self.mode].index：获取筛选出的行的索引
        df.drop(df[df["set-type"] != self.mode].index,
                inplace=True)  # 只要对应的数据使索引从 0 开始连续编号
        df.reset_index(inplace=True)    # 非常重要！ pandas的drop不会改变index
        # 遍历表格，获取每张样本信息
        for idx in range(len(df)):
            path_img = os.path.join(self.root_dir, df.loc[idx, "img-name"])
            label_int = int(df.loc[idx, "label"])
            self.img_info.append((path_img, label_int))

# 该类用于加载数据集 2，图片存放在不同的文件夹，每个文件夹的名字就是类别
class COVID19Dataset_2(Dataset):
    """
    对应数据集形式-2： 数据的划分及标签在文件夹中体现
    """

    def __init__(self, root_dir, transform=None):
        """
        获取数据集的路径、预处理的方法，此时只需要根目录即可，其余信息通过文件目录获取
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []  # [(path, label), ... , ]
        self.label_array = None
        # 由于标签信息是string，需要一个字典转换为模型训练时用的int类型
        self.str_2_int = {"no-finding": 0, "covid-19": 1}

        self._get_img_info()

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index]
        img = Image.open(path_img).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))  # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path, label
        :return:
        """
        for root, dirs, files in os.walk(self.root_dir):  # 遍历根目录下的所有文件夹和文件
            for file in files:  # 遍历当前文件夹中的所有文件
                if file.endswith("png") or file.endswith("jpeg"):  # 只处理 PNG 和 JPEG 格式的图片
                    path_img = os.path.join(root, file)  # 获取图片的完整路径
                    sub_dir = os.path.basename(root)  # 获取当前图片所在的文件夹名称（类别名）
                    label_int = self.str_2_int[sub_dir]  # 将类别名转换为对应的整数标签
                    # 存储 (图片路径, 标签) 到 img_info 列表
                    self.img_info.append((path_img, label_int))


if __name__ == "__main__":
    # =============================== Concat ================================
    # set1
    # root_dir 是数据集 covid-19-demo 的根目录。
    # img_dir = os.path.join(root_dir, "imgs") 获取 imgs 目录的路径，其中存放图片。
    # path_txt_train 指向 labels/train.txt，通常包含图片路径和对应的标签。
    # train_data_1 = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train) 通过 COVID19Dataset 读取数据集。
    root_dir = r"data\datasets\covid-19-demo"  # path to datasets——covid-19-demo
    img_dir = os.path.join(root_dir, "imgs")
    path_txt_train = os.path.join(root_dir, "labels", "train.txt")
    train_data_1 = COVID19Dataset(root_dir=img_dir, txt_path=path_txt_train)
    # set2
    # root_dir_train 直接指向 train 目录，该目录可能包含类别文件夹，每个类别存放相应的图片。
    # train_data_2 = COVID19Dataset_2(root_dir_train) 使用 COVID19Dataset_2 读取该数据集。
    root_dir_train = r"data\datasets\covid-19-dataset-2\train"  # path to your data
    train_data_2 = COVID19Dataset_2(root_dir_train)
    # set3
    # root_dir 指向 covid-19-dataset-3/imgs，图片存放目录。
    # path_csv 指向 dataset-meta-data.csv，通常是一个 CSV 文件，记录了图片的路径、类别等信息。
    # train_data_3 = COVID19Dataset_3(root_dir, path_csv, "train") 读取该数据集，并指定使用 train 部分数据。
    root_dir = r"data\datasets\covid-19-dataset-3\imgs"  # path to your data
    path_csv = r"data\datasets\covid-19-dataset-3\dataset-meta-data.csv"
    train_data_3 = COVID19Dataset_3(root_dir, path_csv, "train")
    # concat
    # ConcatDataset：合并多个数据集，使它们成为一个大的数据集。
    train_set_all = ConcatDataset([train_data_1, train_data_2, train_data_3])
    print(len(train_set_all))  # 2 + 2 + 2 =6

    # =============================== sub set ================================
    # 从 train_set_all 中选择索引 [0, 1, 2, 5] 作为子数据集。
    train_sub_set = Subset(train_set_all, [0, 1, 2, 5])  # 将这4个样本抽出来构成子数据集
    print(len(train_sub_set))
    # =============================== random split ================================
    # random_split：将数据集随机拆分为两个部分：
# set_split_1 包含 4 个样本。
# set_split_2 包含 2 个样本
    set_split_1, set_split_2 = random_split(train_set_all, [4, 2])
    print(len(set_split_1), len(set_split_2))
