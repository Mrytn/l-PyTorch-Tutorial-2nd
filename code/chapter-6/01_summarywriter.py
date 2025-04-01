# -*- coding:utf-8 -*-
"""
@file name  : 02_summarywriter.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-07
@brief      : tensorboard的writer 使用
"""
import os
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
if __name__ == "__main__":
    # ================================ add_scalar ================================
    # writer = SummaryWriter(comment="add_scalar", filename_suffix="_test_tensorboard")
    # x = range(100)
    # for i in x:
    #     writer.add_scalar('y=3x', i * 3, i)
    #     writer.add_scalar('Loss/train', np.random.random(), i)
    #     writer.add_scalar('Loss/Valid', np.random.random(), i)
    # writer.close()
    # ================================ add_scalars ================================
    # 该代码记录了训练损失和验证损失，用于比较两者的变化
    # writer = SummaryWriter(comment="add_scalars", filename_suffix="_test_tensorboard")
    # for i in range(100):
    #     # main_tag: 主标签（这里是 "Loss_curve"）
    #     # tag_scalar_dict: 相关标量的字典，如训练和验证损失
    #     # global_step: 当前步数
    #     writer.add_scalars('Loss_curve', {'train_loss': np.random.random(),
    #                                       'valid_loss': np.random.random()}, i)
    # writer.close()
    #     # ================================ add_histogram ================================
    #     # 这里每个 step 都会记录一个随机数分布直方图，并随 i 递增
    #     # tag: 名称（这里是 "distribution centers"）
    # # values: 需要记录的数值（如模型参数或数据分布）
    # # global_step: 当前步数
    # writer = SummaryWriter(comment="add_histogram", filename_suffix="_test_tensorboard")
    # for i in range(10):
    #     x = np.random.random(1000)
    #     # 这里通过 + i 使得每次循环的数据分布在不同位置（整体右移 i 个单位 ）；i 表示全局步数（global step），通常在训练过程中对应训练步数等，这里可理解为数据记录的次序标识
    #     writer.add_histogram('distribution centers', x + i, i)
    # writer.close()
    #     # ================================ add_image ================================
    #     # 记录单张图片
    #     # img 是 CHW 格式，(3, 100, 100)
    # # img_HWC 是 HWC 格式，(100, 100, 3)
    # # 记录了这两种格式的图片
    # writer = SummaryWriter(comment="add_image",
    #                        filename_suffix="_test_tensorboard")
    # # 创建一个形状为 (3, 100, 100) 的三维数组，用于存储图像数据，其中 3 表示图像的通道数
    # img = np.zeros((3, 100, 100))
    # # img[0]：从 0 到 1 的渐变，表示红色通道。
    # img[0] = np.arange(0, 10000).reshape(100, 100) / 10000
    # # img[1]：从 1 到 0 的渐变，表示绿色通道
    # img[1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    # img[2] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    # img_HWC = np.zeros((100, 100, 3))
    # img_HWC[:, :, 0] = np.arange(0, 10000).reshape(100, 100) / 10000
    # img_HWC[:, :, 1] = 1 - np.arange(0, 10000).reshape(100, 100) / 10000
    # # tag: 图片名称
    # # img_tensor: 图片数据（支持 CHW 或 HWC 格式）
    # # dataformats: 指定格式（CHW 或 HWC）
    # # 0: 当前步数
    # writer.add_image('my_image-shape:{}'.format(img.shape), img, 0)
    # print(img.shape)
    # writer.add_image('my_image_HWC-shape:{}'.format(img_HWC.shape),
    #                  img_HWC, 0, dataformats='HWC')
    # print(img_HWC.shape)
    # print(img_HWC.min(), img_HWC.max())
    # writer.close()
    #     # download dataset from
    #     # 链接：https://pan.baidu.com/s/1szfefHgGMeyh6IyfDggLzQ
    #     # 提取码：ruzz
    #     # 加载外部图片并记录
    # writer = SummaryWriter(comment="add_image",
    #                        filename_suffix="_test_tensorboard")
    # path_img = r"data\datasets\covid-19-dataset-3\imgs\ryct.2020200028.fig1a.jpeg"
    # img_opencv = cv2.imread(path_img)
    # writer.add_image('img_opencv_HWC-shape:{}'.format(img_opencv.shape), img_opencv, 0, dataformats='HWC')
    # writer.close()
    #     # ================================ add_images ================================
    #     # 记录多张图片
    #     # 代码模拟了一个 16 张图片的批次，每张图片的两个通道数据不同
    #     # img_tensor: 多张图片的批量数据，形状 (N, C, H, W)
    # # N: 批量大小（这里是 16）
    # # C: 通道数（这里是 3
    # writer = SummaryWriter(comment="add_images",
    #                        filename_suffix="_test_tensorboard")
    # img_batch = np.zeros((16, 3, 100, 100))
    # for i in range(16):
    #     img_batch[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
    #     img_batch[i, 1] = (
    #         1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
    # writer.add_images('add_images', img_batch, 0)
    # writer.close()
    #     # ================================ add_mesh ================================
    #     # 记录 3D 网格
    #     # 该代码创建了一个 3D 立方体网格并记录
    # import torch
    # vertices_tensor = torch.as_tensor([
    #     [1, 1, 1],
    #     [-1, -1, 1],
    #     [1, -1, -1],
    #     [-1, 1, -1],
    # ], dtype=torch.float).unsqueeze(0)
    # colors_tensor = torch.as_tensor([
    #     [255, 0, 0],
    #     [0, 255, 0],
    #     [0, 0, 255],
    #     [255, 0, 255],
    # ], dtype=torch.int).unsqueeze(0)
    # faces_tensor = torch.as_tensor([
    #     [0, 2, 3],
    #     [0, 3, 1],
    #     [0, 1, 2],
    #     [1, 3, 2],
    # ], dtype=torch.int).unsqueeze(0)
    # writer = SummaryWriter(comment="add_mesh",
    #                        filename_suffix="_test_tensorboard")
    # # vertices: 顶点坐标
    # # colors: 颜色
    # # faces: 三角面片索引
    # writer.add_mesh('add_mesh', vertices=vertices_tensor,
    #                 colors=colors_tensor, faces=faces_tensor)
    # writer.close()
    #     # ================================ add_hparams ================================
    #     # 这里记录了不同的学习率 (lr) 和批量大小 (bsize)，以及对应的准确率和损失
    writer = SummaryWriter(comment="add_hparams",
                           filename_suffix="_test_tensorboard")
    for i in range(5):
        # add_hparams(hparam_dict, metric_dict):
        # hparam_dict: 记录的超参数
        # metric_dict: 对应的评估指标
        writer.add_hparams({'lr': 0.1 * i, 'bsize': i},
                           {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i})
    writer.close()
