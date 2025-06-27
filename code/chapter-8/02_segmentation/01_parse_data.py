# -*- coding:utf-8 -*-
"""
@file name  : 01_parse_data.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-04
@brief      : 解析数据，保存为dataframe形式，并划分数据集
@reference  : https://www.kaggle.com/code/truthisneverlinear/tumor-segmentation-91-accuracy-pytorch
"""
import os
import pandas as pd
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import ImageGrid
from sklearn.model_selection import train_test_split


def cv_imread(path_file):
    cv_img = cv2.imdecode(np.fromfile(
        path_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # 返回值 numpy.ndarray
    return cv_img


def data_parse():
    """
    根据目录结构，读取图片、标签的路径及患者id
    :return:
    """
    data = []

    # 获取图片信息，存储于dataframe
    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                img_path = os.path.join(dir_path, filename)
                data.append([dir_, img_path])
        else:
            print(f'This is not a dir --> {dir_path}')

    # 分别获取图片与标签的路径信息
    df = pd.DataFrame(data, columns=["patient", "image_path"])
    # 过滤出非掩码图像的路径
# str.contains("mask")：返回布尔值，表示每个路径中是否包含 "mask"
# ~ 是取反操作，表示“不是掩码”的路径
    df_imgs = df[~df["image_path"].str.contains("mask")]
    # x[:-4]是截取第一个到倒数第5个
    # 假设原图是 "001_image.tif"，则掩码路径是 "001_image_mask.tif"
# 🔍 x[:-4] + "_mask.tif" 解释：
# x[:-4]：去掉 .tif 扩展名（因为 .tif 是 4 个字符）
# 然后拼接上 _mask.tif，形成完整掩码路径
    df_imgs["mask_path"] = df_imgs["image_path"].apply(
        lambda x: x[:-4] + "_mask.tif")

    # 最终df，包含患者id，图片路径，标签路径
    dff = df_imgs

    # 新增一列判断是否有肿瘤
    # 根据掩码图像是否有“前景像素”来判断该图像是否为“阳性”或“阴性”诊断结果，然后把这个诊断结果添加到 DataFrame 中，并保存成 CSV 文件
    def pos_neg_diagnosis(mask_path):
        # np.max(...)：查看图像中是否有非 0 像素
        return_value = 1 if np.max(cv_imread(mask_path)) > 0 else 0
        return return_value
    dff["diagnosis"] = dff["mask_path"].apply(lambda x: pos_neg_diagnosis(x))
    # 将包含 patient, image_path, mask_path, diagnosis 的 DataFrame 保存成 CSV 文件。
# index=False 表示不要保存行号索引列
    dff.to_csv(PATH_SAVE, index=False)


def data_analysis():
    """
    根据csv文件，分析正负图片比例，分析每个患者的正负图片比例
    :return:
    """

    dff = pd.read_csv(PATH_SAVE)
    # 对 diagnosis 列中不同的值（例如 Positive/Negative）进行计数，返回一个按频数降序排列的 Series
    # color=["violet", "orange"]：设置柱子的颜色列表，按分类顺序应用颜色。stacked=True 的作用是：在柱状图或折线图中，将同一个 x 轴标签下的多个数值“堆叠”在一起绘制，而不是分别并排显示
    ax = dff.diagnosis.value_counts().plot(kind="bar", stacked=True,
                                           figsize=(10, 6), color=["violet", "orange"])
    # 设置坐标轴 & 标题：
    ax.set_xticklabels(["Positive", "Negative"], rotation=45, fontsize=12)
    # 应该改为列表
    ax.set_yticklabels("Total Images", fontsize=12)
    ax.set_title("Distribution of Data Grouped by Diagnosis",
                 fontsize=18, y=1.05)
    # 给每个柱子添加标签
    # 这段代码在你画完柱状图后：
# 对每个柱子都添加了白色的数字标注；
# 位置是“柱子内部接近顶部”；
# 字体大而醒目，适合展示分类统计图
    for i, rows in enumerate(dff.diagnosis.value_counts().values):
        ax.annotate(
            int(rows),                  # 标注的文字内容（即计数值）
            xy=(i, rows - 12),          # 标注的位置，x 为柱子的索引，y 为柱子的高度减去 12
            rotation=0,                 # 水平文字（不旋转）
            color="white",              # 文字颜色为白色
            ha="center",                # 水平居中
            verticalalignment='bottom',  # 垂直底对齐（从指定点向上延伸）
            fontsize=15,                # 字号
            fontweight="bold"          # 加粗
        )

    # 在坐标 (x=1.2, y=2550) 的位置添加一段文本（图表坐标系下的位置）。

    # 文本内容为：例如 "Total 1000 images"。
    # bbox=...：添加了一个带圆角的蓝色背景框（用于突出显示文本）。
    # boxstyle="round"
# 设置文本框的形状（样式）。
# fc 是 facecolor 的简写，表示填充颜色。
# 你这里设置为 "lightblue"，即浅蓝色，用于作为文字框的背景。fc 是 facecolor 的简写，表示填充颜色。
# 你这里设置为 "lightblue"，即浅蓝色，用于作为文字框的背景。
# "round" 表示：圆角矩形，相比普通矩形更加柔和
    ax.text(1.2, 2550, f"Total {len(dff)} images", size=15, color="black", ha="center", va="center",
            bbox=dict(boxstyle="round", fc=("lightblue")))
    # 设置整个图表的背景色为深灰色（RGB 形式，值在 0~1 范围，0.15 ≈ 38/255）
    ax.set_facecolor((0.15, 0.15, 0.15))
    plt.show()
    # --------------------------------------- 观察正负图片数量比例 ------------------------------------------------------------
    patients_by_diagnosis = dff.groupby(["patient", "diagnosis"])[
        "diagnosis"].size().unstack().fillna(0)
    # 更保险的写法应该根据 diagnosis 原始值判断顺序
    # patients_by_diagnosis.columns = ['Positive' if col == 1 else 'Negative' for col in patients_by_diagnosis.columns]
    patients_by_diagnosis.columns = ["Positive", "Negative"]
    # 每一个柱子表示一个患者；
# 柱子中堆叠两种颜色：阳性图像和阴性图像的数量；
    ax = patients_by_diagnosis.plot(kind="bar", stacked=True, figsize=(18, 10), color=["violet", "springgreen"],
                                    alpha=0.85)
    ax.legend(fontsize=20, loc="upper left")
    ax.grid(False)
    ax.set_xlabel('Patients', fontsize=20)
    ax.set_ylabel('Total Images', fontsize=20)
    ax.set_title(
        "Distribution of data grouped by patient and diagnosis", fontsize=25, y=1.005)
    plt.show()


def data_visual():
    """
    将图片、标签读取并可视化
    :return:
    """
    dff = pd.read_csv(PATH_SAVE)

    # masks
    # 读取数据并抽样
    sample_df = dff[dff["diagnosis"] == 1].sample(5).values

    sample_imgs = []
    # 逐个读取图像与掩码，并缩放到一致大小
# 用 .extend() 把图像与掩码依次添加到 sample_imgs 列表中
    for i, data in enumerate(sample_df):
        img = cv2.resize(cv_imread(data[1]), (IMG_SHOW_SIZE, IMG_SHOW_SIZE))
        mask = cv2.resize(cv_imread(data[2]), (IMG_SHOW_SIZE, IMG_SHOW_SIZE))
        # 按顺序添加img, mask，分别为索引0,1
        sample_imgs.extend([img, mask])
    # 提取偶数索引，即原图
    sample_img_arr = np.hstack(sample_imgs[::2])
    # 提取奇数索引，即掩码
    sample_mask_arr = np.hstack(sample_imgs[1::2])

    # Plot
    fig = plt.figure(figsize=(25., 25.))
    # axes_pad=0.1：表示子图之间的间距为 0.1 英寸，控制图像之间的空白。
    # 111	继承自 subplot 的语法，表示占用整个画布区域
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 1),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    grid[0].imshow(sample_img_arr)
    grid[0].set_title("Images", fontsize=25)
    grid[0].axis("off")
    grid[0].grid(False)

    grid[1].imshow(sample_mask_arr)
    grid[1].set_title("Masks", fontsize=25, y=0.9)
    grid[1].axis("off")
    grid[1].grid(False)

    plt.show()


def data_split():
    """
    将数据划分为训练集、验证集，这里以dataframe形式存储
    :return:
    """
    dff = pd.read_csv(PATH_SAVE)

    # 需要根据患者维度划分，不可通过图片维度划分，以下代码可用于常见的csv划分
    grouped = dff.groupby('patient')
    # grouped = dff.groupby('image_path')  # bad method
    # 划分训练/验证集
    train_set, val_set = train_test_split(
        list(grouped), train_size=train_size, random_state=42)
    # ii[1] 是每组对应的 DataFrame（图像记录）
# 这一步把每个患者的数据提取出来放入列表中。
    train_set, val_set = [ii[1] for ii in train_set], [ii[1] for ii in val_set]  # 提取dataframe
    train_df, val_df = pd.concat(train_set), pd.concat(val_set)  # 合并dataframe

    train_df.to_csv(PATH_SAVE_TRAIN, index=False)
    val_df.to_csv(PATH_SAVE_VAL, index=False)
    print(f"Train: {train_df.shape} \nVal: {val_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--data-path", default=r"G:\deep_learning_data\brain-seg\kaggle_3m",
                        type=str, help="dataset path")
    # 将命令行输入的参数解析为 args 对象
    args = parser.parse_args()

    data_dir = args.data_path  # xxx/kaggle_3m
    PATH_SAVE = 'data_info.csv'
    # PATH_SAVE_TRAIN = 'data_train_split_by_img.csv'
    # PATH_SAVE_VAL = 'data_val_split_by_img.csv'
    PATH_SAVE_TRAIN = 'data_train.csv'
    PATH_SAVE_VAL = 'data_val.csv'
    IMG_SHOW_SIZE = 512  # 可视化时，图像大小
    train_size = 0.8  # 训练集划分比例，80%

    data_parse()  # 读取根目录下数据信息，存储为csv
    data_analysis()  # 分析数据数量、比例
    data_visual()  # 可视化原图与标签
    data_split()  # 划分训练集、验证集
