# -*- coding:utf-8 -*-
"""
@file name  : 06_albumentations-demo.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-26
@brief      : albumentations库学习
"""
import cv2
from matplotlib import pyplot as plt
if __name__ == "__main__":
    from albumentations import (
        # HorizontalFlip, Resize, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
        # Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
        # IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
        # IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
        HorizontalFlip, Resize, Blur, RandomRotate90
    )  # 图像变换函数

    path_img = r"data\imgs\lena.png"
    image = cv2.imread(path_img, 1)  # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用 Resize 缩放图像
    resize = Resize(width=224, height=224)  # DualTransform
    img_HorizontalFlip = resize(image=image)['image']
    #  使用 Blur 模糊图像
    blur = Blur(p=1)  # ImageOnlyTransform
    img_ShiftScaleRotate = blur(image=image)['image']
    # 使用 RandomRotate90 随机旋转图像 90 度的倍数
    rotate90 = RandomRotate90(p=1)
    img_RandomRotate90 = rotate90(image=image)['image']
    # 可视化四幅图像
    plt.subplot(221).imshow(image)
    plt.title("raw img")
    plt.subplot(222).imshow(img_HorizontalFlip)
    plt.subplot(223).imshow(img_ShiftScaleRotate)
    plt.subplot(224).imshow(img_RandomRotate90)
    plt.show()
