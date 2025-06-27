# -*- coding:utf-8 -*-
"""
本代码主要由 chatGPT 编写！
本代码主要由 chatGPT 编写！
本代码主要由 chatGPT 编写！
"""
import imageio
import os
import cv2

# 读取文件夹中的所有图像文件
image_folder = 'gif_images'
images = []
for filename in os.listdir(image_folder):
    img = cv2.imread(os.path.join(image_folder, filename))
    # OpenCV 默认读取的是 BGR，而 imageio 使用 RGB，因此要转换。
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

# 生成GIF图像
# 使用 imageio.mimsave 保存为 GIF 格式。
# fps=2 表示每秒播放 2 张图像（即每帧间隔 0.5 秒）
imageio.mimsave('animation.gif', images, fps=2)
