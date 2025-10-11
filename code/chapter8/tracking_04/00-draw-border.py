# -*- coding:utf-8 -*-
"""
@file name  : 00-draw-border.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-21
@brief      : 对视频进行选点， 双击图像，可获得坐标点的信息。 由于cv2.fillPoly函数的要求，需要从左上角，顺时针选点。
"""

import numpy as np
import cv2

'''在视频帧上选择点并记录坐标'''
def main():
    # path_video = r'G:\虎门大桥车流\DJI_0048.MP4'
    path_video = r'G:\虎门大桥车流\DJI_0047.MP4'
    # path_video = r'G:\DJI_0049.MP4'
    capture = cv2.VideoCapture(path_video)  # 打开视频
    scale = 0.8  # 图片缩放尺寸，若图片太大，可以缩小一些进行可视化、选点

    _, img = capture.read()# 读取视频第一帧
    img = cv2.resize(img, None, fx=scale, fy=scale)
    '''交互流程
双击添加点 → 绘制绿色多边形 → 打印原始坐标。
右键 → 清空点集合。
用途
可以用来标注多边形区域（ROI，感兴趣区域），比如车道范围、检测区域。
后续可将 point_set_raw 用于掩膜生成或区域过滤。'''
    # 鼠标交互
    def point_change(x_, y_, scale_):
        # point_change 把缩放坐标转换回原始视频尺寸坐标 (raw_x, raw_y)。
        return int(x_/scale_), int(y_/scale_)
# 鼠标事件回调函数
# (x, y) 是鼠标点击坐标（缩放后的）。
# 每次回调都会复制一张图 imgCopy 来绘制新的点或多边形。
    def mouseHandler(event, x, y, flags, param):
        global imgCopy
        imgCopy = img.copy()
        # 鼠标左键双击事件
        # 把点击点加入 point_set（缩放坐标）和 point_set_raw（原始坐标）。
# 用 cv2.fillPoly 绘制一个绿色多边形（mask）。
# 打印坐标信息。
# 鼠标左键双击
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # 输出坐标
            # 获取点击坐标 (x,y)
            raw_x, raw_y = point_change(x, y, scale)
            # 缩放后的坐标（方便画图）
            point_set.append((x, y))
            # 原始视频坐标（后续处理用）
            point_set_raw.append((raw_x, raw_y))

            point_arr = np.array(point_set)
            # 用 cv2.fillPoly 绘制绿色的多边形。
            imgCopy = cv2.fillPoly(imgCopy, [point_arr], color=[0, 255, 0])  # 绘制mask
            cv2.imshow('win', imgCopy)

            print("点坐标x,y:{},{}, 原始尺寸坐标为:{},{}".format(x, y, raw_x, raw_y))
            print("点前点集：{}".format(point_set_raw))
            # 清空所有已记录的点。
# 更新显示窗口。
# 鼠标右键单击
        elif event == cv2.EVENT_RBUTTONDOWN:
            point_set.clear()
            point_set_raw.clear()
            cv2.imshow('win', imgCopy)

# 注册鼠标回调与显示窗口
# 创建显示窗口。
    cv2.namedWindow('win')
    # 窗口与回调函数绑定
    global point_set, point_set_raw
    point_set = []
    point_set_raw = []
    # 绑定窗口与鼠标事件处理函数。
    cv2.setMouseCallback('win', mouseHandler)
    # 显示图像
    cv2.imshow('win', img)
    # 等待按键事件，否则窗口会立即关闭。
    cv2.waitKey()


if __name__ == '__main__':
    main()


