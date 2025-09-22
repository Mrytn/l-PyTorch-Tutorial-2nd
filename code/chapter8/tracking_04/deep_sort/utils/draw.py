import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        # 通常会根据 ID 返回一个固定颜色，这样同一个目标每帧颜色一致。
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        # 返回文本的宽度和高度，用于绘制背景框。
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        # 绘制边界框
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        # 绘制文本背景
        # 在边框左上角绘制一个填充矩形作为文本背景
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        # 绘制文本
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
