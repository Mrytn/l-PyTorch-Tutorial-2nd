# -*- coding:utf-8 -*-
"""
@file name  : main.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-21
@brief      : 采用yolov5，基于区域撞线机制，实现双向目标计数
"""
'''程序做的事情可以分为四步：
视频读取 → 获取视频帧；
目标检测（YOLOv5） → 检出每帧中的车辆、行人等；
目标跟踪（tracker） → 追踪同一目标的 track_id；
撞线计数（BaseCounter） → 判断目标是否穿越了定义的区域边界，从而计数。
实现效果：
当物体从“外”区域进入“内”区域 → 计入“进入”计数；
当物体从“内”区域到“外”区域 → 计入“离开”计数。'''
import numpy as np
import tracker
import cv2
import copy
from detector import Detector

# 定义两种区域的标志值（相当于 mask 的像素编号）：
# inner：内层区域（蓝色区域）
# outer：外层区域（黄色区域）
# 用不同数值区分区域像素，后续通过 mask[y, x] 得到目标位于哪个区域。
class BoundaryType(object):
    """
    用于边界区域的mask像素填充，basecae：1和2, 由于用了插值，导致2的边界有一圈1，使得计数出错。
    采用了最近邻插值也会导致问题所在，为此，修改两个边界的索引像素，让它们差距大一些就好
    """
    inner = 68  # 内边界索引，用于矩阵像素赋值。从Outner-->inner，表示进入；蓝色区域
    outer = 168  # 外边界索引； 黄色区域

# 保存区域边界点；
# 生成 mask；
# 记录有哪些 track_id 进入此区域；
# 管理计数。
class CountBoundary(object):
    def __init__(self, point_set, mark_index, color, img_raw_shape, img_in_shape):
        """
        :param point_set:  list， # 边界点集， [(x, y), (x1, y1), ...] 要求是左上角开始，顺时针设置
        :param mark_index:  int，索引用的编号，用于区分是哪一个边界
        :param color:  list， [b, g, r]
        :param img_raw_shape: tuple, (w, h)，创建mask
        :param img_in_shape: tuple, (w, h)，缩放mask尺寸，模型输入输出的尺寸
        """
        self.point_set = point_set
        self.mark_index = mark_index
        self.color = color
        self.img_raw_shape = img_raw_shape
        self.img_in_shape = img_in_shape
        # id_container 的意义
# 保存所有经过该区域的 track_id；
# 当目标穿越区域边界后，用于判断是否计数；
# key 是 track_id，value 是累计编号。
        self.id_container = dict()  # 通过字典管理，key是track_id, value是进入边界的总数
        self.total_num = 0

        self._init_mask()

# _init_mask() —— 区域 mask 创建
# 使用 cv2.fillPoly() 根据顶点画出填充区域；
# 将 mask 从原视频尺寸缩放到模型输入尺寸；
# 得到用于显示的彩色 mask（方便叠加显示）。
    def _init_mask(self):
        ndarray_pts = np.array(self.point_set, np.int32)
        # 创建一个全黑的掩膜图像，大小与原始图像相同。
# 注意：self.img_raw_shape 通常是 (width, height) 或 (w, h, c)。
        mask_raw_ = np.zeros((self.img_raw_shape[1], self.img_raw_shape[0]), dtype=np.uint8)
        # 使用 OpenCV 的 fillPoly 方法在 mask_raw_ 上填充多边形区域。
# ndarray_pts 定义了多边形的顶点坐标。
# color=self.mark_index 决定填充颜色值（通常是 1, 2, 3 等标志值，用于区分不同区域）。
# 返回值 polygon_line_mask 是绘制后的掩膜。
        polygon_line_mask = cv2.fillPoly(mask_raw_, [ndarray_pts], color=self.mark_index)  # 绘制mask
        # 使用 OpenCV 的 fillPoly 方法在 mask_raw_ 上填充多边形区域。
# ndarray_pts 定义了多边形的顶点坐标。
# color=self.mark_index 决定填充颜色值（通常是 1, 2, 3 等标志值，用于区分不同区域）。
# 返回值 polygon_line_mask 是绘制后的掩膜。
        polygon_line_mask = polygon_line_mask[:, :, np.newaxis]  # 扩充维度
        # 将掩膜从原始图像大小缩放到模型输入大小。
# cv2.INTER_NEAREST 表示最近邻插值，避免缩放时掩膜的类别值被“平滑”。
# 比如 1 不会被变成 0.5。
        self.mask = cv2.resize(polygon_line_mask, self.img_in_shape, cv2.INTER_NEAREST)  # 缩放到模型输入尺寸
        self.mask = self.mask[:, :, np.newaxis]
        # 深拷贝一份掩膜，用于可视化，不影响后续逻辑。
        mask_ = copy.deepcopy(self.mask)
        # self.color 通常是一个三元组，比如 (0, 255, 0) 表示绿色。
# mask_ * self.color 会把掩膜区域涂上指定颜色，其它部分仍为 0。
        self.mask_color = np.array(mask_ * self.color, np.uint8)  # 可视化用的
    # 注册新的跟踪目标（即把新的 track_id 加入追踪列表）。
    def register_tracks(self, dets_id_list):
        for track_id in dets_id_list:
            self.add_id(track_id)  # x1, y1, x2, y2, label, track_id = bbox   # string, not int!
    # 移除丢失的或已经离开画面的目标 ID。
    def remove_tracks(self, dets_id_list):
        for track_id in dets_id_list:
            self.del_id(track_id)
    # 为一个新的目标 ID 分配编号并记录。
    def add_id(self, id_):
        self.total_num += 1
        self.id_container[id_] = self.total_num
    # 从追踪字典中删除指定 ID。
    def del_id(self, id_):
        self.id_container.pop(id_)

class BaseCounter(object):
    # 创建两个 CountBoundary 实例：
# inner_boundary：表示内层检测区域
# outer_boundary：表示外层检测区域
# 🎨 颜色说明
# [255, 0, 0]：红色（BGR格式），用于显示内边界
# [0, 255, 255]：黄色（青绿色），用于显示外边界
    def __init__(self, point_set, img_raw_shape, img_in_shape):
# 管理两个边界区域（inner / outer），并实现计数逻辑。

        self.inner_boundary = CountBoundary(point_set[0], BoundaryType.inner, [255, 0, 0], img_raw_shape, img_in_shape)
        self.outer_boundary = CountBoundary(point_set[1], BoundaryType.outer, [0, 255, 255], img_raw_shape, img_in_shape)
        # 将两个边界的掩膜相加，得到一个总的区域掩膜。
# 用于后续判断目标（比如人、车）是否进入该区域。
# 把内外边界的彩色 mask 相加，得到一个合并的可视化图像。
# 一般用于在视频帧上叠加显示，便于看到监控区域。
        self.area_mask = self.inner_boundary.mask + self.outer_boundary.mask
        self.color_img = self.inner_boundary.mask_color + self.outer_boundary.mask_color  # 用于外部绘图，size为img_inp
        # inner_total：统计进入内圈的目标数；
# outer_total：统计进入外圈的目标数。
        self.inner_total = 0
        self.outer_total = 0
# 根据目标（track）的当前位置与历史区域状态，判断目标是“从外到内”还是“从内到外”，并更新计数。
# 视频帧
#   │
#   ├── YOLO 检测出目标框 (bbox)
#   │
#   ├── 目标中心点 (cx, cy)
#   │
#   ├── 判断是否在 outer / inner 区域内（用 mask 检查）
#   │
#   ├── 如果目标从 outer → inner：inner_total += 1
#   │
#   ├── 如果目标从 inner → outer：outer_total += 1
#   │
#   └── 用 color_img 绘制结果
    def counting(self, tracks):
        # tracks 是当前帧中检测到的目标框列表。
# 每个元素通常是 [x1, y1, x2, y2, label, track_id]。
# 如果当前帧没有检测到目标（空列表），就直接返回。
        if len(tracks) == 0:
            return
        # bbox[0], bbox[1], bbox[2], bbox[3] 分别是左上角和右下角坐标；
# (x1+x2)/2, (y1+y2)/2 得到目标中心点；
# index_yx 是 (y坐标列表, x坐标列表) —— numpy 的索引是 先行(y) 再列(x)
        # 获取目标在mask上的像素，0， 1， 2组成的一个list
        index_x = [int((bbox[0]+bbox[2])/2) for bbox in tracks]  # x1, y1, x2, y2, label, track_id
        index_y = [int((bbox[1]+bbox[3])/2) for bbox in tracks]  # x1, y1, x2, y2, label, track_id
        index_yx = (index_y, index_x)  # numpy 是，yx
        # self.area_mask 是前面在 BaseCounter 初始化时生成的区域掩膜；
# 每个像素位置的值可能是：
# 0: 不在任何区域；
# 1: 在 inner 区域；
# 2: 在 outer 区域；
# 1 + 2 = 3: （理论上很少）在两个区域重叠处。
# 结果 bbox_area_list 就是一个与 tracks 对应的列表，比如：
        bbox_area_list = self.area_mask[index_yx]  # 获取bbox在图像中区域的索引，1,2分别表示在边界区域. [int,]

        # ======================== 先处理inner区域 ====================================
        # 遍历所有目标；
# 找出 bbox_area_list 中值为 1（inner区域）的目标；
# 提取它们的 track_id；
# 得到当前帧中所有处于 inner 区域的目标 ID。
        inner_tracks_currently_ids = self.get_currently_ids_by_area(tracks, bbox_area_list, BoundaryType.inner)
        # ↑这行有问题，为什么id-13的坐标是在outer的，但是返回的索引是1 ？
        # 这是上一帧（或之前几帧）中已经注册在 outer 区域 的所有目标；
# 它表示这些目标“曾经在 outer 区域中出现过”。
        outer_tracks_history_ids = list(self.outer_boundary.id_container.keys())  # 获取历史帧经过outer区域的目标的id

        # 当前与历史的交集，认为是目标从outer已经到达inner，可以计数，并且删除。
        # 当前在 inner 且 曾经在 outer 的目标
# ⇒ 说明该目标从外圈进入内圈，应当计数。
        outer_2_inner_tracks_id = self.intersection(inner_tracks_currently_ids, outer_tracks_history_ids)
        # 当前在 inner，但之前没在 outer。
# ⇒ 新出现的目标，只注册，不计数。
        only_at_inner_tracks_id = self.difference(inner_tracks_currently_ids, outer_tracks_history_ids)
        # 删除那些“从 outer → inner 已经计数过”的 ID；
# 把当前只在 inner 的目标注册到 inner 边界中；
# 这样下一帧就知道这些目标现在属于 inner 区域了。
        self.outer_boundary.remove_tracks(outer_2_inner_tracks_id)  # 删除outer中已计数的id
        self.inner_boundary.register_tracks(only_at_inner_tracks_id)  # 注册仅inner有的id

        if len(outer_2_inner_tracks_id):
            # 每当检测到新的 outer→inner 事件，就增加计数；
# 输出调试信息。
            self.inner_total += len(outer_2_inner_tracks_id)
            print('inner: {}， append: {}'.format(self.inner_total, outer_2_inner_tracks_id))

        # ======================== 处理outer区域 ====================================
        # 这部分代码是可以再抽象的，让inter与outer共用一个函数，但为了方便理解，就让它重复吧 2023年3月25日20:16:37 by TingsongYu
        # 一部分与上面 inner 几乎对称。
        outer_tracks_currently_ids = self.get_currently_ids_by_area(tracks, bbox_area_list, BoundaryType.outer)
        inner_tracks_history_ids = list(self.inner_boundary.id_container.keys())  # 获取历史帧经过output区域的目标

        # 当前与历史的交集， 存在则认为目标从inner已经到达outer，可以计数，并且删除。
        inner_2_outer_tracks_id = self.intersection(outer_tracks_currently_ids, inner_tracks_history_ids)
        only_at_outer_tracks_id = self.difference(outer_tracks_currently_ids, inner_tracks_history_ids)
        self.inner_boundary.remove_tracks(inner_2_outer_tracks_id)  # 删除inner中已计数的id
        self.outer_boundary.register_tracks(only_at_outer_tracks_id)  # 注册仅outer有的id

        if len(inner_2_outer_tracks_id):
            self.outer_total += len(inner_2_outer_tracks_id)
            print('outer: {}， append: {}'.format(self.outer_total, inner_2_outer_tracks_id))

    @staticmethod
    def get_currently_ids_by_area(tracks, bbox_area_list_, area_index):
        """
        判断跟踪框列表中，在区域1或2的框，的 track_id， 返回list
        :param tracks: list, 目标跟踪的输出
        :param bbox_area_list_: list, [int,] 目标位置对应于区域索引矩阵的索引，用于判断目标在区域1， 区域2，还是区域0
        :param area_index: int， 用于判断位于区域1，还是2。
        :return: list, [str,]
        """
        # 如果只有一个元素满足条件，.squeeze() 会返回一个 标量，而不是数组，这时直接索引会报错，需要做判断。
        area_bbox_index = np.argwhere(bbox_area_list_.squeeze() == area_index).squeeze()  # 进入边界区域 的bbox
        # 根据上一步得到的索引，选出当前帧中位于指定区域的 tracks
#         # tracks 是一个列表，每个元素通常是 [x1, y1, x2, y2, score, track_id]。
# np.array(tracks) 将列表转换为二维数组，方便用索引切片。
        area_tracks = np.array(tracks)[area_bbox_index]
        if len(area_tracks.shape) == 1:
            area_tracks = area_tracks[np.newaxis, :]
        # 取每行的最后一列，也就是 track_id
        area_tracks_currently_ids = list(area_tracks[:, -1])  # 获取当前帧在output区域的目标
        return area_tracks_currently_ids

    @staticmethod
    # 求两个列表 aa 和 bb 的交集。
# 返回一个列表，里面是同时出现在 aa 和 bb 中的元素。
    def intersection(aa, bb):
        return list(set(aa).intersection(set(bb)))

    @staticmethod
    # 求两个列表 aa 和 bb 的差集，即在 aa 中有但在 bb 中没有的元素。
    def difference(aa, bb):
        return list(set(aa).difference(set(bb)))  # a中有，b没有


def main():
    path_video = r'D:\ai\pytorch\l-PyTorch-Tutorial-2nd\bigdata\chapter-8\4\b\tracking\DJI_0049.MP4'
    # path_video = r'G:\DJI_0690.MP4'
    path_output_video = 'track_video.mp4'
    path_yolov5_ckpt = r'D:\ai\pytorch\l-PyTorch-Tutorial-2nd\bigdata\chapter-8\4\b\tracking\best.pt'
    # outer_point_set = [(1772, 1394), (2088, 1388), (2102, 1494), (1730, 1452)]
    # inner_point_set = [(1696, 1542), (2160, 1548), (2152, 1616), (1664, 1628)]
    # 0048
    # outer_point_set = [(845, 626), (1175, 630), (1175, 661), (838, 648)]
    # inner_point_set = [(822, 736), (1231, 735), (1227, 776), (796, 776)]
    # 0049
    # 内外区域用多边形点定义（顺序：左上、右上、右下、左下）
# 这些点用于生成 mask，判断目标进入/离开区域
    outer_point_set = [(616, 666), (1235, 655), (1245, 715), (600, 701)]
    inner_point_set = [(560, 808), (1238, 812), (1243, 848), (556, 837)]

    capture = cv2.VideoCapture(path_video)  # 打开视频
    # capture.get(3) → 视频宽度
# capture.get(4) → 视频高度
# 转成整数保存为 (宽, 高) 的元组
# raw_size_wh 表示原始视频帧尺寸
    w_raw, h_raw = int(capture.get(3)), int(capture.get(4))
    raw_size_wh = (w_raw, h_raw)  # w, h
    # 设置模型输入尺寸
    in_size_wh = (1280, 720)

    # 获取视频的帧率和帧数
    # 用于：
# 创建输出视频
# 统计视频处理进度
# 时间相关计算（比如秒数转换）
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建VideoWriter对象
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # VideoWriter 的参数：
# 输出文件路径
# 视频编码格式
# 帧率 fps
# 输出帧尺寸 in_size_wh
# out.write(frame) 后续可以把每帧写入输出视频
    out = cv2.VideoWriter(path_output_video, fourcc, fps, in_size_wh)
    # 初始化内外区域 mask
# 初始化计数器：inner_total 和 outer_total
    counter = BaseCounter([inner_point_set, outer_point_set], raw_size_wh, in_size_wh)
    detector = Detector(path_yolov5_ckpt)  # 初始化 yolov5
    # 设置计数文字显示位置
    # (x, y) 坐标，用于在输出帧显示计数文字
# 0.01 × 宽 → 左上角偏右一点
# 0.05 × 高 → 左上角偏下一点
# 这样文字不会贴到视频边缘，看起来更清晰
    draw_text_postion = (int(in_size_wh[0] * 0.01), int(in_size_wh[1] * 0.05))

    while True:
        # capture.read() 每次返回：
# _ → 返回值是否成功（True/False）
# im → 当前帧图像（HWC ndarray）
# 当视频读完时 im 为 None，循环结束
# ⚠️ 注意：视频逐帧读取可能慢，尤其是高清 4K 视频
        _, im = capture.read()  # 读取帧
        if im is None:
            break

        # 检测
        # 将视频帧缩放到模型输入尺寸
        im = cv2.resize(im, in_size_wh)  # im为HWC的 ndarray
        # 目标检测
        bboxes = detector.detect(im)  # bboxes是list，[(坐标(原尺寸), 分类字符串, 概率tensor), ]

        # 跟踪
        if len(bboxes) > 0:
            # tracker.draw_bboxes() → 在图像上画出跟踪框
# 如果当前帧没有检测目标，则：
# 直接输出原帧
# 跟踪列表为空
# ✅ 功能：保证每帧都有跟踪信息，并可视化
            list_bboxs = tracker.update(bboxes, im)  # 跟踪器跟踪
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)  # 画框
        else:
            output_image_frame = im
            list_bboxs = []

        # 撞线计数
        counter.counting(list_bboxs)  # 撞线计数

        # 图片可视化
        text_draw = "In: {}, Out: {}".format(counter.inner_total, counter.outer_total)
        # cv2.add() 进行 像素逐点相加，可以把 mask “叠加”在原帧上
# 这样输出的视频可以同时看到 原图 + 区域掩膜
        output_image_frame = cv2.add(output_image_frame, counter.color_img)  # 输出图片
        # 在帧上绘制文字（计数信息）
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=1, color=(255, 255, 255), thickness=2)
        # 写入输出视频
        # 输出视频文件 track_video.mp4 会包含：
# 原视频帧
# 叠加的区域 mask
# 计数文字
        out.write(output_image_frame)
        # 实时显示窗口
        cv2.imshow('demo', output_image_frame)
        # 延迟 1ms，让窗口刷新
        cv2.waitKey(1)
    # capture.release() → 关闭视频读取
# out.release() → 关闭视频写入
# cv2.destroyAllWindows() → 关闭所有 OpenCV 窗口
# 保证 文件正常保存 + 资源不占用
    capture.release()
    out.release()
    cv2.destroyAllWindows()
    # 打印最终计数
    print(text_draw)


if __name__ == '__main__':
    main()


