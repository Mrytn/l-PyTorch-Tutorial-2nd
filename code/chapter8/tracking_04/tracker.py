import cv2
import torch
import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
'''DeepSORT 多目标跟踪集成 YOLOv5 检测结果 的实现，核心作用是：
接收 YOLOv5 输出的检测框
用 DeepSORT 分配唯一 ID 跟踪每个目标
绘制检测框、ID、碰撞检查点'''

# DeepSORT 初始化
cfg = get_config()
cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
# REID_CKPT：特征提取模型（ReID）路径
# max_dist：特征匹配距离阈值
# min_confidence：最小置信度过滤
# max_age：多少帧内没匹配上还保留轨迹
# nn_budget：最大特征数缓存
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

'''bboxes 每个元素 (x1, y1, x2, y2, 类别, track_id)
在框上显示类别和 ID
绘制一个 碰撞检测点（框顶部 60% 高度）并用红色标记
支持自动计算线条粗细'''
# 绘制检测框函数
def draw_bboxes(image, bboxes, line_thickness):
    #如果没指定 line_thickness，则根据图像尺寸动态计算线宽，保证在大图上不显得太细。
    line_thickness = line_thickness or round(
        0.001 * (image.shape[0] + image.shape[1]) * 0.5) + 1
    # 存放小方形的四个顶点。
    list_pts = []
    # 小点的半径，绘制红点时用。
    point_radius = 4

    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)# 绿色边框

        # 撞线的点
        check_point_x = x1
        # check_point 是目标框中的一个参考点，取在 左边框（x1）+ 框高度 60% 的位置。
# 常用于计数、过线检测，比如“车轮经过某条线”。
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        # 用绿色矩形画目标框。
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
        # 设置字体的粗细
        font_thickness = max(line_thickness - 1, 1)
        # cv2.getTextSize 用来计算某段文字的 宽度和高度，返回 (width, height)
        # 第一个参数 cls_id → 要写的文字（类别 ID）。
# 第二个参数 0 → 字体类型，这里是 cv2.FONT_HERSHEY_SIMPLEX。
# fontScale=line_thickness / 3 → 根据边框的粗细动态调整文字大小。
# [0] 表示取结果的第一个元素（文字的尺寸 (w, h)），忽略基线。
# 作用：根据 cls_id 的字符串内容和字体设置，算出文字占多少空间。
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        # 横向加上文字宽度。
        # 纵向向上留出足够空间（文字高度 + 一点间距）。
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # -1 → 表示 填充矩形，而不是只画边框。
# cv2.LINE_AA → 抗锯齿绘制，更平滑。
        # 作用：在目标框的上方绘制一个实心矩形作为 文字背景，防止文字与画面混淆。
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # 在图像 image 上写文字。
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)
        # 用四个顶点围成一个矩形区域（小红点）。
# fillPoly：用红色填充该区域。
# list_pts.clear()：每次循环清空，避免和下一个 bbox 混在一起。
        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

        ndarray_pts = np.array(list_pts, np.int32)

        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))

        list_pts.clear()

    return image

'''更新跟踪
DeepSORT 需要 中心点 + 宽高 形式的 bbox
outputs 返回 (x1, y1, x2, y2, track_id)
然后用 search_label 将 DeepSORT 输出的每个跟踪框 匹配回 YOLOv5 的类别'''
def update(bboxes, image):
    # 保存转换后的检测框（DeepSort 输入格式）
    bbox_xywh = []
    # 保存置信度
    confs = []
    # 最后要返回的结果（含类别、跟踪 ID）
    bboxes2draw = []
    # 遍历检测框
    if len(bboxes) > 0:
        for x1, y1, x2, y2, lbl, conf in bboxes:
            obj = [
                int((x1 + x2) * 0.5), int((y1 + y2) * 0.5),
                x2 - x1, y2 - y1
            ]
            # 转换成 DeepSort 需要的 (center_x, center_y, w, h)。
# 同时存下检测框的置信度。
            bbox_xywh.append(obj)
            confs.append(conf)
        # 转为 Tensor
        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)
    # outputs：DeepSort 的跟踪结果，通常格式为 (x1, y1, x2, y2, track_id)。
# 内部会做 检测框关联 → 卡尔曼滤波 → 匈牙利匹配 → 输出带 ID 的跟踪框。
# 得到该帧所有目标的跟踪结果。
        outputs = deepsort.update(xywhs, confss, image)
        # 遍历每个跟踪框：
        for x1, y1, x2, y2, track_id in list(outputs):
            # x1, y1, x2, y2, track_id = value
            # 计算中心点 (center_x, center_y)。
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5
            # 调用 search_label：根据中心点去检测框里匹配类别 label，如果中心点距离检测框中心点 < 20.0 像素就认为是同一个目标。
            label = search_label(center_x=center_x, center_y=center_y,
                                 bboxes_xyxy=bboxes, max_dist_threshold=20.0)
            # 将结果 (x1, y1, x2, y2, label, track_id) 加入 bboxes2draw
            bboxes2draw.append((x1, y1, x2, y2, label, track_id))
        # 在 Python 里，pass 是一个占位语句，它什么都不做，只是用来保持语法结构的完整性。
        # 有时候语法要求必须有一条语句，但你又暂时不想写任何逻辑，这时候就用 pass 来“占坑”。
        pass
    pass

    return bboxes2draw

'''搜索类别函数
用 中心点距离 将跟踪 ID 对应到 YOLOv5 的检测类别
设置阈值 max_dist_threshold 避免匹配错误'''
# center_x, center_y：目标点（通常是跟踪框的中心点）。
# bboxes_xyxy：检测框列表，每个元素是 (x1, y1, x2, y2, lbl, conf)。
# max_dist_threshold：最大允许的横向/纵向距离，用来过滤掉差太远的检测框。
def search_label(center_x, center_y, bboxes_xyxy, max_dist_threshold):
    """
    在 yolov5 的 bbox 中搜索中心点最接近的label
    :param center_x:
    :param center_y:
    :param bboxes_xyxy:
    :param max_dist_threshold:
    :return: 字符串
    """
    # 最终返回的类别标签
    label = ''
    # min_label = ''
    # 记录“目前找到的最小距离”，初始值设为 -1.0 表示还没找到。
    min_dist = -1.0
    # 遍历检测框
    for x1, y1, x2, y2, lbl, conf in bboxes_xyxy:
        # 计算检测框中心点 (center_x2, center_y2)。
        center_x2 = (x1 + x2) * 0.5
        center_y2 = (y1 + y2) * 0.5

        # 横纵距离都小于 max_dist
        min_x = abs(center_x2 - center_x)
        min_y = abs(center_y2 - center_y)
        # 只有在横纵方向的差距都小于阈值时，才认为这个检测框可能是目标。
        if min_x < max_dist_threshold and min_y < max_dist_threshold:
            # 距离阈值，判断是否在允许误差范围内
            # 取 x, y 方向上的距离平均值
            avg_dist = (min_x + min_y) * 0.5
            if min_dist == -1.0:
                # 第一次赋值
                min_dist = avg_dist
                # 赋值label
                label = lbl
                pass
            else:
                # 若不是第一次，则距离小的优先
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    # label
                    label = lbl
                pass
            pass
        pass

    return label
