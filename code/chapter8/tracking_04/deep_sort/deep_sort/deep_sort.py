import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

__all__ = ['DeepSort']

# max_dist=0.2
# 用于度量相似度的阈值，常用于 ReID 特征的余弦距离，越小表示匹配要求越严格。
# min_confidence=0.3
# 检测框的最低置信度阈值，低于这个值的检测结果会被过滤掉。
# nms_max_overlap=1.0
# 用于非极大值抑制（NMS）的最大重叠比例，默认 1.0 表示不做限制。
# max_iou_distance=0.7
# IOU 匹配时的最大允许距离，超过这个阈值就认为不是同一个目标。
# max_age=70
# 表示目标在多少帧没有被更新后会被删除（即“丢失”）。
# n_init=3
# 一个目标需要连续多少次匹配成功后才被确认为真实轨迹，避免误检。
# nn_budget=100
# 最近邻特征存储的上限，防止内存无限增长。
# use_cuda=True
# 是否使用 GPU 加速。
class DeepSort(object):
    # 设置目标检测的置信度、IOU 阈值等参数。
# 初始化 ReID 特征提取器，负责提取目标的外观特征。
# 初始化 跟踪器，结合外观特征与 IOU 进行数据关联与轨迹管理。
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        # Extractor 用来加载一个 ReID 模型，根据检测到的目标图像裁剪区域提取外观特征向量。
# 这些特征用于后续的目标匹配（同一个人/物体的特征向量应该比较接近）。
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        # NearestNeighborDistanceMetric 用于定义目标之间的相似性计算方法。
# 这里使用 "cosine" 余弦距离来度量特征相似度。
# max_cosine_distance 表示两个特征向量之间的最大距离阈值。
# nn_budget=100 表示最多保存 100 个历史特征，旧的特征会被替换掉。
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # racker 是核心跟踪模块：
# 负责管理轨迹集合。
# 利用外观特征（ReID）和运动预测（卡尔曼滤波）来进行匹配。
# 参数说明：
# metric：用于外观匹配的度量方式。
# max_iou_distance：IOU 匹配的距离阈值。
# max_age：目标最大丢失帧数。
# n_init：确认一个新轨迹所需的连续匹配次数。
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
    # bbox_xywh: 检测框列表，格式是 [x_center, y_center, width, height]。
# confidences: 每个检测框的置信度分数。
# ori_img: 原始图像，用来提取外观特征。
    def update(self, bbox_xywh, confidences, ori_img):
        # 保存图像的高度和宽度，方便后续归一化或边界检查。
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        # 调用 ReID 模型（Extractor）提取每个检测框的 外观特征向量。
# 这些特征会用于目标匹配。
        features = self._get_features(bbox_xywh, ori_img)
        # 将检测框从 (x_center, y_center, w, h) 转换为 (top_left_x, top_left_y, w, h)。
# DeepSort 内部通常用 左上角 + 宽高 表示框。
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        # 为每个检测框创建 Detection 对象，包含：
# 位置 bbox_tlwh[i]
# 置信度 conf
# 特征向量 features[i]
# 过滤掉低于 min_confidence 的框，避免无效目标。
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        # 非极大值抑制（NMS）
        # 为每个检测框创建 Detection 对象，包含：
# 位置 bbox_tlwh[i]
# 置信度 conf
# 特征向量 features[i]
# 过滤掉低于 min_confidence 的框，避免无效目标。
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        # 预测与更新轨迹
        # 调用卡尔曼滤波器预测所有轨迹在当前帧的位置（基于上一帧状态）。
        self.tracker.predict()
        # 用匈牙利算法（Hungarian Algorithm）做 检测与轨迹的匹配。
# 结合 IOU + 外观特征（ReID 余弦距离）。
# 更新成功匹配的轨迹，未匹配的轨迹会增加 time_since_update，超过 max_age 就会被删除。
        self.tracker.update(detections)

        # output bbox identities
        # 遍历所有轨迹：
# is_confirmed()：确保该轨迹是“确认过的”目标（至少连续匹配 n_init 次）。
# time_since_update <= 1：只保留最近更新过的轨迹，丢失太久的忽略。
# 取出轨迹的坐标 to_tlwh() → 再转成 (x1, y1, x2, y2)。
# x1, y1, x2, y2: 框的左上角和右下角坐标。
# track_id: 唯一的目标编号（跟踪 ID）。
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int32))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features
