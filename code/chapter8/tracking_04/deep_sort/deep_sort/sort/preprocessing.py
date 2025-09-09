# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2

'''NMS，非极大值抑制）函数是检测后常用的 去重筛选 方法，避免同一个物体被多个重叠框重复检测。'''
def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    # boxes: ndarray，形状 (N, 4)，每行一个检测框 (x, y, w, h)
# (x, y) 左上角坐标
# (w, h) 宽高
# max_bbox_overlap: float，最大允许重叠比（IoU 阈值）
# scores: array_like，每个框的置信度（可选，若无则按 y2 排序）
    if len(boxes) == 0:
        return []
    # 转成浮点数避免整数计算误差
# pick 用来存储保留下来的框的索引
    boxes = boxes.astype(np.float32)
    pick = []
    # (x1, y1) 左上角
# (x2, y2) 右下角（x+w, y+h）
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]
    # 其实就是area = (width + 1) * (height + 1)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 如果有置信度，按 分数升序
# 否则，按 y2（右下角 y 坐标）升序
# 注意：后面每次取 idxs[last]，即分数 最高的框
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)
    # 每次挑选当前分数最高的框 i
# 把它加入结果 pick
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # w、h 和 overlap 都是向量（多个值），不是单个数
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        # np.where(overlap > max_bbox_overlap)[0] → 返回 重叠比例超过阈值的候选框在 idxs[:last] 中的索引
        # np.delete 从 idxs 数组中删除这些索引对应的元素
# 剩下的就是 下一轮循环要处理的候选框
# 这样就保证：
# 当前选中框不再参与下一轮
# 与当前框重叠过大的候选框也被删除
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick
