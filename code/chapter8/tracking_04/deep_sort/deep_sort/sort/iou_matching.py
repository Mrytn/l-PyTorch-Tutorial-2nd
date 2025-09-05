# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment

'''实现了 IOU (Intersection over Union) 匹配代价计算，主要用于目标跟踪（tracking）中的 数据关联 (data association)'''
# 计算一个目标框和多个候选框的 IOU（交并比）
# bbox: 单个目标框，格式 (x, y, w, h)（左上角坐标 + 宽高）
# candidates: 多个候选框，每一行 (x, y, w, h)
def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    # 分别是目标框的左上角和右下角坐标。
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    # 每个候选框的左上和右下角。
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    # 交集的左边界 x_left = 两个框左边界的 最大值
    # 交集的上边界 y_top = 两个框上边界的 最大值
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    # 交集的右边界 x_right = 两个框右边界的 最小值
# 交集的下边界 y_bottom = 两个框下边界的 最小值
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    # wh = np.maximum(0., br - tl)
    wh = np.maximum(0., br - tl)
    # a.prod() → 所有元素的乘积得到面积
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

'''把 IOU 转换成 代价矩阵 (cost matrix)，用于后续的 匈牙利匹配'''
# tracks: 跟踪到的目标（Track 对象列表）。
# detections: 新一帧的检测框（Detection 对象列表）。
# track_indices: 可选参数，指定要参与匹配的轨迹下标。
# detection_indices: 可选参数，指定要参与匹配的检测下标
def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    # 如果没有指定要匹配的轨迹或检测框，就默认 全部参与匹配。
# np.arange(len(tracks)) → [0, 1, 2, …, len(tracks)-1]
# 这样后面循环就可以直接用索引访问轨迹和检测框。
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    # 代价矩阵 shape = (num_tracks, num_detections)
# 每一行表示一个轨迹，每一列表示一个检测框
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    # 遍历轨迹，计算代价
    # time_since_update > 1 表示这个轨迹已经太久没更新了（可能被遮挡或丢失）
# 对这些轨迹，不做匹配，直接把整行赋为 无限代价
# continue 跳过后续计算
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue
        # 计算轨迹与检测框的 IOU
        # bbox：当前轨迹的预测框 (x, y, w, h)
# candidates：所有待匹配检测框的 (x, y, w, h)
# iou(bbox, candidates) → 返回长度 = num_detections 的数组
# 1 - iou → 代价矩阵中对应的这一行
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
