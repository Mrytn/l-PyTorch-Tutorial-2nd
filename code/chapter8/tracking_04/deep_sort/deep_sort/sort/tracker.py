# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

'''Deep SORT 的核心多目标跟踪器
匈牙利算法匹配
Kalman Filter 更新
新轨迹初始化
跟踪器状态管理
距离度量（ReID 特征）更新'''
# metric	NearestNeighborDistanceMetric	用于外观匹配（特征向量）计算代价矩阵
# max_iou_distance	float	用于 IOU 匹配的最大距离阈值
# max_age	int	轨迹连续未匹配的最大帧数，超过后删除轨迹
# n_init	int	轨迹确认前需要连续检测的帧数
# kf	KalmanFilter	Kalman 滤波器，用于轨迹预测和更新
# tracks	list[Track]	当前活跃轨迹列表
# _next_id	int	下一个轨迹的唯一 ID
class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # 对所有轨迹做 Kalman filter 预测，将状态分布推进到当前帧
# 会增加 age 和 time_since_update
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # _match → 根据 距离矩阵（欧式或余弦） + 级联匹配 得到匹配结果
# 输出：
# matches → 成功匹配的 (track_idx, detection_idx)
# unmatched_tracks → 没匹配到任何检测的轨迹索引
# unmatched_detections → 没匹配到任何轨迹的检测索引
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)   # 匈牙利算法匹配

        # Update track set.
        # 匹配成功 → 用检测更新 Kalman Filter 和 ReID 特征缓存
# update 方法前面讲过：
# 测量更新 (kf.update)
# 特征缓存 (track.features.append(detection.feature))
# hits 增加，time_since_update 重置
# Tentative → Confirmed 状态转换
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        # 没匹配到检测 → 调用 mark_missed()
# 逻辑：
# Tentative → 直接 Deleted
# Confirmed → 超过 _max_age 才 Deleted
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        # 没匹配到轨迹 → 新建轨迹（Tentative 状态）
# _initiate_track 通常会：
# 初始化 Kalman Filter 状态
# 给轨迹分配唯一 track_id
# features 里缓存 ReID 特征
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        # 清理掉状态为 Deleted 的轨迹
# 保留 Tentative 和 Confirmed 轨迹
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        # 更新距离度量（ReID 特征）
        # 收集所有 Confirmed 轨迹 的特征向量
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features → 所有特征向量
# targets → 对应的轨迹 ID
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            # features += track.features
# 将轨迹的所有特征向量加入总列表
# track.features 是 list，每个元素是检测帧提取的特征向量
            features += track.features
            # 每个特征向量对应轨迹 ID
            targets += [track.track_id for _ in track.features]
            # 清空轨迹特征缓存，避免重复加入下一帧
            track.features = []
        # 更新度量器
        # partial_fit() 会：
# 将新的特征加入 metric.samples 缓存
# 丢弃不再活跃的轨迹特征
# 如果设置了 budget，只保留最近 budget 个特征
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)  # 重要！记录所有目标的所有特征向量

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
