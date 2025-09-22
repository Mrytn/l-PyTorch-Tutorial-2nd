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
            # features 是 N×M 数组，每行一个特征向量
# targets 是长度 N 的列表，每个元素是该行特征向量对应的轨迹 ID
# 保证 特征和轨迹 ID 一一对应
# 重复多次id
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
        # 计算 目标检测框与已有轨迹之间的代价矩阵，并结合卡尔曼滤波的门控（gating）机制，剔除不可能的匹配
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # 从 dets（检测框列表）中，提取指定索引 detection_indices 的检测特征（ReID embedding 向量）。
# 结果是一个二维矩阵，例如 (num_detections, feature_dim)。
            features = np.array([dets[i].feature for i in detection_indices])
            # 从 tracks（轨迹列表）中，提取指定索引 track_indices 的 track_id。
# 注意：这里不是提取轨迹的特征向量，而是轨迹的 ID。
            targets = np.array([tracks[i].track_id for i in track_indices])
            # 外观距离矩阵
            # 每个元素表示某个轨迹与某个检测框之间的外观相似度（数值越小越相似
            cost_matrix = self.metric.distance(features, targets)
            # 卡尔曼滤波器 gating 对代价矩阵进行约束
            # 先根据卡尔曼滤波预测的轨迹状态（位置、速度），计算当前检测框是否在**门控范围（gating threshold）**内。
# 如果某个检测框太远（不可能是该轨迹），则在 cost_matrix 中对应位置赋予一个很大的值（例如 ∞）。
# 保证只在合理的空间范围内匹配检测和轨迹。
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
        # 遍历 self.tracks（当前维护的轨迹列表）。
# 筛选出 已确认（confirmed）的轨迹索引。
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        # unconfirmed_tracks刚新建、还没确认（未到 n_init 次数）的轨迹，不能直接用外观特征匹配（因为特征太少、不稳定）所以要在第二阶段用 IOU 去匹配
        # 这里的 unmatched_tracks_a 是第一阶段（外观特征匹配）后没有成功配对的“已确认轨迹”。
# 但是 只挑出那些“上一帧刚刚更新过”的轨迹（time_since_update == 1）。
# 为什么？因为这些轨迹的位置预测比较靠谱（只错过 1 帧），所以还值得再尝试用 IOU 匹配。
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        # 把那些 time_since_update == 1 的轨迹“移走”，因为它们已经加进 iou_track_candidates 里了。
# 剩下的 unmatched_tracks_a 就是那些没匹配成功、而且预测可能比较不可靠的轨迹（错过多于 1 帧）
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        # 调用 IOU 匹配：
# iou_cost 会计算轨迹框和检测框的 IOU 距离（1 - IOU）。
# max_iou_distance 是阈值（超过就不算可匹配）。
# 在候选轨迹 (iou_track_candidates) 和剩下的检测 (unmatched_detections) 之间做匈牙利算法匹配。
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        # 合并两次匹配结果
        matches = matches_a + matches_b
        # unmatched_tracks = 第一阶段没匹配上的（剩下的） + 第二阶段没匹配上的
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        # matches: 成功匹配的轨迹-检测对
# unmatched_tracks: 没匹配上的轨迹
# unmatched_detections: 没匹配上的检测
        return matches, unmatched_tracks, unmatched_detections
    # 建新轨迹（Track）
    def _initiate_track(self, detection):
        # 卡尔曼滤波器初始化
        # mean：均值向量，通常是一个 8 维向量（位置 + 速度）
        # covariance：协方差矩阵 (8×8)，表示不确定性（初始速度部分一般方差较大，因为观测不到）。
        mean, covariance = self.kf.initiate(detection.to_xyah())
        # 新建一个 Track 实例
        # mean / covariance：刚算出来的初始状态。
# self._next_id：轨迹的唯一 ID（自增）。
# self.n_init：确认需要的连续检测次数（防止假目标）。
# self.max_age：允许丢失多少帧后删除轨迹。
# detection.feature：该检测框的外观特征向量（ReID embedding），用于后续外观匹配。
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
