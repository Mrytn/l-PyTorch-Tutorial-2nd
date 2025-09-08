# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter
'''匈牙利算法'''

INFTY_COST = 1e+5

# 多目标跟踪中的匹配问题：怎么把预测的轨迹（tracks）和当前检测结果（detections）对应起来。
def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    # 如果没有指定索引，就默认对所有 track 和 detection 进行匹配
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    # 没有目标或者没有检测，没法匹配，直接返回。
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.
    # distance_metric：通常是 ReID 特征的余弦距离或马氏距离，得到一个 N×M 的代价矩阵。
# 行对应轨迹（track）
# 列对应检测（detection）
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    # 超过 max_distance（阈值）的代价直接置为“无效”（大数）。
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    # linear_assignment 是匈牙利算法，解决的是 最小化总代价的匹配问题。
# 返回的是匹配的行、列索引（即 track ↔ detection）
    row_indices, col_indices = linear_assignment(cost_matrix)  # 匈牙利算法求解，得到配对的（raw, col）

    matches, unmatched_tracks, unmatched_detections = [], [], []
    # 未匹配的 detection
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    # 未匹配的 track
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        # 即使匈牙利算法给出一个匹配，如果代价太大（> 阈值），我们还是认为它“不可信”，强行归为 unmatched
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections

'''在 min_cost_matching 之上，增加了一个 分层级的匹配策略
在多目标跟踪（MOT）里，轨迹（track）可能会有不同的“新鲜程度”：
time_since_update = 1：刚更新过的轨迹（很新，可靠性高）。
time_since_update = 2, 3, ...：已经几帧没被检测到的轨迹（老化，不太可靠）。
👉 如果一上来就把所有轨迹和检测放在一起匹配，可能导致“老的轨迹”把新的检测抢走（错误关联）。
所以 matching_cascade 采用 逐层匹配 的方式，优先让“最新的轨迹”先匹配'''
def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    # 一开始，所有检测都“未匹配”。
    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        # 遍历每一层（level = 0, 1, 2, ...）。
# 如果没有检测剩下了，就提前结束。
        if len(unmatched_detections) == 0:  # No detections left
            break
        # 它用 列表推导式 遍历 track_indices 中的每个轨迹索引 k；
# 只把满足条件
# tracks[k].time_since_update == 1 + level
# 的 k 收集到新的列表 track_indices_l 里。
        # time_since_update = 1 → 轨迹“新鲜”，上一帧刚更新过。
# time_since_update = 2 → 上一帧确实没更新。
# time_since_update = 3 → 连续两帧没更新。
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level  # 为每个跟踪框记录它被更新的次数，优先选择新跟踪框进行匹配， 1+0
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue
        # 调用 min_cost_matching（就是你前面看的函数）。
# 当前层的轨迹与剩余未匹配的检测进行匹配。
# 更新 unmatched_detections，减少后续层的竞争
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    # 得到“所有轨迹”减去“已匹配轨迹” = 未匹配轨迹。
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    # matches: 已匹配的轨迹-检测对
# unmatched_tracks: 没有匹配上的轨迹（可能要增加丢失帧计数）
# unmatched_detections: 没有匹配上的检测（可能是新目标）
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
