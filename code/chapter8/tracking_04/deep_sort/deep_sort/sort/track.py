# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """
    # Tentative（试探）：新创建的轨迹，刚被检测到，还未确认。
# Confirmed（已确认）：连续检测到足够次数后，轨迹状态确认。
# Deleted（删除）：轨迹失效，需要从活跃轨迹列表中删除。
    Tentative = 1
    Confirmed = 2
    Deleted = 3

# Track 对象表示单个目标的轨迹，主要用于：
# 存储目标状态 (x, y, a, h)：
# (x, y) = 目标框中心
# a = 宽高比（width / height）
# h = 高度
# 存储与 Kalman filter 相关的均值和协方差。
# 记录轨迹 ID、更新次数、年龄等信息。
# 缓存目标特征向量，用于外观匹配。
class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """
# 属性	类型	含义
# mean	ndarray	当前状态均值向量 [x, y, a, h, vx, vy, …]
# covariance	ndarray	状态协方差矩阵
# track_id	int	唯一轨迹 ID
# hits	int	累计被检测更新次数
# age	int	轨迹从创建到现在经过的帧数
# time_since_update	int	距离上次更新的帧数
# state	TrackState	当前轨迹状态（Tentative / Confirmed / Deleted）
# features	List[ndarray]	轨迹历史特征向量，用于匹配
# _n_init	int	连续检测次数阈值，用于确认轨迹
# _max_age	int	连续未匹配的最大帧数，超过则删除轨迹
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    # # 返回 (top-left-x, top-left-y, width, height)
    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    # 返回 (min-x, min-y, max-x, max-y)
    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    # 状态预测
    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        # 预测下一帧的状态和协方差
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        # 目标轨迹存活的总帧数，每帧加 1
        self.age += 1
        # 自上次更新（匹配到检测）以来的帧数
# 如果 time_since_update 太大，说明目标可能已经丢失
# DeepSORT 会用这个值来判断是否删除跟踪器
        self.time_since_update += 1

    # 状态更新
    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # detection.to_xyah() → 将检测框转换成 [center_x, center_y, aspect_ratio, height]
# kf.update(mean, cov, measurement) → 测量更新：
# 根据测量更新状态向量 mean
# 更新协方差矩阵 covariance
# 作用：把实际观测（检测框）融入预测结果，使轨迹更准确
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        # 更新特征缓存
        # ：存储最近的外观特征，用于后续的目标匹配
        self.features.append(detection.feature)

        self.hits += 1
        # 匹配到检测 → 重置为 0
        self.time_since_update = 0
        # 新轨迹刚初始化时是 Tentative
# 如果连续匹配到足够多的检测（hits >= n_init），转为 Confirmed
# 作用：防止误匹配或噪声目标被误认为有效轨迹
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        # Tentative → 刚创建的轨迹，还没确认
# 如果第一次就没匹配到检测，直接标记为 Deleted
# 作用：防止噪声检测创建无效轨迹
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        # 超过最大未匹配帧数，标记为 Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
