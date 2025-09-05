# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

'''用于目标跟踪的 Kalman Filter 实现，核心用于在检测框 (bounding box) 上做预测、更新和 gating'''
"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
# 在 Kalman Filter 的 gating 里，自由度就是你用于计算 Mahalanobis 距离的维度数：
# 如果只用位置 (x, y) → dof = 2 → 阈值 chi2inv95[2] = 5.9915
# 如果用位置 + 形状 (x, y, a, h) → dof = 4 → 阈值 chi2inv95[4] = 9.4877
# 所以虽然表里有 1–9 的值，但实际使用时只取你关注的维度对应的自由度（2 或 4）。
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """
    # 始化一个 Kalman Filter（卡尔曼滤波器）
    def __init__(self):
        # ndim = 4：状态空间的维度，通常表示 位置和尺寸，例如 (x, y, w, h)
# dt = 1.：时间间隔，假设每帧之间的时间为 1 单位
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        # _motion_mat 是8×8的状态转移矩阵
        # _motion_mat：状态转移矩阵 (8x8)，前四维加上速度预测下一步状态
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            # 初始化为单位矩阵后，通过循环在特定位置设置dt值
            # 物体下一时刻的位置 = 当前的位置 + 当前速度 × 时间间隔
            self._motion_mat[i, ndim + i] = dt
        # _update_mat：观测矩阵 (4x8)，观测状态只包含 [cx, cy, a, h]
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        # 这两个参数用于计算过程噪声的标准差。位置的噪声权重相对较大，说明位置的不确定性比速度大。速度的噪声权重很小，表明假设速度变化相对平稳。
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    # 输入是一个4维向量，表示检测框的信息：x和y是中心坐标，a是宽高比（aspect ratio），h是高度
    # 格式 (x, y, a, h)
    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        # 将观测到的位置信息作为初始位置，速度初始化为0（因为刚开始无法知道目标的运动速度）。np.r_用于垂直拼接数组，最终得到8维状态向量 [x, y, a, h, vx, vy, va, vh]。
        mean_vel = np.zeros_like(mean_pos)
        # 这是卡尔曼滤波的 初始状态。
        mean = np.r_[mean_pos, mean_vel]
        # 计算初始协方差矩阵
        std = [
            2 * self._std_weight_position * measurement[3],    # x的标准差
    2 * self._std_weight_position * measurement[3],    # y的标准差
    1e-2,                                              # a的标准差
    2 * self._std_weight_position * measurement[3],    # h的标准差
    10 * self._std_weight_velocity * measurement[3],   # vx的标准差
    10 * self._std_weight_velocity * measurement[3],   # vy的标准差
    1e-5,                                              # va的标准差
    10 * self._std_weight_velocity * measurement[3]]   # vh的标准差
        # 把 std 平方，得到方差
# 放到对角阵上 → 8×8 协方差矩阵
# 表示各维度独立，互不相关（非对角元=0）
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
