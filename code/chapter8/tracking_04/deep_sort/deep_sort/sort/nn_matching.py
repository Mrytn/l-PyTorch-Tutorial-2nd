# vim: expandtab:ts=4:sw=4
import numpy as np
'''这段代码是 DeepSORT/ReID 距离度量部分的实现，用于在目标跟踪中比较特征（embedding）之间的相似度。'''

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    # a2：形状 (N,)，每个向量的平方和，相当于 ‖a[i]‖²。
# b2：形状 (L,)，每个向量的平方和，相当于 ‖b[j]‖²。
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    # 计算矩阵 a 和 b 中所有点对之间的 欧氏平方距离。
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    # 数值计算时可能会因为浮点误差出现 负数（理论上距离不可能小于 0），这里用 clip 保证结果在 [0, ∞
    r2 = np.clip(r2, 0., float(np.inf))
    return r2

'''计算 余弦距离 (cosine distance) 矩阵'''
def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    # 如果为 True，表示 a 和 b 已经是 单位向量（norm=1），否则会先做归一化。
    if not data_is_normalized:
        # keepdims=True 表示 在做归约运算（比如求和、求范数）后，保留被消掉的维度，但长度为 1
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    # np.dot(a, b.T) → 得到 (N,L) 的余弦相似度矩阵。
# 减去 1. → 得到余弦距离矩阵。
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    # 调用之前的 _pdist 函数，得到 平方欧氏距离矩阵形状 (N, L)。
    distances = _pdist(x, y)
    # 对 每一列（每个查询点 y[j]）取最小值
# 意思是：找出 y[j] 到所有 x[i] 的最小平方距离
# 结果形状 (L,)
# 用 np.maximum 保证最小距离不会出现
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    # 得到 (N, L) 的余弦距离矩阵
    # 然后对 每一列（每个查询点 y[j]）取最小值
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

'''这是核心类，用于存储每个目标的历史特征，并计算新特征与它们的最近邻距离'''
class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """
# metric：使用 "euclidean" 或 "cosine"
# matching_threshold：匹配阈值（大于这个值 → 视为不匹配）
# budget：历史特征存储上限（只保留最近的 budget 个特征）
# samples：字典，映射 target_id → [features...]
    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        # 用新提取的特征更新数据库
        for feature, target in zip(features, targets):
            # self.samples 是一个字典 {target_id: [feature1, feature2, ...]}
# setdefault(target, []) 如果 target 不存在就创建一个空列表
# append(feature) 把新特征加入目标的样本列表
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                # self.budget 表示 每个目标最多保存多少特征样本
# 取 最近的 budget 个特征（保留新特征，舍弃旧特征）
                self.samples[target] = self.samples[target][-self.budget:]
        # 只保留 当前活跃的目标
# 移除不在 active_targets 中的目标样本，防止数据库无限增长
        self.samples = {k: self.samples[k] for k in active_targets}

    '''计算给定特征与目标样本库之间的距离矩阵，常用于 多目标跟踪 (MOT) 或 ReID 匹配'''
    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        # 创建空的代价矩阵
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            # self._metric(self.samples[target], features) → 计算 目标特征与所有输入特征的距离向量
# 结果赋值到第 i 行
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
