# vim: expandtab:ts=4:sw=4
import numpy as np
'''你贴的这个 Detection 类，其实就是在目标检测/跟踪（tracking）里常见的 检测框数据结构，它把一张图里某个检测框的信息（位置、置信度、特征向量）封装成对象，方便后续处理'''

class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """
# 把 tlwh、confidence、feature 转成固定格式（float32 数组或浮点数）。
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        # 把 (x, y, w, h) 转换成 (x1, y1, x2, y2)：
# x2 = x + w
# y2 = y + h
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        # 转换成 (cx, cy, a, h)：
# cx = x + w/2
# cy = y + h/2
# a = w / h
# h = h
# 这个格式在 卡尔曼滤波 (Kalman Filter) 里很常见，用来跟踪对象
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2     # x,y 变成中心点坐标
        ret[2] /= ret[3]           # w/h 变成宽高比
        return ret
