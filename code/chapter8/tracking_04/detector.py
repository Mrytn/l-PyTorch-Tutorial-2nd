import torch
import numpy as np

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

'''加载模型并在单张图像上检测特定类别对象'''
class Detector:
    # 初始化模型
    def __init__(self, path_yolov5_ckpt):
        self.img_size = 1280
        self.threshold = 0.3
        self.stride = 1

        self.weights = path_yolov5_ckpt

        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights)
        model.to(self.device).eval()
        model.half()

        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names
    # 预处理函数
    def preprocess(self, img):

        img0 = img.copy()
        # 按 img_size 缩放图像，同时保持纵横比并填充边界。
        img = letterbox(img, new_shape=self.img_size)[0]
        # ::-1 表示“步长为 -1”，意思是倒序取元素
        # ::-1 将通道反转 BGR → RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        # 保证图像数据在内存中是连续存储的
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        # 转换为 半精度浮点（float16）
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img
    # 检测函数
    def detect(self, im):

        im0, img = self.preprocess(im)
        # 模型输出预测
        # 每一行预测包含：
        # 每个候选框的坐标 + 置信度 + 各类别概率。
        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        # 对预测框进行 NMS，过滤重叠框
        pred = non_max_suppression(pred, self.threshold, 0.4)

        boxes = []
        # 遍历每张图片的检测结果
        for det in pred:

            if det is not None and len(det):
                # 把预测框坐标从缩放图像映射回原始图像尺寸
                # 因为 NMS 可能返回多个检测框，所以 det 可以有多行，每行是一个框。
                det[:, :4] = scale_boxes(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # for 会遍历 det 的每一行，所以这里的 *x, conf, cls_id 不是针对整个 det，而是针对每一行的。
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    # if lbl not in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']:
                    # 过滤掉非目标类别
                    if lbl not in ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                                   'tricycle', 'awning-tricycle', 'bus', 'motor']:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    # 返回列表 boxes，每个元素 (x1, y1, x2, y2, 类别, 置信度)
                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return boxes
