# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""
'''======================1.导入安装好的python库====================='''
import contextlib
import argparse
from copy import deepcopy # 数据拷贝模块 深拷贝
from pathlib import Path
import platform
import sys
import os


'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve()
# parents[1] 表示 yolo.py 文件的 上上级目录（因为 YOLOv5 项目一般是 yolov5/models/yolo.py，往上两层就到了 yolov5/ 项目根目录）
# 要用项目里的其他模块代码，都必须要先把项目根目录加到sys.path,且在其他模块导入前加
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
'''===================3..加载自定义模块============================'''
# yolov5的网络结构(yolov5)
from models.common import *
# 导入在线下载模块
from models.experimental import *
# 导入检查anchors合法性的函数
from utils.autoanchor import check_anchor_order
# 定义了一些常用的工具函数
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
# 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.plots import feature_visualization
# 定义了一些与PyTorch有关的工具函数
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)

try:
    # thop 是一个用于 计算神经网络 FLOPs（浮点运算次数）和参数量 的库
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    '''===================1.获取预测得到的参数============================'''
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        # 数据集类别数量
        self.nc = nc  # number of classes
        # 表示每个anchor的输出数，前nc个01字符对应类别，后5个对应：是否有目标，目标框的中心，目标框的宽高
        self.no = nc + 5  # number of outputs per anchor
        # 表示预测层数，yolov5是3层预测
        self.nl = len(anchors)  # number of detection layers
        # 表示anchors的数量，除以2是因为[10,13, 16,30, 33,23]这个长度是6，对应3个ancho
        self.na = len(anchors[0]) // 2  # number of anchors
        # 每个检测层都需要一个 grid（网格坐标） 来辅助解码预测框（从特征图坐标还原到输入图像坐标）。
# self.nl 表示有多少个检测层，就初始化多少个空的 grid。
# 后面在 forward 推理时，会根据当前特征图大小生成对应的 grid（如果和上一次的大小不一致），这样每个检测层就有自己的一套网格
# 存放 feature map 的网格坐标
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        # 存放 anchor 尺寸（在 forward 时 reshape）
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        # 把 anchors 存为 buffer，不会被优化器更新（但会随模型保存/加载）。
# 形状 (nl, na, 2)：
# nl = 层数
# na = 每层 anchor 数
# 2 = (w,h)
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # 这里的 1 是卷积核的大小，也就是 kernel_size=1。只做通道数映射，不改变空间尺寸
        # self.m：每个检测层的卷积输出层。
# 输入通道：x（由 ch 列表给定，不同 feature map 通道数可能不同）。
# 输出通道：no * na，即每个像素点预测的 (nc+5) * anchors。
# kernel size = 1 → pointwise 卷积，相当于预测头。
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # 是否在 forward 中使用 inplace 操作，节省内存并提升速度。
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    '''===================2.向前传播============================'''
    # 把 feature map 转换成最终的预测框
    # x是一个列表，存放各个尺度的特征图 [x_P3, x_P4, x_P5]
    def forward(self, x):
        # z 用来收集每一层（P3, P4, P5）的预测结果（只在推理时用）
        z = []  # inference output
        # 遍历每个检测层
        for i in range(self.nl):
            # self.m 模块列表，存放每个尺度的输出卷积 Conv2d，输出通道 = no * na
            # self.m[i](x[i])对第 i 个尺度的特征图应用对应卷积层，得到预测通道
            # self.m 里就存了 三个卷积层P3 P4 P5
            x[i] = self.m[i](x[i])  # conv
            # x[i].shape = (batch_size, channels, height, width)
            # 通道数等于na*n0
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            # 把通道维拆分成 anchor 数 和 每个 anchor 的预测通道数。(batch_size, na, no, ny, nx)
            #batch_size = 批大小
# na = anchor 数
# no = 每个 anchor 输出数（类别+坐标+置信度）
# ny, nx = 特征图高宽
# permute(0, 1, 3, 4, 2)把维度顺序改成 方便后续解码
# (batch_size, anchor, y, x, no)原来是 (batch, anchor, no, ny, nx)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            '''
            YOLOv5 Detect/Segment 推理阶段的核心解码逻辑，主要是把卷积输出转换成 真实坐标的预测框
            '''
            if not self.training:  # inference
                # self.dynamic 表示是否采用 动态推理模式。
# 如果为 True，则每次推理都会重新生成网格（适合输入图像分辨率经常变化的情况）。
# 如果为 False，则只在检测到特征图尺寸改变时才重新生成一次，节省计算
# self.grid[i]：保存的网格偏移（grid offsets），形状一般是 (1, na, ny, nx, 2)。
# x[i]：来自第 i 个检测层的输出特征图，形状是 (bs, na*no, ny, nx)。
# x[i].shape[2:4]：取特征图的 (ny, nx)（高度和宽度）。
# self.grid[i].shape[2:4]：取之前缓存的网格的 (ny, nx)。
# 两者不一致时说明当前输入图像尺寸和之前的不一样，需要重新生成网格。
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # grid[i]：每个网格点的偏移坐标，形状 (1, na, ny, nx, 2)。
# anchor_grid[i]：当前层锚框缩放后的尺寸，形状 (1, na, ny, nx, 2)。
# _make_grid 生成新的网格坐标和锚框网格
# 网络输出 (tx, ty, tw, th, confidence, class_prob)，其中 (tx, ty) 是相对网格点左上角的偏移。
# 所以必须先生成整张特征图的网格坐标，用它和网络输出做解码，得到真实图像坐标。
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    # x[i] 是第 i 个检测层的卷积输出，shape 一般是(bs, na, ny, nx, no)
                    #bs = batch size
# na = anchor 数量（通常是 3）
# ny, nx = feature map 的高度、宽度
# no = 每个 anchor 的输出维度
# no = 4 + 1 + nc + nm
# 4 → box 坐标 (x, y, w, h)
# 1 → objectness
# nc → 分类数
# nm → mask 原型系数（segmentation head 特有）
                    # 把检测头的输出 x[i] 按照语义拆分成不同的部分
                    # 在第 4 维（dim=4）切成 4 块
                    # 前 2 维：xy → 预测框的中心坐标 (tx, ty)
# 后 2 维：wh → 预测框的宽高 (tw, th)
# self.nc + 1 维：conf → 目标置信度（1）+ 类别概率（nc）
# 剩下的 self.no - self.nc - 5 维：mask → 分割掩码系数
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    # 输入的 xy 是网络预测的偏移量 (tx, ty)，范围是实数。
# sigmoid() → 把它压缩到 (0,1)。
# * 2 → 把范围扩展到 (0,2)，这样预测点不仅可以落在网格单元内，还能稍微超出。
# + self.grid[i] → 加上当前网格的左上角坐标，得到相对于特征图的绝对位置。
# * self.stride[i] → 映射回原图坐标系（比如 stride=8，则每个 cell 对应原图 8×8 像素）。
# 👉 结果：xy 就是预测框中心点在 原图坐标里的位置。
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    # 输入的 wh 是网络预测的宽高偏移 (tw, th)。
# sigmoid() → 把值限制在 (0,1)。
# * 2 → 范围 (0,2)，使得宽高可以比 anchor 小或大。
# ** 2 → 再平方，扩大动态范围，使得预测框大小更灵活（既能预测小目标，也能预测大目标）。
# * self.anchor_grid[i] → 与对应层的 anchors 相乘，得到最终的宽高。
# 👉 结果：wh 就是预测框的宽和高（原图尺度）。
# **2 和前面的 *2 虽然写法不一样，但本质都是 为了扩大预测框的范围，让模型可以更灵活地预测框大小。
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    # 拼接所有预测结果，以最后一个维度拼接
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)Detect 模块（只检测 box）
                    # 逻辑与 Segment 类似，只是没有 mask：
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                    # 把每层 feature map 展平：
# bs：batch size
# na * nx * ny：每层所有 anchor 的格子数
# no：每个 anchor 的输出维度 (nc+5) 或 (nc+5+mask)
# 便于不同尺度的预测结果拼接。
                z.append(y.view(bs, self.na * nx * ny, self.no))
        # 训练模式
        # 返回原始 feature map 的卷积输出（未解码）。
# 因为训练时 loss 函数会自己处理 raw 输出，不需要解码成坐标。
# 导出模式
# torch.cat(z, 1)：把所有尺度预测结果拼接：
# z 每个元素形状 (bs, na*ny*nx, no)
# 拼接后 (bs, sum(na*ny*nx), no) 拼接是增加改维度多个值，而不是求和得到一个值
# 注意返回的是一个 tuple (tensor,)，符合 ONNX/TensorRT 导出要求。
# 普通推理模式
# 返回 拼接后的预测 + 原始 feature map list：
# 拼接后的预测 (bs, all_anchors, no)，可以直接进行 NMS 得到最终检测框。
# 原始 feature map x 可以用于其他后处理（比如可视化、Segmentation 分支等）。
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    '''===================3.相对坐标转换到grid绝对坐标系============================'''
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None

# d: 就是读取的 yolov5s.yaml 这种配置文件的 dict（包含 anchors, nc, depth_multiple, width_multiple, backbone/head 的结构定义）。
# ch是一个用来保存之前所有的模块输出的channle。
def parse_model(d, ch):  # model_dict, input_channels(3)
    '''===================1. 获取对应参数============================'''
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # anchors：锚框定义（每层有多少个 anchor）。
# nc：类别数（比如 COCO 是 80）。
# gd：depth_multiple，深度系数，决定每个模块重复的次数。
# gw：width_multiple，宽度系数，决定卷积层通道数的缩放。
# act：激活函数，可选（比如 SiLU）。
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        # 如果 yaml 里定义了激活函数，就修改 Conv 的默认激活函数。
        # eval() 是 Python 的内置函数，它会把字符串当作 Python 表达式执行。
# act 很可能是一个字符串，比如 'nn.SiLU()' 或 'nn.ReLU()'。
# eval(act) 会返回字符串表示的 Python 对象
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    # na：每层的 anchor 数量（如果 anchors 是 list，就取第一个层的长度/2，因为每个 anchor 由宽高2个值组成
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    # no：每个预测层的输出维度 = anchors × (类别数 + 5)。
# 其中 5 = (x, y, w, h, obj_conf)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    '''===================2. 搭建网络前准备============================'''
    # 遍历 backbone + head，构建每一层模块，并处理参数
    # layers：保存每一层构建好的 PyTorch 模块。
# save：保存需要在 forward 中保留输出的层索引（如 skip connection）。
# c2：当前层输出通道数，初始化为输入通道 ch[-1]（通常是 3，RGB）。
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 遍历 backbone 和 head 定义
    # f：from —— 输入来自哪些层（索引）。
# n：number —— 模块重复次数。
# m：module —— 模块类型（如 Conv, Bottleneck, C3 等）。
# args：模块参数列表（如通道数、kernel_size、stride 等）。
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 如果模块类型 m 是字符串，就用 eval 转成实际 Python 类对象。
# 例："Conv" → <class 'models.common.Conv'>。
# 如果已经是类对象，就直接使用。
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # 遍历模块参数 args
        for j, a in enumerate(args):
            # 如果参数是字符串且可 eval 成 Python 对象，就 eval。
# 例："3*ch[0]" → 9（假设 ch[0]=3）。
# 用 contextlib.suppress(NameError) 忽略 eval 时可能产生的 NameError。
# 有些字符串可能依赖于外部变量（如 ch），如果变量不存在就跳过，不报错。
# 最终 args 中每个参数都被转换为实际数值或对象，准备传给模块构造函数。
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
        '''===================3. 更新当前层的参数，计算c2============================'''
        # n：模块重复次数（深度）。
# gd 是 depth_multiple，用来缩放模型深度（例如 YOLOv5s → YOLOv5m 会加深）。
# round(n * gd)：根据深度倍率调整重复次数。
# max(..., 1)：保证重复至少为 1 层。
# n_：保存原始重复次数，用于打印或日志。
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        # 判断当前模块是否属于 卷积/瓶颈类模块（可以缩放通道的层）。
# YOLOv5 中大多数 backbone/head 模块都是这些类型。
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            # 输入通道数，来自 ch[f]。
# f 可以是单个索引或列表（skip connection）。
# c2：输出通道数，一般是 args[0]（即 yaml 中定义的输出通道）
            c1, c2 = ch[f], args[0]
            # c2 != no：排除最后 Detect 层，因为它的输出通道数已经固定（no = na * (nc + 5)）
            if c2 != no:  # if not output
                # make_divisible(..., 8)：保证通道数是 8 的倍数（硬件优化要求，如 GPU/CPU SIMD 对齐）。
# 例如，原本通道 32，gw=1.25 → 32*1.25=40 → make_divisible(40,8) = 40。
                c2 = make_divisible(c2 * gw, 8)

            '''===================4.使用当前层的参数搭建当前层============================'''
            # c1：输入通道
# c2：输出通道
# *args[1:]：保留原本剩余参数（kernel、stride 等）
            args = [c1, c2, *args[1:]]
            # 对 CSP/Bottleneck 类模块：
# 需要指定重复次数 n，插入到参数列表第 3 个位置（索引 2）。
# 然后将 n=1，因为后续会直接用 nn.Sequential(*[m(*args) for _ in range(n)]) 来重复模块
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        # 对于批量归一化只需要输入通道数 ch[f]，其他参数都用默认值
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        # Concat（拼接模块）：
# 输出通道 = 输入各层通道之和。
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        # Detect/Segment 层（预测层）
        elif m in {Detect, Segment}:
            # 添加每层输入通道列表 [ch[x] for x in f]
            args.append([ch[x] for x in f])
            # args[1] 就是 Detect/Segment 模块在构造函数里的第二个参数——锚点信息，所以通常用来表示锚点数量。
            # 如配置文件中   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
            if isinstance(args[1], int):  # number of anchors
                # 如果是整数，比如 3，说明每个输出通道有 3 个 anchor。
# [list(range(args[1] * 2))] * len(f)：
# args[1]*2 是因为每个 anchor 有 (x, y) 两个坐标，所以乘 2。
# list(range(...)) 生成索引列表 [0, 1, 2, 3, 4, 5]。
# * len(f) 表示为每个输入索引复制一份。
# 目的是把整数形式的锚点数转换成列表形式，方便 Detect/Segment 模块使用
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                # 如果模块是 Segment（分割头），对 args[3] 做调整。
# args[3] 通常是输出通道数或者某个卷积层的通道数。
# gw 是宽度增益（width multiplier），用于调整模型大小。
# make_divisible(..., 8) 会将通道数向上凑整为 8 的倍数，保证卷积核对齐硬件（尤其是 GPU）效率。
                args[3] = make_divisible(args[3] * gw, 8)
        # Contract 是“收缩”模块，将空间分辨率下采样，分辨率减小，但通道数增加。
        # 例如输入通道 c=64，args[0]=2（缩小 2 倍），则 c2 = 64 * 2^2 = 256
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        # Expand 是“扩展”模块，将空间分辨率上采样，分辨率增加，但通道数减少
        elif m is Expand:
            # 例如输入通道 c=256，args[0]=2（放大 2 倍），则 c2 = 256 // 4 = 64
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        '''===================5.打印和保存layers信息============================'''
        # parse_model 的最后关键步骤，完成了 模块实例化、属性绑定、日志输出、forward 保存层索引以及输出通道更新
        # *args传给模块的参数，比如 [512, 1, 1] → 输出通道=512, 卷积核=1, 步幅=1。
        # m(*args)实例化一个模块对象，比如 Conv(512, 1, 1)。
        # for _ in range(n)生成 n 个相同的模块对象。
        # *(...)把生成的模块对象解包成位置参数。它要的是多个独立参数，而不是一个生成器或列表。*将列表解包成一个个参数
        # nn.Sequential(...)把这 n 个模块顺序堆叠，作为一个整体模块。
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 获取模块类型名称，用于打印日志
        # str(m) 会返回类似 <class 'models.common.Conv'>，这里去掉前后的 <class '...'>，只保留 Conv
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 计算该模块的参数总数（weights + bias）。
# x.numel() 返回每个 tensor 的元素数量，累加得到总参数量。
        np = sum(x.numel() for x in m_.parameters())  # number params
        # 动态绑定属性到模块对象：
# i：层索引
# f：该层输入来自哪些层
# type：模块类型名称
# np：参数总数
# 方便后续 debug 或 forward trace
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        # 打印日志，显示该层信息：
# i：层索引
# f：from
# n_：重复次数
# np：参数数量
# t：模块类型
# args：模块参数列表
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        # if x != -1
# 跳过 -1，因为 -1 表示“上一层输出”，它必然已经在计算中，不需要专门保存。
# x % i 其实是为了把负索引（倒数第几层）转成正索引
# 如x = -2   →  -2 % 7 = 5   # 倒数第2层就是正向索引5
# x = 6    →   6 % 7 = 6   # 正常索引保持不变
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将实例化的模块加入 layers 列表，最终构建网络。
        layers.append(m_)
        # 更新通道列表 ch：
# ch[i] 表示第 i 层输出通道数，用于后续层计算。
# 第 0 层特殊处理，先清空列表
        if i == 0:
            ch = []
        ch.append(c2)
    # nn.Sequential(*layers)：完整的 YOLOv5 网络。
# sorted(save)：forward 时需要缓存的层索引列表
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
