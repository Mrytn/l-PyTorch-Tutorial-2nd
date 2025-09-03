# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""
'''======================1.导入安装好的python库====================='''
import ast
import contextlib
import json
import math
import platform
import warnings # 警告程序员关于语言或库功能的变化的方法
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2  # 调用OpenCV的cv库
import numpy as np
import pandas as pd
import requests # Python的HTTP客户端库
import torch # pytorch深度学习框架
import torch.nn as nn  # 专门为神经网络设计的模块化接口
from PIL import Image # 图像基础操作模块
from torch.cuda import amp # 混合精度训练模块

'''===================2.加载自定义模块============================'''
from utils import TryExcept
# 加载数据集的函数
from utils.dataloaders import exif_transpose, letterbox
# 定义了一些常用的工具函数
from utils.general import (LOGGER, ROOT, Profile, check_requirements, check_suffix, check_version, colorstr,
                           increment_path, is_jupyter, make_divisible, non_max_suppression, scale_boxes, xywh2xyxy,
                           xyxy2xywh, yaml_load)
# 定义了Annotator类，可以在图像上绘制矩形框和标注信息
from utils.plots import Annotator, colors, save_one_box
# 定义了一些与PyTorch有关的工具函数
from utils.torch_utils import copy_attr, smart_inference_mode

'''===========1.autopad：根据输入的卷积核计算该卷积模块所需的pad值================'''
# 根据卷积核大小和膨胀系数自动计算 padding，保证卷积后的输出和输入尺寸一致
# k：卷积核大小，可以是整数（方形卷积）或列表/元组（长宽不同）。
# p：padding，默认 None，表示自动计算。
# d：dilation（膨胀系数），默认 1。卷积膨胀会增加卷积感受野。
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        # 膨胀卷积的有效卷积核尺寸 = d * (k - 1) + 1
        # 普通卷积，k=3, d=1 → 实际核大小 = 3
# 膨胀卷积，k=3, d=2 → 实际核大小 = 2*(3-1)+1=5
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    # 如果用户没有指定 p，函数会自动计算 padding
    if p is None:
        # 这个 autopad 函数本质上是一个 “偷懒的小工具”，它的核心假设就是：
# 步长 s = 1（stride=1），否则公式不成立；
# 输出尺寸允许用整除取整（//）来近似，所以 p = k // 2；
# 卷积核大小 k_eff 必须是奇数，这样才能保证 out = in，如果是偶数，就会出现 out = in 或 out = in - 1（差 1 的情况）
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

'''===========2.Conv：标准卷积 由Conv + BN + activate组成================'''
class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    # default_act = nn.SiLU()：默认激活函数是 SiLU（Sigmoid Linear Unit，也叫 Swish-1），它比 ReLU 平滑，在 YOLOv5 中作为标准激活函数。
    default_act = nn.SiLU()  # default activation
    # c1：输入通道数
# c2：输出通道数
# k：卷积核大小（默认1x1）
# s：stride 步长（默认1）
# p：padding，如果 None 就自动计算（autopad）
# g：groups 分组卷积（默认1，标准卷积；如果 g=c1 就是 depthwise 卷积）
# d：dilation 膨胀系数
# act：激活函数（默认 True → SiLU；False → Identity；或者传入自定义激活函数）
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # bias=False，因为后面会接 BatchNorm，BN 本身带有偏置参数，卷积层就可以去掉偏置，减少冗余
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 批归一化层（BN），作用：加快收敛、提高稳定性，减少过拟合。
# c2 表示输出通道数，每个输出通道都有独立的缩放和偏移参数。
        self.bn = nn.BatchNorm2d(c2)
        # 如果 act=True → 使用默认激活函数 SiLU。
# 如果 act 是 nn.Module（例如 nn.ReLU()），就直接用它。
# 如果 act=False，则使用 nn.Identity()（恒等映射，不做激活）。
# 👉 这样写的好处是灵活，既能默认用 SiLU，也能自定义激活函数，或者干脆不要激活
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # 输入 → 卷积 → BN → 激活函数 → 输出。
# 这是标准的 Conv-BN-Activation 模式，在现代 CNN 里非常常见（比如 ResNet、YOLO）
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # 融合前向传播：
# 输入 → 卷积 → 激活函数 → 输出。
# 为什么没有 BN？因为在模型推理部署时，可以把卷积和 BN 融合成一个等效卷积层（weight 和 bias 融合）。
# 好处：减少计算量，提高推理速度
        return self.act(self.conv(x))

'''===========3.DWConv：深度可分离卷积================'''
class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        # g=math.gcd(c1, c2)，即 输入通道数和输出通道数的最大公约数。
        # 如果 c1 == c2，那么 gcd(c1, c2) = c1 = c2，此时就是 标准 Depthwise 卷积（每个通道独立卷积）。
# 如果 c1 != c2，比如输入 32 通道，输出 64 通道，gcd(32,64)=32 → 这就变成了 group convolution（32组，每组2个输出通道）。
# 这样写更通用，既能支持纯 depthwise，又能支持某些情况的 group convolution。
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

'''===========1.TransformerLayer：================'''
class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        # 三个线性层：分别生成 Query、Key、Value，输入输出维度都是 c。
# bias=False，因为注意力里偏置作用不大
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        # PyTorch 的多头注意力层，embed_dim=c，表示输入输出的特征维度。
# num_heads：多少个注意力头并行
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        # 对应 Transformer 里的 前馈网络 (Feed-Forward Network, FFN)。
# 在标准 Transformer 里，FFN 是 c -> 4c -> c 两层，这里简化成 c -> c -> c。
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        # 先通过 Q、K、V 投影。
# 输入 MultiheadAttention，输出注意力结果。
# ma(...)[0] 取的是注意力结果（第一个返回值），第二个是注意力权重，不用。
# 残差连接：加上原始 x
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        # 前馈网络：先 fc1 再 fc2。
# 再做一次 残差连接
        x = self.fc2(self.fc1(x)) + x
        return x

'''===========2.TransformerBlock：================'''
# TransformerBlock 是在 YOLOv5 里把 Transformer 引入卷积特征图的一种方式，可以看作是 Vision Transformer (ViT) 的轻量化变体
class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        # 如果输入通道 c1 不等于 Transformer 需要的通道 c2，先用一个卷积做通道对齐。
        if c1 != c2:
            self.conv = Conv(c1, c2)
        # 用一个可学习的线性层当作 位置编码
        # 注意这里和 ViT 的 固定正弦位置编码 不同，YOLO 直接让网络学。
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        # 堆叠了 num_layers 层 TransformerLayer。
# 每层就是你刚才看的 多头注意力 + 残差 + FFN。
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        # 保存输出通道数
        self.c2 = c2
    # 前向传播
    def forward(self, x):
        if self.conv is not None:
            #先卷积对齐防止通道数不匹配
            x = self.conv(x)
        b, _, w, h = x.shape
        # x.flatten(2)从 第 2 个维度（下标从 0 开始算） 一直到最后，全部 flatten 成一个维度。
        # .permute(2, 0, 1) → [w*h, b, c2]，满足 nn.MultiheadAttention 的输入格式：[seq_len, batch, embed_dim]
        p = x.flatten(2).permute(2, 0, 1)
        # self.linear(p)：生成位置编码，加到 p 上。
# self.tr(...)：送进多层 Transformer。输出仍是 [w*h, b, c2]。
# .permute(1, 2, 0) → [b, c2, w*h]。
# .reshape(b, self.c2, w, h) → 还原成卷积特征图的形式。
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

'''===========4.Bottleneck：标准的瓶颈层 由1x1conv+3x3conv+残差块组成================'''
class Bottleneck(nn.Module):
    # Standard bottleneck
    # c1：输入通道数
# c2：输出通道数
# shortcut：是否使用残差连接（默认 True）
# g：groups（卷积分组，默认1 → 普通卷积）
# e：通道压缩比例（expansion），默认 0.5 → 先把通道压缩到 50%
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        # c_ 是瓶颈中间通道数（hidden channels）：
# 例如 c2=64, e=0.5 → c_ = 32
# 先用 1×1 卷积把输入通道压缩到 c_，减少计算量
        c_ = int(c2 * e)  # hidden channels
        # 压缩通道，减小计算量
        # 输入通道 = c1，输出通道 = c_
        self.cv1 = Conv(c1, c_, 1, 1)
        # 恢复通道数，并提取特征
        # 输入通道 = c_，输出通道 = c2
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # shortcut=True 且 输入通道等于输出通道 → self.add=True
        self.add = shortcut and c1 == c2

    def forward(self, x):
        # 如果 self.add=True → 加上残差 x
# 如果 self.add=False → 直接输出卷积结果
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

'''===========5.BottleneckCSP：瓶颈层 由几个Bottleneck模块的堆叠+CSP结构组成================'''
# CSP 的核心思想是：将输入拆分为两部分，一部分经过 Bottleneck 堆叠，另一部分直接跳过，最后融合
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # c1：输入通道数
# c2：输出通道数
# n：Bottleneck 堆叠数量
# shortcut：是否使用残差连接
# g：groups（分组卷积）
# e：隐藏通道压缩比例（expansion），默认 0.5
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        # 中间通道数 c_：
# 用于 Bottleneck 堆叠的隐藏通道
# 例如 c2=64, e=0.5 → c_=32
        c_ = int(c2 * e)  # hidden channels
        # 第一条主分支：压缩通道
        # 1×1 卷积，将输入 c1 压缩到 c_
# 用于 Bottleneck 堆叠
        self.cv1 = Conv(c1, c_, 1, 1)
        # 第二条跳过分支：直接从输入到隐藏通道 c_
# 1×1 卷积（不带 BN + 激活），用于保留部分原始特征
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        # 主分支经过 Bottleneck 堆叠后的 过渡卷积
# 1×1 卷积用于进一步处理主分支特征
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        # 最后融合卷积：
# 将主分支和跳过分支拼接（2*c_）
# 再用 1×1 卷积恢复输出通道数 c2
# 带 BN + 激活（因为 Conv 内部有 BN + SiLU）
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        # BN + 激活作用在 拼接后的特征 上
# 拼接前两条分支特征：y1（主分支）、y2（跳过分支）
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        # 堆叠 n 个 Bottleneck
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # 主分支：
# cv1(x) → 压缩通道
# self.m(...) → 堆叠 Bottleneck
# cv3(...) → 过渡卷积
# → 得到 y1
        y1 = self.cv3(self.m(self.cv1(x)))
        # -跳过分支：
# cv2(x) → 保留原始特征
# → 得到 y2
        y2 = self.cv2(x)
        # torch.cat((y1, y2), dim=1) → 拼接通道
# self.bn(...) → 批归一化
# self.act(...) → 激活函数
# self.cv4(...) → 1×1 卷积输出最终通道数
# BottleneckCSP 更注重 残差特征稳定融合
# 拼接的两条分支通道可能 尺度/分布不同
# BN + 激活 可以让融合后的特征 尺度统一 + 非线性增强
# 特别是深层网络，BN + 激活 可以稳定梯度，防止梯度消失或爆炸
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

'''===========6.C3：和BottleneckCSP模块类似，但是少了一个Conv模块================'''
# C3是一种简化版的BottleneckCSP，模块和BottleneckCSP模块类似，但是少了一个Conv模块，只有3个卷积，可以减少参数，所以取名C3。其实结构是一样的，写法略微有差异。
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 主分支 cv1：输入 → 1×1 Conv → 压缩通道 → 进入 Bottleneck 堆叠
        self.cv1 = Conv(c1, c_, 1, 1)
        # 跳过分支 cv2：输入 → 1×1 Conv → 保留原始特征（不经过 Bottleneck）
        self.cv2 = Conv(c1, c_, 1, 1)
        # 拼接主分支和跳过分支 → 2*c_ 通道
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # cv1、cv2 本身是 Conv → 自带 BN + 激活
# 拼接的两条分支特征 已经归一化 + 激活过
# 在 YOLOv5 轻量化设计中，省略拼接前 BN + 激活不会损失太多性能
# 节省了 计算量和内存
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # CSP 的隐藏通道数，和原来的 C3 一样
# 用于 TransformerBlock 的输入通道
        c_ = int(c2 * e)
        # 原 C3 里 self.m 是若干个 Bottleneck 堆叠
# 在 C3TR 中，换成 TransformerBlock：
        self.m = TransformerBlock(c_, c_, 4, n)

# 继承自 C3，n 个 Bottleneck 更换为 1 个 SPP
class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # 加入了 SPP（Spatial Pyramid Pooling），用来增强感受野和多尺度特征表达
        # 将 C3 模块原来的 Bottleneck 堆叠 self.m 替换为 SPP 模块
# 也就是说：
# 主分支不再是 Bottleneck × n
# 而是 SPP → 多尺度池化特征提取
# SPP 可以捕获 不同尺度的上下文信息，增强感受野
        self.m = SPP(c_, c_, k)

# 继承自 C3，Bottleneck 更换为 GhostBottleneck
class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # 将 C3 原来的 Bottleneck 堆叠 替换为 GhostBottleneck × n
# GhostBottleneck 是 GhostNet 的轻量化残差模块：
# 通过 GhostConv 生成部分特征，另一部分通过廉价操作生成剩余特征
# 达到减少卷积计算量的目的
# 这里堆叠 n 个 GhostBottleneck 替代原来的 Bottleneck 堆叠
# 适合 轻量化 YOLOv5 版本（如 YOLOv5n, YOLOv5s）
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

'''===========7.SPP：空间金字塔池化模块================'''
# 用于增强 多尺度特征感受野 的重要模块。
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    # c1：输入通道数
# c2：输出通道数
# k：池化核大小列表（默认 (5,9,13)），表示不同尺度的 MaxPooling
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        # c_ 是隐藏通道数，用于压缩输入特征
        c_ = c1 // 2  # hidden channels
        # 1×1 卷积压缩通道 → c_
# 作用：降低 SPP 后续池化操作的计算量
        self.cv1 = Conv(c1, c_, 1, 1)
        # 输入通道 = 压缩后通道 * (池化核数量 + 1)
# +1 是为了保留原始特征（x 本身）
# 输出通道 = c2
# 1×1 卷积 → 融合多尺度特征
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        # 每个池化：
# kernel_size=x → 不同感受野
# stride=1 → 输出尺寸不变
# padding=x//2 → 保持输出特征图大小和输入一致
# 奇数x://2 = (x-1)/2，偶数x//2 = x/2
# 奇数2*pad/2-x=-1，偶数2*pad/2-x=0
# 奇数因为pad/2向下取整能约掉后面的+1，偶数不行，偶数不能保证输入输出相等，所以输入x必须是奇数
# 这样就能同时捕获 不同尺度的局部特征
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        # 在上下文内捕获或修改警告行为，并在退出上下文后自动恢复原来的警告设置
        with warnings.catch_warnings():
            # simplefilter 用来设置警告的处理规则
# 'ignore' 表示 忽略警告，不打印、不抛出
# 所以在 with 块中，所有警告都会被忽略。
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

# 快速版的空间金字塔池化
# 实现类似 SPP 的多尺度感受野，但计算更少、更快
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        # 输入通道 = 4 × c_
# 因为 SPPF 会生成 4 个特征图（原始 + 3 次池化）
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # 通过 多次堆叠池化 来模拟 SPP 多尺度效果
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            # cv1(x) → 压缩通道
# y1 = m(x) → 第一次池化，增加感受野
# y2 = m(y1) → 第二次池化，感受野更大
# m(y2) → 第三次池化
# 拼接 [x, y1, y2, m(y2)] → 模拟多尺度 SPP
# cv2(...) → 1×1 Conv 融合通道 → 输出 c2
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


'''===========8.Focus：把宽度w和高度h的信息整合到c空间================'''
# 将输入特征图的 宽高信息（w×h） “聚合”到 **通道维度（c）”
# 相当于 空间压缩 → 通道扩展，减少特征图大小，同时保留局部信息
class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        # 下面会将 输入通道 c1 扩展到 4*c1
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # x → (batch, c, w, h)
        # x[..., ::2, ::2] → 取 偶数行 + 偶数列
# x[..., 1::2, ::2] → 奇数行 + 偶数列
# x[..., ::2, 1::2] → 偶数行 + 奇数列
# x[..., 1::2, 1::2] → 奇数行 + 奇数列
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))

'''===========1.GhostConv：幻象卷积  轻量化网络卷积模块================'''
class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


'''===========2.GhostBottleneck：幻象瓶颈层 ================'''
class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        # 中间隐藏通道数，GhostBottleneck 设计上比原始通道少一半，减少计算量。
        c_ = c2 // 2
        self.conv = nn.Sequential(
            # 做通道压缩 (pointwise)
            GhostConv(c1, c_, 1, 1),  # pw
            # 假设 DWConv 是 Depthwise Convolution（深度可分卷积）
# 参数解释：
# c_：输入通道
# c_：输出通道
# k：卷积核大小
# s：stride
# act=False：是否加激活函数（这里不加）
# 功能：对输入做卷积下采样（stride = 2 时尺寸减半）
# nn.Identity()
# 什么都不做的层
# 直接返回输入（原封不动）
# 相当于占位，方便写成统一的网络结构，不用额外判断
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            # 再做通道扩张回输出通道数 (pw-linear)
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

'''===========9.Contract：收缩模块：调整张量的大小，将宽高收缩到通道中。================'''
# 收缩模块：调整张量的大小，将宽高收缩到通道中、
# 把 空间维度 (w,h) 压缩进 通道维度
# 和 Focus 是一类操作（空间换通道）
# 类似于 PixelUnshuffle
class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        # h,w 各缩小 gain 倍
# 通道数扩张 gain² 倍
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        # 要求 h % s == 0，w % s == 0（能被整除）
        s = self.gain
        # 把空间拆成小块：
# h // s 和 w // s → 压缩后的尺寸
# 多出来的 s, s → 存储在新维度中
#  子: (1,64,80,80) → (1,64,40,2,40,2)
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        # 调整维度顺序
# 把 s, s 提到通道位置前
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        # 最终合并 s*s 到通道维度
        # 如果张量经过了 permute、transpose 等操作，内存顺序已经改变，不再是连续的，这时 view 可能报错或者得到错误结果
        # 所以代码里常见：x = x.permute(...).contiguous().view(...)
        # 仅仅是改变张量的维度顺序（类似 numpy 的 transpose），但不会真正拷贝数据。
# 这一步往往是必须的，因为你想把 (b, c, h, w) 变成 (b, c*s*s, h//s, w//s) 时，元素顺序需要重新组织，不只是 reshape
# 如果 x 的内存布局刚好符合目标形状（比如 PixelShuffle 里有特殊保证），那可以直接 view。
# 但一般情况下，不 permute 直接 view，得到的结果会错乱（channel 和空间位置混了）。
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)

'''===========10.Expand：扩张模块，将特征图像素变大================'''
class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        # 拆分通道,增加两个维度
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        # 调整维度顺序
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)

'''===========11.Concat：自定义concat模块，dimension就是维度值，说明沿着哪一个维度进行拼接================'''
# 拼接函数，将两个tensor进行拼接
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        # 默认在通道维度 (dim=1) 上拼接
        self.d = dimension

    def forward(self, x):
        # 把传入的 tensor 列表 x 沿着 self.d 维拼接
        return torch.cat(x, self.d)


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != 'cpu'  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {'config.txt': ''}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files['config.txt']:  # load metadata dict
                d = json.loads(extra_files['config.txt'],
                               object_hook=lambda d: {int(k) if k.isdigit() else k: v
                                                      for k, v in d.items()})
                stride, names = int(d['stride']), d['names']
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
            check_requirements('opencv-python>=4.5.4')
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if 'stride' in meta:
                stride, names = int(meta['stride']), eval(meta['names'])
        elif xml:  # OpenVINO
            LOGGER.info(f'Loading {w} for OpenVINO inference...')
            check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch
            ie = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
            network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
            if network.get_parameters()[0].get_layout().empty:
                network.get_parameters()[0].set_layout(Layout('NCHW'))
            batch_dim = get_batch(network)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            executable_network = ie.compile_model(network, device_name='CPU')  # device_name="MYRIAD" for Intel NCS2
            stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == 'cpu':
                device = torch.device('cuda:0')
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings['images'].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f'Loading {w} for CoreML inference...')
            import coremltools as ct
            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
            import tensorflow as tf
            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=''), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, 'rb') as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs='x:0', outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf
                Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                delegate = {
                    'Linux': 'libedgetpu.so.1',
                    'Darwin': 'libedgetpu.1.dylib',
                    'Windows': 'edgetpu.dll'}[platform.system()]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, 'r') as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode('utf-8'))
                    stride, names = int(meta['stride']), meta['names']
        elif tfjs:  # TF.js
            raise NotImplementedError('ERROR: YOLOv5 TF.js inference is not supported')
        elif paddle:  # PaddlePaddle
            LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
            check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
            import paddle.inference as pdi
            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix('.pdiparams')
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f'Using {w} as Triton Inference Server...')
            check_requirements('tritonclient[all]')
            from utils.triton import TritonRemoteModel
            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith('tensorflow')
        else:
            raise NotImplementedError(f'ERROR: {w} is not a supported format')

        # class names
        if 'names' not in locals():
            names = yaml_load(data)['names'] if data else {i: f'class{i}' for i in range(999)}
        if names[0] == 'n01440764' and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / 'data/ImageNet.yaml')['names']  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings['images'].shape:
                i = self.model.get_binding_index('images')
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings['images'].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype('uint8'))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({'image': im})  # coordinates are xywh normalized
            if 'confidence' in y:
                box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input['quantization']
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input['index'], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output['index'])
                    if int8:
                        scale, zero_point = output['quantization']
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != 'cpu' or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p='path/to/model.pt'):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url
        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ['http', 'grpc']), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path('path/to/meta.yaml')):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d['stride'], d['names']  # assign stride, names
        return None, None

'''===========2.AutoShape：自动调整shape,该类基本未用================'''
# AutoShape是一个模型扩展模块，给模型封装成包含前处理、推理、后处理的模块(预处理 + 推理 + nms)
class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        super().__init__()
        if verbose:
            LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        # Inference from various sources. For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f'image{i}'  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
                files.append(Path(f).with_suffix('.jpg').name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(y if self.dmb else y[0],
                                        self.conf,
                                        self.iou,
                                        self.classes,
                                        self.agnostic,
                                        self.multi_label,
                                        max_det=self.max_det)  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


'''===========3.Detections：对推理结果进行处理================'''
class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1E3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path('')):
        s, crops = '', []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f'\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(', ')
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({
                                'box': box,
                                'conf': conf,
                                'cls': cls,
                                'label': label,
                                'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label if labels else '', color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display
                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip('\n')
            return f'{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}' % self.t
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    @TryExcept('Showing images is not supported in this environment')
    def show(self, labels=True):
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp', exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.ims[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['ims', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def print(self):
        LOGGER.info(self.__str__())

    def __len__(self):  # override len(results)
        return self.n

    def __str__(self):  # override print(results)
        return self._run(pprint=True)  # print results

    def __repr__(self):
        return f'YOLOv5 {self.__class__} instance\n' + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


'''===========4.Classify：二级分类模块================'''
# 把卷积特征图 [b, c1, H, W] 转成最终的类别预测 [b, c2]
class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 dropout_p=0.0):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        # 这是中间通道数，用于分类特征的压缩/扩展，YOLOv5 的作者沿用 EfficientNet-B0 的通道数作为经验值。
        c_ = 1280  # efficientnet_b0 size
        # 1×1 或指定核大小的卷积，把输入通道 c1 转成 c_
# 用于特征整合和降/升维。
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        # 全局平均池化，把 H×W 压成 1×1
# 结果 [b, c_, 1, 1]，每个通道得到一个全局特征
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        # Dropout 防止过拟合
# p 是丢弃概率
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        # 全连接层，把特征 [b, c_] 映射到类别数 c2
# 输出就是分类概率的 logits
# logits 是神经网络最后一层的 未归一化输出，也就是线性层输出的实数值。
# 它 还不是概率，只是每个类别的“得分
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        # 支持输入是多个特征图列表的情况（可能从不同层拼接）
# 沿通道维度合并
        if isinstance(x, list):
            x = torch.cat(x, 1)
        # self.pool(self.conv(x)).flatten(1)卷积 conv(x) → [b, c_, H, W]
# 自适应平均池化 pool(...) → [b, c_, 1, 1]
# flatten(1) → [b, c_]，为全连接层做准备
# self.linear(self.drop(...))
# 先 Dropout
# 再线性映射到 c2 个类别
# 输出 [b, c2]
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
