# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""
"""
Train a YOLOv5 model on a custom dataset
在数据集上训练 yolo v5 模型
Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
    训练数据为coco128 coco128数据集中有128张图片 80个类别，是规模较小的数据集
"""
'''======================1.导入安装好的python库====================='''
from torch.optim import lr_scheduler
import os
from tqdm import tqdm
import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.plots import plot_evolve
from utils.metrics import fitness
import time
import sys
from copy import deepcopy   # 深度拷贝模块
import random
import subprocess
import torch.distributed as dist    # 分布式训练模块
import torch
import numpy as np
from datetime import datetime
from pathlib import Path    # Path将str转换为Path对象 使字符串路径易于操作的模块
import yaml
import torch.nn as nn
import argparse
import math

'''===================2.获取当前文件的绝对路径========================'''
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# 这个是测试集
'''===================3..加载自定义模块============================'''
# 实验性质的代码，包括MixConv2d、跨层权重Sum等
# yolo的特定模块，包括BaseModel，DetectionModel，ClassificationModel，parse_model等
# 定义了自动生成锚框的方法
# 定义了自动生成批量大小的方法
# 定义了回调函数，主要为logger服务
# dateset和dateloader定义代码
# 谷歌云盘内容下载
# 定义了一些常用的工具函数，比如检查文件是否存在、检查图像大小是否符合要求、打印命令行参数等等
# 日志打印
# 存放各种损失函数
# 模型验证指标，包括ap，混淆矩阵等
# 定义了Annotator类，可以在图像上绘制矩形框和标注信息
# 定义了一些与PyTorch有关的工具函数，比如选择设备、同步时间等

'''================4.分布式训练初始化==========================='''
# LOCAL_RANK：用于指定 当前进程在当前机器上的 GPU 编号
# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
# 获取全局进程编号（rank），通常是跨多机训练时每个进程的唯一编号
# RANK 是一个进程在所有进程中的编号，比如你有 4 台机器，每台跑 2 个进程，总共 8 个进程，那么 RANK 从 0 到 7。
RANK = int(os.getenv('RANK', -1))
# WORLD_SIZE 表示所有机器总共运行了多少个训练进程
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
# 这个函数通常是自定义的，用来读取当前代码仓库的 Git 信息，如：
# 当前 Git commit ID
# 当前是否是 clean 状态
# 当前分支名等
# 用于在日志中记录版本，方便追踪模型训练时对应的代码版本。
GIT_INFO = check_git_info()

''' =====================1.载入参数和初始化配置信息==========================  '''
'''
        1.1 载入参数
'''
# hyp超参数 可以是超参数配置文件的路径或超参数字典 path/to/hyp.yaml or hyp
#   opt main中opt参数
# hyp：  超参数，不使用超参数进化的前提下也可以从opt中获取
# opt：  全部的命令行参数
# device：  指的是装载程序的设备
# callbacks：  指的是训练过程中产生的一些参数


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    # 从opt获取参数。日志保存路径，轮次、批次、权重、进程序号(主要用于分布式训练)等
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    # 回调函数触发
    # callbacks 是 YOLOv5 中的 回调机制 对象，用来在训练过程的不同阶段触发自定义事件。
# 它类似于 PyTorch Lightning 或 Keras 的 Hook 系统。
# callbacks 是一个 Callbacks 类 实例，内部维护了一个字典：
# self._callbacks = {
#     'on_pretrain_routine_start': [],
#     'on_pretrain_routine_end': [],
#     'on_train_start': [],
#     'on_train_epoch_start': [],
#     'on_train_epoch_end': [],
#     'on_fit_epoch_end': [],
#     'on_model_save': [],
#     'on_train_end': [],
#     ...
# }
    callbacks.run('on_pretrain_routine_start')

    '''
    1.2 创建训练权重目录，设置模型、txt等保存的路径
    '''
    # Directories
    # Directories 获取记录训练日志的保存路径
    # 设置保存权重路径 如runs/train/exp1/weights
    # ave_dir 是之前那行 Path(opt.save_dir) 得到的训练结果保存目录（比如 runs/train/exp）。
    w = save_dir / 'weights'  # weights dir
    # parents=True：如果父目录不存在，递归创建。
    # exist_ok=True：如果目录已经存在，不会报错。
    # w.parent 是 w 的父目录
    # 如果是超参数进化模式 (evolve=True)，就只创建父目录（不创建 weights 子目录），因为进化模式下会用不同子目录保存权重。
# 否则（普通训练模式）直接创建 weights 目录。
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    # 在 weights 目录下定义两个模型权重文件路径：
# last → 训练过程的最新权重文件（每个 epoch 都会更新）。
# best → 在验证集表现最好的权重文件（通常是 mAP 最高时保存）。
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    '''
    1.3 读取hyp(超参数)配置文件
    '''
    # 加载超参数配置 并保存到 opt 中
    # 检查 hyp 是否是 字符串。
# 如果是字符串，说明它是 超参数 YAML 文件的路径，而不是直接的字典
    if isinstance(hyp, str):
        # 打开超参数文件，errors='ignore' 用于忽略编码异常。
        with open(hyp, errors='ignore') as f:
            # 解析 YAML 文件为 Python 字典。
            # 加载yaml文件
            hyp = yaml.safe_load(f)  # load hyps dict
    # 打印超参数
    LOGGER.info(colorstr('hyperparameters: ') +
                ', '.join(f'{k}={v}' for k, v in hyp.items()))
    # 将解析后的超参数字典 hyp 复制一份保存到 opt.hyp。
# 为什么用 .copy()？
# 避免后续训练过程中修改 hyp 时，影响到原始字典，保证安全
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        # 普通训练模式下，保存超参数和训练配置，方便复现。
        yaml_save(save_dir / 'hyp.yaml', hyp)
        # 超参数进化时保存训练的全部配置参数路径opt 是 命令行参数对象（argparse.Namespace），vars(opt) 将其转换为 字典
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    '''
    1.5 加载相关日志功能:如tensorboard,logger,wandb
    '''
    # Loggers
    # 这个变量后面会存放 data.yaml 解析出来的内容，比如训练集路径、验证集路径、类别信息等
    data_dict = None
    # 设置wandb和tb两种日志, wandb和tensorboard都是模型信息，指标可视化工具
    # -1 表示非分布式训练（单机单卡）。
# 0 表示分布式训练的主进程（master）。
# 只有主进程才会执行下面的日志初始化和数据集检查，防止多进程重复写日志
    if RANK in {-1, 0}:
        # 创建 Loggers 类的实例，用于管理所有日志输出（终端、文件、本地/远程监控工具）。
        # Loggers 会负责：
        # 终端日志打印（console）
        # 文件日志记录（results.txt）
        # 可视化日志（TensorBoard、WandB、ClearML 等）
        # 远程数据集下载（remote_dataset 属性）
        loggers = Loggers(save_dir, weights, opt, hyp,
                          LOGGER)  # loggers instance
        # Register actions
        # 注册回调
        # methods(loggers) 会获取 loggers 对象的所有方法名（例如 on_train_start, on_epoch_end）
        for k in methods(loggers):
            # 把这些方法注册到 YOLOv5 的 回调系统 callbacks 里。
            # 这样在训练流程中，当触发 on_train_start 等事件时，就会调用 loggers 里的对应方法。
            # 作用：让日志记录器参与到训练的每一个关键阶段。
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        # 处理远程数据集
        # 有些用户的数据集可能存在远程平台（如 WandB Artifact、Google Drive）。
# remote_dataset 会处理：
# 如果是远程路径 → 下载到本地 → 返回数据集配置信息（字典）
# 如果是本地路径 → 直接读取配置
        data_dict = loggers.remote_dataset
        # 恢复训练
        if resume:  # If resuming runs from remote artifact
            # 如果是 恢复训练模式（resume=True），从上次保存的训练状态继续：
            # weights：使用上次训练保存的权重
            # epochs：继续剩余的训练轮数
            # hyp：使用上次训练的超参数
            # batch_size：使用上次的 batch 大小
            # 这样可以在中断后无缝接着训练，不会丢失配置。
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    '''
    1.6 配置:画图开关,cuda,种子,读取数据集相关的yaml文件
    '''
    # Config
    # 是否绘制训练、测试图片、指标图等，使用进化算法则不绘制
    # 如果 plots=True，后面会在训练结束时生成 results.png 等文件
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    # 设定随机数种子，保证结果可复现。
# opt.seed 是用户设定的基础种子值。
# +1是为了防止RANK=0 时，种子等于原种子
# YOLOv5 在用 GPU（尤其是 NVIDIA GPU + cuDNN 库）做卷积、池化等操作时，cuDNN 有多种实现方式，有些算法速度快，但内部会用到非确定性操作（比如多线程的结果写入顺序不固定），导致同样的输入和种子，结果可能略有差别。
# deterministic=True 的作用
# 告诉 cuDNN：
# “请不要用那些带随机性的高性能算法，只用能保证结果完全一样的算法。”
# 这样可以保证：
# 同样的代码
# 同样的输入数据
# 同样的随机种子
# 每次运行的输出完全一致（可复现性）
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 在分布式训练时，rank=0 的主进程先执行里面的代码
# 其他进程等待主进程完成后再继续，避免多进程同时下载数据集
# 其他进程（LOCAL_RANK != 0）
# 在进入 with 里的代码之前，会阻塞等待主进程完成
# 等主进程搞定数据集后，才会继续执行 check_dataset
# 因为这时候数据集已经在本地了，所以不会重复下载
# check_dataset(data) 内部会先检查本地路径是否已经有数据
# 如果有，就直接返回，不会再下载。
# 所以即使其他进程执行到这里，下载逻辑也不会触发。
    with torch_distributed_zero_first(LOCAL_RANK):
        # 如果 data_dict 是 None，就调用 check_dataset(data) 检查/解析数据集配置文件（YAML）。
        # check_dataset 会返回一个字典
        data_dict = data_dict or check_dataset(data)  # check if None
    # 提取训练集和验证集路径
    train_path, val_path = data_dict['train'], data_dict['val']
    # 如果 single_cls=True，说明只检测一个类别 → nc=1
# 否则从数据集配置中读取 nc（类别数量）
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    # 如果 single_cls=True 且 数据集的 names 不是单个类别 → 用 {0: 'item'} 作为类别名
# 否则直接用数据集的 names 列表：
    names = {0: 'item'} if single_cls and len(
        data_dict['names']) != 1 else data_dict['names']  # class names
    # 判断是否是 COCO 数据集
    # 如果验证集路径是字符串，并且以 coco/val2017.txt 结尾 → 说明数据集是标准 COCO 验证集。
# 这样后续评估时会按 COCO 的 mAP 计算方式来做（比如 mAP@0.5:0.95）
    is_coco = isinstance(val_path, str) and val_path.endswith(
        'coco/val2017.txt')  # COCO dataset

    ''' =====================2.model：加载网络模型==========================  '''
    # Model
    # Model 载入模型
    # 检查文件后缀是否是.pt
    check_suffix(weights, '.pt')  # check weights
    # 是pt文件则 pretrained=True
    pretrained = weights.endswith('.pt')
    '''
    2.1预训练模型加载
    '''
    if pretrained:
        # 只允许 主进程（LOCAL_RANK=0） 下载模型文件
        with torch_distributed_zero_first(LOCAL_RANK):
            # 如果本地不存在就从google云盘中自动下载模型
            # 通常会下载失败，建议提前下载下来放进weights目录
            # download if not found locally
            weights = attempt_download(weights)
        # 加载权重文件（checkpoint）。
# map_location='cpu' 的作用：
# 避免一开始直接加载到 GPU，防止大模型占用过多显存甚至内存泄漏
# 后续再 .to(device) 移动到 GPU
        # load checkpoint to CPU to avoid CUDA memory leak
        ckpt = torch.load(weights, map_location='cpu')
        # 如果用户提供了 cfg 文件 → 用它创建模型
# 否则使用 checkpoint 中保存的模型结构 YAML
# ch=3 → 输入通道数（RGB 图像）
# nc → 类别数
# anchors=hyp.get('anchors') → 使用超参数里指定的锚框
# .to(device) → 将模型移动到 CPU/GPU
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get(
            'anchors')).to(device)  # create
        # 当用户提供了 cfg 或超参数里指定了 anchors 且不是恢复训练：
# 不加载 checkpoint 里的 anchor 参数
# 因为用户自定义的 anchor 会覆盖 checkpoint 的 anchor
# 否则 exclude=[]，加载 checkpoint 所有参数
# 如果用户自定义了 anchors：
# 就不要加载 checkpoint 的 anchor，防止覆盖
# 如果是恢复训练（resume=True）：
# 说明 checkpoint 本身就是训练状态 → anchors 可以直接加载
# 这样做可以让 预训练权重和自定义 anchor 配置共存。
        exclude = ['anchor'] if (cfg or hyp.get(
            'anchors')) and not resume else []  # exclude keys
        # 获取 checkpoint 的 state_dict
        # checkpoint state_dict as FP32
        csd = ckpt['model'].float().state_dict()
        # intersect_dicts：
# 找到 checkpoint 与当前模型结构 共有的参数键
# 排除 exclude 列表里的参数
# load_state_dict(csd, strict=False)：
# 加载权重到模型
# strict=False：允许 checkpoint 中少一些参数或多一些参数
        csd = intersect_dicts(csd, model.state_dict(),
                              exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        # 输出日志：
# 成功加载的参数数量 / 模型总参数数量
# 帮助调试是否权重加载完整
        LOGGER.info(
            f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        # 如果不是 .pt 文件（没有预训练权重），直接用 cfg 创建模型并初始化参数。
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get(
            'anchors')).to(device)  # create
        # 检查 AMP（混合精度训练）
    # check_amp 会检查当前环境是否支持 Automatic Mixed Precision (AMP)
# AMP 可以加速训练并减少显存占用
    amp = check_amp(model)  # check AMP

    '''
    2.2 冻结层
    '''
    # Freeze 冻结训练的网络层
    """
    冻结模型层,设置冻结层名字即可
    作用：冰冻一些层，就使得这些层在反向传播的时候不再更新权重,需要冻结的层,可以写在freeze列表中
    freeze为命令行参数，默认为0，表示不冻结
    """
    # 如果传入的 freeze 长度大于 1 → 直接使用列表
# 否则 → 用 range(freeze[0]) 生成序列
# freeze = ['model.0.', 'model.1.', ...]
# 字符串用于匹配模型层的名字（named_parameters()）
    freeze = [f'model.{x}.' for x in (freeze if len(
        freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # 遍历模型参数
    # model.named_parameters() 返回一个迭代器，包含模型的所有参数名和对应的参数张量。
    # 目的是 遍历模型所有权重和偏置，决定哪些层需要训练、哪些层冻结
    for k, v in model.named_parameters():
        # 先将所有参数都设为可训练
        # 方便后续有选择地冻结特定层
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        # 遍历 freeze 列表，判断参数名 k 是否属于需要冻结的层：
# any(x in k for x in freeze) → 匹配层名中包含 model.0. 或 model.1. 等
        if any(x in k for x in freeze):
            # 打印日志 freezing model.0.conv.weight
            # 设置 requires_grad = False → 该参数在训练时不会被更新
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size 设置训练和测试图片尺寸
    # model.stride：模型每个检测头的 stride（步长）
# YOLOv5 通常有 3 个检测头（例如 8、16、32）
# 表示特征图相对于原图的下采样倍数
# model.stride.max()：取三个检测头中最大的步长（通常是 32）
# max(..., 32)：确保最小是 32，避免太小影响推理
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # opt.imgsz：用户在命令行设置的图像输入尺寸（例如 640）
    # check_img_size：
# 检查是否是 gs 的整数倍
# 如果不是 → 调整到最近的合法值
# floor=gs*2 → 设置最小尺寸（一般是 64）
# 原因：YOLOv5 的网络结构需要输入尺寸是最大 stride 的倍数，否则最后的特征图大小不匹配
    # verify imgsz is gs-multiple
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)

    # Batch size
    # 估算最优 batch size
    # RANK == -1 → 不是分布式训练（单 GPU）
# batch_size == -1 → 用户没有指定 batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        # 调用 check_train_batch_size，动态测试 GPU 能承受的最大 batch size
        # 会用给定的 model 和 imgsz 反复试验，直到显存溢出为止
        # amp（自动混合精度）可能会影响最大 batch size（AMP 更省显存）
        batch_size = check_train_batch_size(model, imgsz, amp)
        # 把最终 batch size 记录下来，方便后续分析
        loggers.on_params_update({'batch_size': batch_size})

    '''
    2.3 优化器设置
    '''
    # Optimizer
    # 这是 YOLOv5 设计的一个基准批量大小，用来做学习率、权重衰减等超参数的缩放参考
# 目的：即使用户实际 batch size 不一样，也能保证训练超参数等效
    nbs = 64  # nominal batch size
    """
    nbs = 64
    batchsize = 16
    accumulate = 64 / 16 = 4
    模型梯度累计accumulate次之后就更新一次模型 相当于使用更大batch_size
    """
    # 如果实际 batch size 比 64 小，比如 16：
# nbs / batch_size = 64 / 16 = 4
# 表示会累积 4 次梯度再执行一次优化步骤
# 如果 batch size ≥ 64：
# accumulate = 1 → 正常每个 batch 更新一次
    # accumulate loss before optimizing
    accumulate = max(round(nbs / batch_size), 1)
    # 如果批量大小是 128：
# 梯度累积次数：64 ÷ 128 = 0.5（取 1）
# 实际有效批量大小：128 × 1 = 128
# 新的权重衰减值 = 0.0005 × (128 ÷ 64) = 0.001（变大一倍）
# 保证不管 batch size 怎么变，正则化强度是一致的
# 避免小 batch 时 weight decay 过大，大 batch 时 weight decay 过小
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    # 创建优化器
    # smart_optimizer 是一个封装函数，根据 opt.optimizer（sgd / adam / adamw 等）创建优化器
# 会自动将模型参数传入，并使用超参数：
# 初始学习率 hyp['lr0']
# 动量 hyp['momentum']
# 权重衰减 hyp['weight_decay']
    optimizer = smart_optimizer(
        model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    '''
    2.4 学习率设置
    '''
    # Scheduler 设置学习率策略:两者可供选择，线性学习率和余弦退火学习率
    # 使用余弦退火
    if opt.cos_lr:
        # one_cycle 会生成一个长度为 epochs 的函数，表示每个 epoch 的学习率系数。此处是单周期
        # 初始值 = 1（即 100% 的初始学习率）
        # 最终值 = hyp['lrf']（最终学习率比例，比如 0.1 代表降到初始学习率的 10%）
        # 学习率下降曲线是余弦形状：前期下降快，后期平缓。取余弦函数X轴上方大于零部分函数
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        # 线性下降
        # 输入 x：当前训练步数（或 epoch）。
        # 输出：当前学习率，按线性规则从 1.0 衰减到 hyp['lrf']。
        # 初始值（x=0）：
        # lr = (1 - 0) * (1.0 - hyp['lrf']) + hyp['lrf'] = 1.0
        # 结束时（x=epochs）：
        # lr = (1 - 1) * (1.0 - hyp['lrf']) + hyp['lrf'] = hyp['lrf']
        def lf(x): return (1 - x / epochs) * \
            (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    # 创建学习率调度器
    # LambdaLR 是 PyTorch 的调度器，可以用一个函数 lf(epoch) 决定每个 epoch 的学习率缩放系数。
# 缩放系数 × 初始学习率 = 当前学习率
# 每个 epoch 更新时，都会用 lf 计算新的学习率
    # plot_lr_scheduler(optimizer, scheduler, epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    '''
    2.5 训练前最后准备
    '''
    # EMA 指数滑动平均
    # EMA 会维护一个“平滑版”的模型参数，用于验证和推理时更稳定。
# 原理是每次更新参数时，让 EMA 参数 = α × 旧 EMA 参数 + (1 - α) × 当前模型参数。
# 这样可以减少训练波动带来的影响。
# RANK in {-1, 0}：只在主进程（RANK=-1 或 0）上启用 EMA，多 GPU 分布式训练时，其他进程不用维护 EMA（防止重复占内存）
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume 断点续训
    # 断点续训其实就是把上次训练结束的模型作为预训练模型，并从中加载参数
    # best_fitness：存储历史上模型在验证集上的最佳表现（通常是 mAP 或综合指标）。刚开始设为 0.0。
    # start_epoch：记录从哪一轮开始训练。刚开始设为 0（意味着从头开始训练）。
    best_fitness, start_epoch = 0.0, 0
    # 如果加载了预训练权重
    # 这里的 pretrained 是在前面通过：pretrained = weights.endswith('.pt')判断出来的。也就是说，如果你的 weights 文件是 .pt 格式（YOLOv5 保存的 checkpoint），pretrained 就是 True。
    if pretrained:
        # 如果要恢复训练
        if resume:
            # resume=True 表示你不是全新训练，而是想从上一次训练中断的地方继续。
            # smart_resume 会做几件事：
            # 读取 ckpt（checkpoint）里的训练记录：
            # 上次训练到的 epoch（赋给 start_epoch）。
            # 上次的最佳性能指标（赋给 best_fitness）。
            # 模型权重（加载到 model）。
            # 优化器状态（恢复优化器内部的动量、学习率等）。
            # EMA 权重（如果有的话，也恢复）。
            # 根据上次中断的位置，调整剩余训练 epochs。
            # 确保继续训练时，学习率、权重衰减等参数跟中断前保持一致。
            best_fitness, start_epoch, epochs = smart_resume(
                ckpt, optimizer, ema, weights, epochs, resume)
        # 删除不再需要的变量
        # ckpt：完整 checkpoint（包含模型、优化器、训练状态等）。
# csd：checkpoint 的 state_dict()（只包含模型参数）。
# 这两个在加载进 model 和 optimizer 后就没用了，删掉可以释放内存。
        del ckpt, csd

    # DP mode 使用单机多卡模式训练，目前一般不使用
    # rank为进程编号。如果rank=-1且gpu数量>1则使用DataParallel单机多卡模式，效果并不好（分布不平均）
    # rank=-1且gpu数量=1时,不会进行分布式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        # YOLOv5 官方不推荐使用 Data Parallel (DP)，因为它效率低、扩展性差。
# 推荐使用 Distributed Data Parallel (DDP)，尤其是在多 GPU 环境下，性能更好。
# PyTorch 自带的 DataParallel：把模型复制到每个 GPU 上，自动把 batch 切分到各个 GPU。
# 优点：代码简单，容易上手。
# 缺点：
# GPU 利用率不均衡
# 对大 batch 或多机训练效率低
# 单机多 GPU 性能有限
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    # opt.sync_bn：用户是否启用同步 BN（命令行参数）
# cuda：有 GPU 可用
# RANK != -1：表示是分布式训练（每个进程对应一个 GPU）
# 意思：只有在 多 GPU 分布式训练 且用户开启 sync_bn 时才执行
    if opt.sync_bn and cuda and RANK != -1:
        #         PyTorch 提供的 SyncBatchNorm 可以在 多个 GPU 上同步计算 batch 的均值和方差。
        # 默认的 普通 BatchNorm 只在单个 GPU 的 mini-batch 上计算均值和方差，多 GPU 时会导致统计偏差。
        # convert_sync_batchnorm(model) 会把模型里的所有 BatchNorm 层替换为 SyncBatchNorm。
        # to(device) 把模型移到当前 GPU。#
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        # 打印提示，表示已经启用同步 BN，方便调试和确认训练配置。
        LOGGER.info('Using SyncBatchNorm()')

    ''' =====================3.加载训练数据集==========================  '''
    '''
    3.1 创建数据集
    '''
    # Trainloader 训练集数据加载
    '''
      返回一个训练数据加载器，一个数据集对象:
      训练数据加载器是一个可迭代的对象，可以通过for循环加载1个batch_size的数据
      数据集对象包括数据集的一些参数，包括所有标签值、所有的训练数据路径、每张图片的尺寸等等
    '''
    # train_path：训练集路径（通常是 .txt 列表文件，里面列出所有训练图片的路径）
    # batch_size // WORLD_SIZE：每个 GPU 分到的 batch 大小（分布式训练时要平分）
    # single_cls：是否将所有类别视为 1 类（常用于数据少时的单类别训练）。
    # augment=True：开启数据增强（训练集才会用）。
    # cache：是否提前缓存图片（加快 IO）
    # rect：是否使用矩形训练（保持宽高比，提高推理精度）
    # rank=LOCAL_RANK：分布式训练时的进程 ID
    # workers：DataLoader 的进程数（多线程加速数据加载）
    # image_weights：是否按类别权重采样图片（处理类别不平衡问题）
    # quad：是否使用四图拼接（提高显存利用率）
    # prefix：打印日志时的前缀（这里是 train:）
    # shuffle=True：是否打乱数据（训练集必须打乱）
    # seed=opt.seed：随机种子，保证可复现
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    # dataset.labels 是一个 列表，每个元素是当前图片的标签数组，形状为 (num_objects, 5)（列顺序：类别ID、x_center、y_center、w、h）。
# np.concatenate(..., 0) 把所有图片的标签在第 0 维拼成一个大数组。
# 这样就能一次性获取整个训练集的标签信息。
    labels = np.concatenate(dataset.labels, 0)
    # labels[:, 0]：取出所有标签的类别 ID 列。
# .max()：找到最大的类别 ID（例如最大类别是 79，就说明类别范围是 0～79）
    mlc = int(labels[:, 0].max())  # max label class
    # mlc < nc：确保标签中最大的类别 ID 不超过模型设置的类别数 nc
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    # 单gpu或分布式训练的主进程（RANK=-1 或 0）会执行以下操作，以节省内存和 I/O 资源。
    if RANK in {-1, 0}:
        # 加载验证集
        # 这里的 [0] 是因为 create_dataloader 返回 (dataloader, dataset)，验证只需要 dataloader。
        # 验证时的 batch_size 是训练集的一倍，因为验证集不需要反向传播，显存压力小。
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]
        '''
        3.2 计算anchor
        '''
        if not resume:
            if not opt.noautoanchor:  # 计算默认锚框anchor与数据集标签框的高宽比
                # check_anchors：对数据集做统计，看看现有的 anchor 尺寸是否匹配数据集的目标框，如果差距太大会重新计算。
                # run AutoAnchor
                check_anchors(dataset, model=model,
                              thr=hyp['anchor_t'], imgsz=imgsz)
                '''
                参数dataset代表的是训练集，hyp['anchor_t']是从配置文件hpy.scratch.yaml读取的超参数，anchor_t:4.0
                当配置文件中的anchor计算bpr（best possible recall）小于0.98时才会重新计算anchor。
                best possible recall最大值1，如果bpr小于0.98，程序会根据数据集的label自动学习anchor的尺寸
                '''
            # 先转 FP16 再回 FP32，可以减少浮点精度的累计误差（和权重初始化精度优化有关）。
            model.half().float()  # pre-reduce anchor precision
        # 触发 "on_pretrain_routine_end" 事件，把标签和类别名传给回调系统（可能用于日志、可视化、统计等
        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode 多 GPU 分布式训练时
    if cuda and RANK != -1:
        # 多 GPU 分布式训练时，用 DistributedDataParallel（DDP）封装模型。
        # smart_DDP 是 YOLOv5 的封装，自动帮你处理 find_unused_parameters 等设置。
        model = smart_DDP(model)

    ''' =====================4.训练==========================  '''
    '''
    4.1 初始化训练需要的模型参数
    '''
    # Model attributes 根据自己数据集的类别数和网络FPN层数设置各个损失的系数
    # YOLOv5 的最后一层是 检测层（Detect），通常有 3 个输出层（对应 3 个特征尺度）。
# .nl 就是检测层的数量。（一般是3）
# 为什么要取这个？👉 因为损失函数的某些超参数（box、cls、obj）需要 按检测层数缩放，保证不同模型（比如 YOLOv5n、YOLOv5x）训练时数值一致
    # number of detection layers (to scale hyps)
    nl = de_parallel(model).model[-1].nl
    # box（边框损失权重）
# 默认按照 3 层来设置，如果检测层数不是 3，就按比例调整。
    hyp['box'] *= 3 / nl  # scale to layers
# cls（分类损失权重）
# nc / 80：把 COCO（80类）为基准的参数，调整为你的数据集类别数。
# 3 / nl：再按检测层数缩放
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    # obj（目标置信度损失权重）
# (imgsz / 640)^2：输入图像越大，目标数也越多，损失要按图像面积比例缩放。
# 3 / nl：按检测层数缩放
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    # 设置 标签平滑系数，防止过拟合。
# （比如真实标签 1，会变成 0.95，非目标类从 0 变成 0.05，避免模型过度自信。）
    hyp['label_smoothing'] = opt.label_smoothing
    # 把类别数和超参数存到模型里，方便训练和推理过程中直接使用。
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    # 计算 类别权重，用来缓解类别不平衡：
# 出现多的类 → 权重小（降低损失贡献）。
# 出现少的类 → 权重大（增加损失贡献）
# 乘 nc 是为了让权重缩放跟 类别数规模匹配，避免在小数据集和大数据集之间失衡。
    model.class_weights = labels_to_class_weights(
        dataset.labels, nc).to(device) * nc  # attach class weights
    # 保存类别名称（比如 {0: 'person', 1: 'car', ...}），方便后续训练日志和推理输出。
    model.names = names

    '''
    4.2 训练热身部分
    '''
    # Start training
    # 训练开始时间，用于最后统计总耗时。
    t0 = time.time()
    # 每个 epoch 的 batch 数量（多少个 mini-batch）。
    nb = len(train_loader)  # number of batches
    # number of warmup iterations, max(3 epochs, 100 iterations)
    # Warmup 的作用是让学习率、动量等参数逐渐增加，避免训练一开始就震荡或发散
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    # 记录上次优化器更新的 step（初始 -1，表示还没更新过）
    last_opt_step = -1
    # 存放每个类别的 mAP（初始为 0）
    maps = np.zeros(nc)  # mAP per class
    # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    # epoch ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            iw = labels_to_image_weights(
                dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(
                range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem',
                    'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            # progress bar
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        optimizer.zero_grad()
        # batch -------------------------------------------------------------
        for i, (imgs, targets, paths, _) in pbar:
            callbacks.run('on_train_batch_start')
            # number integrated batches (since train start)
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / \
                255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(
                    ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(
                        ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(
                    imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # new shape (stretched to gs-multiple)
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = nn.functional.interpolate(
                        imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(
                    pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / \
                    (i + 1)  # update mean losses
                # (GB)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni,
                              imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(
                model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # Update best mAP
            # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi = fitness(np.array(results).reshape(1, -1))
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals,
                          epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch,
                              final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            # broadcast 'stop' to all ranks
            dist.broadcast_object_list(broadcast_list, 0)
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(
            f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(
                            mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # 预训练权重文件
    parser.add_argument('--weights', type=str, default=ROOT /
                        'yolov5s.pt', help='initial weights path')
    # 训练模型
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # 训练路径，包括训练集，验证集，测试集的路径，类别总数等
    parser.add_argument('--data', type=str, default=ROOT /
                        'data/coco128.yaml', help='dataset.yaml path')
    # 指定超参数（hyperparameters）配置文件路径
    # hpy超参数设置文件（lr/sgd/mixup）./data/hyps/下面有5个超参数设置文件，每个文件的超参数初始值有细微区别，用户可以根据自己的需求选择其中一个
    parser.add_argument('--hyp', type=str, default=ROOT /
                        'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='total batch size for all GPUs, -1 for autobatch')
    # 设置图片大小, 默认640*640
    parser.add_argument('--imgsz', '--img', '--img-size', type=int,
                        default=640, help='train, val image size (pixels)')
    # 是否采用矩形训练，默认为False
    # 矩形训练就是在每个 batch 内，将尺寸相似的图像分组到一起，并使用它们的最大尺寸来统一缩放，避免过多 padding，从而提升显存效率和训练速度。
    parser.add_argument('--rect', action='store_true',
                        help='rectangular training')
    # resume: 是否接着上次的训练结果，继续训练
    # 矩形训练：将比例相近的图片放在一个batch（由于batch里面的图片shape是一样的）
    parser.add_argument('--resume', nargs='?', const=True,
                        default=False, help='resume most recent training')
    # 不保存模型  默认False(保存)  在./runs/exp*/train/weights/保存两个模型 一个是最后一次的模型 一个是最好的模型
    # 不建议运行代码添加 --nosave
    parser.add_argument('--nosave', action='store_true',
                        help='only save final checkpoint')
    # 最后进行测试, 设置了之后就是训练结束都测试一下， 不设置每轮都计算mAP, 建议不设置
    parser.add_argument('--noval', action='store_true',
                        help='only validate final epoch')
    #  不自动调整anchor, 默认False, 自动调整anchor
    parser.add_argument('--noautoanchor', action='store_true',
                        help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true',
                        help='save no plot files')
    # --evolve 参数，用于启动超参数进化（evolution）过程，让 YOLOv5 自动通过遗传算法优化训练超参数（比如学习率、置信度阈值等）
    # --evolve 参数进化训练”是 YOLOv5 中的一个自动超参数优化机制，用来自动搜索出更好的训练超参数（hyp）组合，从而提升模型性能。
    parser.add_argument('--evolve', type=int, nargs='?', const=300,
                        help='evolve hyperparameters for x generations')
    # 谷歌优盘 / 一般用不到
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # 是否提前缓存图片到内存，以加快训练速度，默认False
    parser.add_argument('--cache', type=str, nargs='?',
                        const='ram', help='image --cache ram/disk')
    # 使用图片采样策略，默认不使用
    parser.add_argument('--image-weights', action='store_true',
                        help='use weighted image selection for training')
    # 设备选择
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否进行多尺度训练
    parser.add_argument('--multi-scale', action='store_true',
                        help='vary img-size +/- 50%%')
    # 数据集是否只有一个类别，默认False
    parser.add_argument('--single-cls', action='store_true',
                        help='train multi-class data as single-class')
    # 优化器选择 / 提供了三种优化器
    parser.add_argument('--optimizer', type=str,
                        choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    # 是否使用跨卡同步BN,在DDP模式使用
    parser.add_argument('--sync-bn', action='store_true',
                        help='use SyncBatchNorm, only available in DDP mode')
    # dataloader的最大worker数量 （使用多线程加载图片）
    parser.add_argument('--workers', type=int, default=8,
                        help='max dataloader workers (per RANK in DDP mode)')
    # 训练结果的保存路径
    parser.add_argument('--project', default=ROOT /
                        'runs/train', help='save to project/name')
    # 训练结果的文件名称
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 项目位置是否存在 / 默认是都不存在
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    # 四元数据加载器: 允许在较低 --img 尺寸下进行更高 --img 尺寸训练的一些好处
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # cos-lr: 余弦学习率
    parser.add_argument('--cos-lr', action='store_true',
                        help='cosine LR scheduler')
    # 标签平滑 / 默认不增强， 用户可以根据自己标签的实际情况设置这个参数，建议设置小一点 0.1 / 0.05
    parser.add_argument('--label-smoothing', type=float,
                        default=0.0, help='Label smoothing epsilon')
    # 早停止耐心次数 / 100次不更新就停止训练
    parser.add_argument('--patience', type=int, default=100,
                        help='EarlyStopping patience (epochs without improvement)')
    # 冻结训练 可以设置 default = [0] 数据量大的情况下，建议不设置这个参数
    parser.add_argument('--freeze', nargs='+', type=int,
                        default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    # 多少个epoch保存一下checkpoint
    parser.add_argument('--save-period', type=int, default=-1,
                        help='Save checkpoint every x epochs (disabled if < 1)')

    parser.add_argument('--seed', type=int, default=0,
                        help='Global training seed')
    # 进程编号 / 多卡使用
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    # 在线可视化工具，类似于tensorboard工具
    parser.add_argument('--entity', default=None, help='Entity')
    # 是否上传dataset到wandb tabel(将数据集作为交互式 dsviz表 在浏览器中查看、查询、筛选和分析数据集) 默认False
    parser.add_argument('--upload_dataset', nargs='?', const=True,
                        default=False, help='Upload data, "val" option')
    # 设置界框图像记录间隔 Set bounding-box image logging interval for W&B 默认-1   opt.epochs // 10
    parser.add_argument('--bbox_interval', type=int, default=-1,
                        help='Set bounding-box image logging interval')
    # 使用数据的版本
    parser.add_argument('--artifact_alias', type=str,
                        default='latest', help='Version of dataset artifact to use')
    # 作用就是当仅获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    # 如果传入了 known=True，就使用 parse_known_args()[0] 解析已知参数；
# 否则，就使用 parse_args() 正常解析所有参数。
# parser.parse_args()
# 这是最常用的 argparse 方法
# 它会严格解析所有参数
# 如果命令行中有未定义的参数，它会 抛出错误
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Checks
    '''
    2.1  检查分布式训练环境
    '''
    if RANK in {-1, 0}:  # 若进程编号为-1或0
        print_args(vars(opt))  # 输出所有训练参数
        check_git_status()  # 检测YOLO v5的github仓库是否更新，若已更新，给出提示
        check_requirements()  # 检查requirements.txt所需包是否都满足

    '''
    2.2  判断是否断点续训
    '''
    # Resume (from specified or most recent last.pt)
    # 只在“普通本地训练的 resume”情况下才执行以下逻辑
    # 用户指定了 --resume
    # check_comet_resume(opt) 的作用是：
# 检查当前是否启用了 Comet 的自动 resume 功能
# Comet.ml 是一个类似于 WandB 的工具，可以远程记录训练日志、指标、模型文件等。它有自己的 resume 机制，可以在 Web UI 上点一下就恢复上次训练。
# 当前不是超参数进化（evolve）过程
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        # 如果 opt.resume 是字符串（比如路径），检查文件是否存在；这个函数会返回合法的 .pt 路径（字符串），否则报错。
        # 否则自动从默认目录中找最近的一次训练（get_latest_run()）；
        # 返回 last 是一个 Path 对象，指向 last.pt
        last = Path(check_file(opt.resume) if isinstance(
            opt.resume, str) else get_latest_run())
        # last 通常是 runs/train/exp/weights/last.pt
# 所以 last.parent.parent 是 runs/train/exp
# 所以最终指向：runs/train/exp/opt.yaml
# 这是上一次训练保存的所有 CLI 参数配置。
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        # 记录当前用户提供的 --data 配置
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            # 当读取文件过程中遇到无法解码的字符（比如编码错误、非法字符），就忽略掉这些错误字符，而不是抛出异常。
            with open(opt_yaml, errors='ignore') as f:
                # 使用 yaml.safe_load 读取 opt.yaml 文件，转成 Python 字典；
                # d 里包含之前训练时所有的命令行参数。
                d = yaml.safe_load(f)
        else:
            # 从 last.pt 里取出 opt 字典，作为替代（也保存在模型权重里）
            d = torch.load(last, map_location='cpu')['opt']
        # 用 d 创建新的 argparse.Namespace，模拟上次训练的 CLI 参数；
# 相当于恢复成上次训练时的状态。
        opt = argparse.Namespace(**d)  # replace
        # 手动修正几个关键参数：
        # opt.cfg = ''	不再使用 YAML 配置文件（已恢复了）
        # opt.weights = str(last)	把 last.pt 作为起始权重
        # opt.resume = True	显式标记是 resume 模式
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        # 如果用户传的是在线 data.yaml（如 GitHub 链接），先下载下来；
# 避免 resume 时 HUB 自动认证失败（比如 GitHub 的 403 问题）
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:  # 不使用断点续训，就从文件中读取相关参数
        # # check_file （utils/general.py）的作用为查找/下载文件 并返回该文件的路径。
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(
                opt.hyp), str(opt.weights), str(opt.project)  # checks
        # 如果模型文件和权重文件为空，弹出警告
        assert len(opt.cfg) or len(
            opt.weights), 'either --cfg or --weights must be specified'
        # 如果要进行超参数进化，重建保存路径
        if opt.evolve:
            # 设置新的项目输出目录
            # if default project name, rename to runs/evolve
            if opt.project == str(ROOT / 'runs/train'):
                opt.project = str(ROOT / 'runs/evolve')
            # 将resume传递给exist_ok
            # pass resume to exist_ok and disable resume
            opt.exist_ok, opt.resume = opt.resume, False
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        # 根据opt.project生成目录，并赋值给opt.save_dir  如: runs/train/exp1
        opt.save_dir = str(increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    '''
    2.3  判断是否分布式训练
    DDP（Distributed Data Parallel）用于单机或多机的多GPU分布式训练，但目前DDP只能在Linux下使用。这部分它会选择你是使用cpu还是gpu，假如你采用的是分布式训练的话，它就会额外执行下面的一些操作，我们这里一般不会用到分布式，所以也就没有执行什么东西。
    '''
    # DDP mode
    # DDP mode -->  支持多机多卡、分布式训练
    # 选择程序装载的位置
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 当进程内的GPU编号不为-1时，才会进入DDP
    # 分布式方式启动才会设置LOCAL_RANK值，因此默认-1就是单机训练
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        # 不能使用图片采样策略
        assert not opt.image_weights, f'--image-weights {msg}'
        # 不能使用超参数进化
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != - \
            1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        # 设置装载程序设备
#         设置当前进程默认使用的 GPU 设备。
# 在多进程 DDP 中，每个进程都只能访问自己的 GPU。这行代码告诉 PyTorch：
# 这个进程以后所有的 .cuda() 操作，都默认指向 LOCAL_RANK 对应的 GPU。
        torch.cuda.set_device(LOCAL_RANK)
        # 显式创建一个表示当前使用 GPU 的 device 对象
        device = torch.device('cuda', LOCAL_RANK)
        # 初始化 PyTorch 的进程通信组，开启分布式训练模式。
# 如果没有这一步，DDP 模型在反向传播时不会同步梯度。
# nccl（推荐）：NVIDIA 的高效 GPU 通信库，GPU 间通信非常快
# gloo：PyTorch 提供的后备方案，主要用于 CPU 或非 NVIDIA GPU
        dist.init_process_group(
            backend='nccl' if dist.is_nccl_available() else 'gloo')

    '''
    2.4  判断是否进化训练
    '''
    # Train
    if not opt.evolve:  # Train 训练模式: 如果不进行超参数进化，则直接调用train()函数，开始训练
        # 开始训练
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Evolve hyperparameters (optional) 遗传进化算法，边进化边训练
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # (突变尺度相当于系数，最小值，最大值)
        # 超参数列表(突变范围 - 最小值 - 最大值)
        meta = {
            # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lr0': (1, 1e-5, 1e-1),
            # final OneCycleLR learning rate (lr0 * lrf)
            'lrf': (1, 0.01, 1.0),
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            # focal loss gamma (efficientDet default gamma=1.5)
            'fl_gamma': (0, 0.0, 2.0),
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            # image HSV-Saturation augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            # image perspective (+/- fraction), range 0-0.001
            'perspective': (0, 0.0, 0.001),
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)
        # 加载默认超参数
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            # 如果超参数文件中没有'anchors'，则设为3
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
         # 使用进化算法时，仅在最后的epoch测试和保存
        opt.noval, opt.nosave, save_dir = True, True, Path(
            opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            # 用 gsutil cp 命令从你提供的 GCS bucket 下载 evolve.csv 文件到本地。
            # 相当于运行了：gsutil cp gs://my-bucket/evolve.csv runs/evolve/evolve.csv
            # 这会使得：
            # 你上一次训练保存的进化结果能够被当前训练接着用；
            # 进化不需要从零开始（加快收敛速度）
            # download evolve.csv if exists
            subprocess.run([
                'gsutil',
                'cp',
                f'gs://{opt.bucket}/evolve.csv',
                str(evolve_csv),])
        # 选择超参数的遗传迭代次数 默认为迭代300次
        for _ in range(opt.evolve):  # generations to evolve
            # 如果evolve.csv文件存在
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                # 遗传算法核心步骤是：
                # 选择（Selection）：从历史种群中选出优秀“父代”；
                # 变异（Mutation）：在父代基础上添加扰动，生成“子代”；
                # 评估（Fitness Evaluation）：每一代训练模型打分，记录得分。
                # 'single': 选出最好的一个父代；
                # 'weighted': 选出多个好父代加权平均；
                # 通常默认是 'single'，意味着变异是围绕一个个体进行的。
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                # num_generations	一共有多少代历史记录（即 CSV 的行数 - 1）
                # num_hyperparameters	每代保存了多少个超参数（比如 lr0, lrf, ...
                # +1	最后一列是模型的得分（如 mAP），用来排序和筛选
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                # 只考虑最近 5 代最优秀的个体
                n = min(5, len(x))  # number of previous results to consider
#                 按照适应度（fitness）得分降序排序，取前 n 个（最好的 n 个）
# fitness(x) 是你自定义的打分函数（一般基于 mAP、F1 等）
# fitness()为x前四项加权 [P, R, mAP@0.5, mAP@0.5:0.95]
                # np.argsort只能从小到大排序, 添加负号实现从大到小排序, 算是排序的一个代码技巧
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                # 根据(mp, mr, map50, map)的加权和来作为权重计算hyp权重
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    # 根据权重的几率随机挑选适应度历史前5的其中一个
                    # random.choices 会返回一个列表，即便只选择一个元素，所以你通常会写 [0] 来取值
                    x = x[random.choices(range(n), weights=w)[
                        0]]  # weighted selection
                elif parent == 'weighted':
                    # 对hyp乘上对应的权重融合层一个hpy, 再取平均(除以权重和
                    # 对 x 的每一行乘以对应的权重
                    # 对所有样本按列求和，即将每个特征的加权值累加#
                    # 除以总权重，得到加权平均
                    x = (x * w.reshape(n, 1)).sum(0) / \
                        w.sum()  # weighted combination

                # Mutate
                # Mutate 突变（超参数进化）
                # 在前面你已经选出了一个“父代超参数组合” x，这一步就是对其进行带噪声的突变，产生“子代超参数组合”
                # mp：每个超参数发生突变的概率为 80%
# s：突变的标准差（变异幅度）
                mp, s = 0.8, 0.2  # mutation probability, sigma
                # 使用当前时间戳作为随机种子（保证每次运行变异不一样）
                npr = np.random
                npr.seed(int(time.time()))
                # 获取突变初始值, 也就是meta三个值的第一个数据
                # 三个数值分别对应着: 变异初始概率, 最低限值, 最大限值(mutation scale 0-1, lower_limit, upper_limit)
                # meta[k][0]：是该超参数的 gain，控制该超参数突变时的“影响程度”
                # g 是一个数组，长度等于超参数个数
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)  # 超参数个数
                v = np.ones(ng)  # 初始化变异因子为 1（即不变）
                # 如果没有任何突变（全部 v == 1），就重新抽样
                # 防止子代与父代完全一样，必须有至少一个超参数发生变化
                # all() 是 Python 的一个内置函数，用来判断一个可迭代对象中的所有元素是否都为 True。
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    # npr.random(ng) < mp：以 mp=0.8 的概率选中每个超参数进行突变 → 得到布尔 mask
                    # npr.randn(ng)：正态分布扰动（均值 0，方差 1）
                    # npr.random()：再乘一个 [0,1) 的随机数，使变异大小分布更加连续
                    # * s：缩放变异幅度（例如 0.2）
                    # + 1：通过+1相当于变成乘法变异因子了
                    # .clip(0.3, 3.0)：限制突变因子范围，防止突变过大或过小
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng)
                         * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    # 父代值 × 突变因子 = 子代值
                    # x[i + 7]：是从 evolve.csv 中读出来的父代超参数值（注意前 7 列是别的字段，所以超参数从第 7 列开始）
                    # v[i]：是该超参数的突变因子
                    # hyp[k]：被赋值为突变后的值
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            # 约束突变后的超参数 `hyp[k]` 在合法范围内
            for k, v in meta.items():
                # 先限定最小值，选择二者之间的大值 ，这一步是为了防止hyp中的值过小
                hyp[k] = max(hyp[k], v[1])  # lower limit
                # 再限定最大值，选择二者之间的小值
                hyp[k] = min(hyp[k], v[2])  # upper limit
                # 四舍五入到小数点后五位
                hyp[k] = round(hyp[k], 5)  # significant digits
                # 最后的值应该是 hyp中的值与 meta的最大值之间的较小者

            # Train mutation
            # Train mutation 使用突变后的参超，测试其效果
            # 返回的 results 包含这次训练的指标结果，比如 mAP、loss 等。
            results = train(hyp.copy(), opt, device, callbacks)
            # 重置 callback（回调函数）。
            # 每轮训练完后要清空 callback 环境，防止状态污染。
            callbacks = Callbacks()
            # Write mutation results
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            # 把这次突变结果（性能和对应的超参数）打印并保存到 evolve.csv
            # keys 指定要追踪的指标。
# print_mutation(...) 会记录一行 CSV 结果：
# 超参数组合
# 训练结果（如 mAP）
# 保存在 save_dir / evolve.csv
# 下一代突变时会从这个 CSV 中选取 top-N 最好的组合继续“繁殖”
            # 每行前七个数字 (P, R, mAP, F1, test_losses(GIOU, obj, cls)) 之后为hyp
            # 保存hyp到yaml文件
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        # 将结果可视化 / 输出保存信息
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
