# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

'''============1.导入安装好的python库=========='''
from utils.torch_utils import select_device, smart_inference_mode
import json # 实现字典列表和JSON字符串之间的相互解析
from utils.dataloaders import create_dataloader
from utils.callbacks import Callbacks
from models.common import DetectMultiBackend
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
import sys
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from threading import Thread # python中处理多线程的库
import argparse
import os
import subprocess


'''===================2.获取当前文件的绝对路径========================'''
# 将当前项目添加到系统路径上，以使得项目中的模块可以调用。
# 将当前项目的相对路径保存在ROOT中，便于寻找项目中的文件
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

'''===================3..加载自定义模块============================'''
# yolov5的网络结构(yolov5)
# 和日志相关的回调函数
# 加载数据集的函数
#  定义了一些常用的工具函数
# 在YOLOv5中，fitness函数实现对 [P, R, mAP@.5, mAP@.5-.95] 指标进行加权
# # 定义了Annotator类，可以在图像上绘制矩形框和标注信息
# 定义了一些与PyTorch有关的工具函数

'''======================1.保存预测信息到txt文件====================='''
# predn：单张图片的预测框，格式通常是 [x1, y1, x2, y2, conf, cls]
# save_conf：布尔值，是否在文件里保存置信度 conf。
# shape：原图尺寸 (height, width)，用于归一化坐标。
# file：要保存的目标 .txt 文件路径。


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    # 归一化增益
    # [1,0,1,0] → (w, h, w, h)
    # 用于把[x, y, w, h] 坐标归一化到[0, 1]
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    # 遍历预测框
    # xyxy → [x1, y1, x2, y2]
# conf → 置信度
# cls → 类别 id
    for *xyxy, conf, cls in predn.tolist():
        # xyxy2xywh：把 [x1, y1, x2, y2] 转换为 [x_center, y_center, w, h]
        # 除以 gn → 归一化到 [0,1]
        # .view(-1).tolist() → 转成 Python list [x, y, w, h]
        # -1 是一个特殊值，表示 自动推导这一维度的大小。
        # .view(-1) 就是把张量拉平成一维向量。
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                gn).view(-1).tolist()  # normalized xywh
        # 生成写入行
        # line的形式是： "类别 xywh"，若save_conf为true，则line的形式是："类别 xywh 置信度"
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        # 打开文件追加模式 'a'
        with open(file, 'a') as f:
            # '%g ' * len(line) → 根据行长度动态生成格式字符串
            # .rstrip() 去掉末尾空格
            # % line → 将数值填入
            # \n → 每个框写一行
            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


'''======================2.保存预测信息到coco格式的json字典====================='''
# predn：模型预测结果，形状 (num_boxes, 6)，格式 [x1, y1, x2, y2, conf, cls]。
# jdict：一个 list，用于收集所有预测框，最后可以统一 dump 成 JSON。
# path：当前图片路径，用于提取图片 id。
# class_map：类别映射表，把 YOLO 内部的类别 id 转换成 COCO 的类别 id


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    # path.stem → 图片文件名（不带扩展名）。
    # 如果是数字（比如 "42.jpg"），就转成 int 42；否则保持字符串。
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    # 把 x1,y1,x2,y2 转换为 [x_center, y_center, w, h] 格式
    box = xyxy2xywh(predn[:, :4])  # xywh
    # 这里再把中心坐标 (x_center, y_center) 转换成左上角 (x_min, y_min)，因为 COCO JSON 格式要求 bbox 是 [x_min, y_min, width, height]。
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        # category_id = class_map[int(p[5])] → 把 YOLO 类别 id 映射到 COCO 的类别 id
        # bbox 保留 3 位小数，score 保留 5 位小数
        # 把结果追加进 jdict
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


'''========================三、计算指标==========================='''
# 给定一批预测框和真实框，算出在 不同 IoU 阈值下 哪些预测是正确的


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    # detections: shape [N, 6]，预测框
# [x1, y1, x2, y2, conf, cls]
# labels: shape [M, 5]，真实框
# [cls, x1, y1, x2, y2]
# iouv: IoU 阈值列表，通常是 [0.5, 0.55, 0.6, …, 0.95]，一共 10 个，用于计算 mAP@0.5:0.95
# 返回：
# correct: shape [N, len(iouv)]，布尔矩阵，表示每个预测在不同 IoU 阈值下是否正确
# 建一个矩阵 [N, 10]，先全是 False，等会儿标记 True 表示“该预测在该 IoU 阈值下是正确的”。
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    # 计算 IoU
    # labels[:, 1:] → 真实框坐标 [M, 4]
# detections[:, :4] → 预测框坐标 [N, 4]
# iou → shape [M, N]，表示每个真实框和预测框的 IoU
    iou = box_iou(labels[:, 1:], detections[:, :4])
    # 判断类别是否匹配
    # labels[:, 0:1] → 真实类别 [M, 1]
# detections[:, 5] → 预测类别 [N]
# correct_class → shape [M, N]，True 表示类别一致
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        # IoU > threshold and classes match
        # 找出 IoU ≥ 阈值 且 类别一致 的配对
        # x 是二元索引 (label_index, detection_index)
        x = torch.where((iou >= iouv[i]) & correct_class)
        # x[0] 是 ground truth 的索引数组，x[0].shape[0] 就是匹配的数量。
# 如果有匹配（数量 > 0），才继续执行
        if x[0].shape[0]:
            # torch.stack(x, 1) 按第列堆叠
            # x[0] → 真实框索引 [num_matches]
            # x[1] → 预测框索引 [num_matches]得到 [匹配的真实框id, 预测框id]
            # iou 是一个 IoU 矩阵，形状是 [num_gts, num_preds]。
            # iou[x[0], x[1]] 会按匹配索引取出对应 IoU 值。
            # [:, None] 把它变成列向量，例如 (N, ) → (N, 1)。
            # 拼接 [[gt_idx, pred_idx], iou]，形成一个 (N, 3) 的 Tensor。
            # 每一行就是 [gt_idx, pred_idx, iou]
            # cat里的1表示按列拼接
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu(
            ).numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                # [::-1]
                # 将升序结果 翻转，变成降序。
                matches = matches[matches[:, 2].argsort()[::-1]]
                # np.unique 默认会返回排序后的唯一值，并且如果你传 return_index=True，它会告诉你每个唯一值在原数组中第一次出现的位置
                # 每个预测框只保留 IoU 最大的
                # 目的：一对一匹配
                # 预测框唯一
                matches = matches[np.unique(
                    matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                # 真实框唯一
                matches = matches[np.unique(
                    matches[:, 0], return_index=True)[1]]
            # matches[:, 1] → 预测框 id
# 在 correct[预测框, iou阈值] 置 True
            correct[matches[:, 1].astype(int), i] = True
    # 把结果转回 Torch tensor（布尔型）
# 大小 [N, 10]，每行对应一个预测框，每列对应一个 IoU 阈值
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


'''======================1.设置参数====================='''


@smart_inference_mode()
def run(
        data,  # 数据集配置文件地址 包含数据集的路径、类别个数、类名、下载地址等信息 train.py时传入data_dict
        # model.pt path(s)
        weights=None,  # 模型的权重文件地址 运行train.py=None 运行test.py=默认weights/yolov5s
        batch_size=32,  # batch size # 前向传播的批次大小 运行test.py传入默认32 运行train.py则传入batch_size // WORLD_SIZE * 2
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold# object置信度阈值 默认0.001
        iou_thres=0.6,  # NMS IoU threshold进行NMS时IOU的阈值 默认0.6
        max_det=300,  # maximum detections per image
        task='val',  # train, val, test, speed or study设置测试的类型 有train, val, test, speed or study几种 默认val
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference# 测试时增强
        verbose=False,  # verbose output是否打印出每个类别的mAP 运行test.py传入默认Fasle 运行train.py则传入nc < 50 and final_epoch
        save_txt=False,  # save results to *.txt是否以txt文件的形式保存模型预测框的坐标 默认True
        # save label+prediction hybrid results to 是否保存预测每个目标的置信度到预测txt文件中 默认True*.txt
        save_hybrid=False,
        save_conf=False,  # save confidences in --save-txt labels保存置信度
        # save a COCO-JSON results file是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签）,
        save_json=False,
                      # 运行test.py传入默认Fasle 运行train.py则传入is_coco and final_epoch(一般也是False)
        project=ROOT / 'runs/val',  # save to project/name验证结果保存的根目录 默认是 runs/va
        name='exp',  # save to project/name验证结果保存的目录 默认是exp  最终: runs/val/exp
        # existing project/name ok, do not increment如果文件存在就increment name，不存在就新建  默认False(默认文件都是不存在的)
        exist_ok=False,
        half=True,  # use FP16 half-precision inference使用 FP16 的半精度推理
        dnn=False,  # use OpenCV DNN for ONNX inference在 ONNX 推理时使用 OpenCV DNN 后段端
        # 如果执行val.py就为None 如果执行train.py就会传入( model=attempt_load(f, device).half() )
        model=None,
        dataloader=None,  # 数据加载器 如果执行val.py就为None 如果执行train.py就会传入testloader
        # 件保存路径 如果执行val.py就为‘’ , 如果执行train.py就会传入save_dir(runs/train/expn
        save_dir=Path(''),
        plots=True,  # 是否可视化 运行val.py传入，默认True
        callbacks=Callbacks(),  # 回调函数
        # 损失函数 运行val.py传入默认None 运行train.py则传入compute_loss(train)
        compute_loss=None,
):
    '''======================2.初始化/加载模型以及设置设备====================='''
    # Initialize/load model and set device
    # 判断当前脚本是不是在 训练流程里被调用。
# train.py 调用 val.run(model=...) 时会传入一个 model，此时 training=True。
# 如果是用户直接运行 val.py（比如 python val.py --weights yolov5s.pt），那么 model=None，training=False。
    training = model is not None
    if training:  # called by train.py
        # get model device, PyTorch model
        # 获得记录在模型中的设备 next为迭代器
        # 直接从已有模型获取它所在的设备（CPU/GPU）
        # 这里记录模型格式（pt 表示 PyTorch 模型，jit 和 engine 用于 TorchScript 或 TensorR
        device, pt, jit, engine = next(
            model.parameters()).device, True, False, False
        # half = half & (device.type != 'cpu')
        # device.type 是当前计算设备的类型，比如 'cuda' 或 'cpu'。
        # 如果 half=True，就执行 model.half()，否则转成 float32
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        # 如果是 直接运行 val.py
        # 调用 select_device() 来选择计算设备（如 "0" 表示第0块GPU，"cpu" 表示CPU）。
        # 这里还会传入 batch_size，因为某些设备可能会有限制
        device = select_device(device, batch_size=batch_size)

        # Directories
        # 设置结果保存目录
        # Path(project) / name 就是 runs/val/exp 这种路径。
# increment_path() 会自动给目录编号，比如 exp, exp2, exp3，防止覆盖之前的结果
        save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok)  # increment run
# 如果设置了 save_txt=True，说明要保存预测的标签（txt 格式），就会创建 save_dir/labels/。
# 否则就只创建 save_dir/。
# mkdir(parents=True, exist_ok=True) 确保父目录也会被递归创建，不会报错。
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                              exist_ok=True)  # make dir

        # Load model
        # 加载模型，DetectMultiBackend 是 YOLOv5 封装的一个类，支持多种推理后端
        # weights：权重文件（如 yolov5s.pt, .onnx, .engine 等）。
# device：运行设备（CPU/GPU）。
# dnn：是否启用 OpenCV DNN 后端（运行 .onnx 模型时用）。
# data：数据集配置（如 coco.yaml）。
# fp16：是否使用半精度（只在 CUDA 下生效）。
        model = DetectMultiBackend(
            weights, device=device, dnn=dnn, data=data, fp16=half)
        # model.pt：是否是 PyTorch 格式模型（.pt）。
# model.jit：是否是 TorchScript 格式。
# model.engine：是否是 TensorRT 引擎。
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        # 确保输入图片大小 imgsz 可以被 stride 整除
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            # 如果模型是 TensorRT 引擎（engine=True）：直接使用引擎自带的 batch_size
            batch_size = model.batch_size
        else:
            # 把运行设备同步为 model.device
            device = model.device
            # 如果模型不是 PyTorch（pt=False）也不是 TorchScript（jit=False），比如是 ONNX、CoreML 模型，那么强制设置 batch_size=1
            # 这是因为非 PyTorch 模型一般不支持动态 batch 推理，YOLOv5 会自动限制 batch=1
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(
                    f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        # 检查并加载数据集配置
        # check_dataset(data) 会解析 YAML 文件，返回一个 dict，里面包含：
# train: 训练集路径
# val: 验证集路径
# nc: 类别数量
# names: 类别名列表
        data = check_dataset(data)  # check

    '''======================3.加载配置====================='''
    # Configure
    # 切换到 评估模式
    model.eval()
    # 判断是否在 GPU 上运行
    cuda = device.type != 'cpu'
    # 检查当前验证集是不是 COCO 官方的 val2017 验证集。
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(
        f'coco{os.sep}val2017.txt')  # COCO dataset
    # 确定类别数量 nc
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # iou vector for mAP@0.5:0.95
    # torch.linspace(0.5, 0.95, 10) 生成 [0.50, 0.55, 0.60, ..., 0.95] 共 10 个阈值。
# 这是 COCO 标准的评估指标：mAP@0.5:0.95。
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    # IoU 阈值个数（这里是 10）
    niou = iouv.numel()

    '''======================4.加载val数据集====================='''
    # Dataloader
    # 直接运行 val.py 的情况
    # 在这种情况下，需要自己加载数据、做一些检查。
    if not training:
        # 如果权重是 PyTorch 格式 (pt=True) 且 不是单类训练 (single_cls=False)，就检查权重和数据集类别数是否一致
        if pt and not single_cls:  # check --weights are trained on --data
            # model.model.nc：权重文件里保存的类别数。
            # nc：从数据集配置读取的类别数。
            ncm = model.model.nc
            # 如果不一致，就 assert 报错：提示你用错了数据集配置和权重。
            assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        # 模型预热：先跑一次假数据，避免首次推理时因为 CUDA 图编译或内存分配而变慢。
        # 输入形状是 (batch, 3, imgsz, imgsz)
        model.warmup(imgsz=(1 if pt else batch_size,
                     3, imgsz, imgsz))  # warmup
        # 设置 数据加载方式：
# 如果任务是 speed（只测推理速度，不做真实评估）：
# pad=0.0 → 不做 padding
# rect=False → 不用矩形推理，统一缩放到正方形
# 否则（正常验证）：
# pad=0.5 → padding 填充 50% 保持长宽比
# rect=pt → 只有 PyTorch 模型才支持矩形推理
# 👉 矩形推理 (rect=True)：在目标检测里保持长宽比缩放输入，减少信息丢失，提高精度
        pad, rect = (0.0, False) if task == 'speed' else (
            0.5, pt)  # square inference for benchmarks
        # path to train/val/test images
        # 确保 task 在 ('train', 'val', 'test') 三者之一
        task = task if task in ('train', 'val', 'test') else 'val'
        # 创建验证集的 DataLoader
        dataloader = create_dataloader(data[task],  # data[task]：数据集路径，比如 coco/val2017.txt。
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       # prefix：打印时的前缀，比如 "val:"。
                                       prefix=colorstr(f'{task}: '))[0]

    '''======================5.初始化====================='''
    # 初始化已完成测试的图片数量
    seen = 0
    # 初始化一个 混淆矩阵
    # 后面会根据预测框和真实框的匹配情况更新这个矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 有些模型存在 model.names（常见于 YOLOv5 保存的 .pt 权重）。
# 如果是多 GPU 模式（nn.DataParallel），名字可能在 model.module.names 里
    names = model.names if hasattr(
        model, 'names') else model.module.names  # get class names
    # 兼容旧格式：
# 以前 names 可能是列表，比如 ["person", "car", ...]。
# 这里转成字典：{0: "person", 1: "car", ...}。
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    # 如果验证集是 COCO（is_coco=True），则需要把 YOLOv5 的 80 类（coco80）映射到 COCO 官方的 91 类（coco91）。
# 否则就生成一个 [0, 1, 2, ..., 999] 的映射表（最多 1000 类）
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # 构建验证结果的表头字符串：
# Class：类别名
# Images：验证的图片数量
# Instances：标注的目标数量
# P：Precision 精确率
# R：Recall 召回率
# mAP50：mAP@0.5
# mAP50-95：mAP@[0.5:0.95]
# 👉 在 tqdm 进度条上会用到这个表头
    s = ('%22s' + '%11s' * 6) % ('Class', 'Images',
                                 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    # 初始化各种评估指标：
# tp, fp：正负样本数
# p, r：Precision、Recall
# f1：F1 score
# mp, mr：mean precision, mean recall
# map50：mAP@0.5
# ap50：每类 AP@0.5
# map：mAP@[0.5:0.95]
    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # 初始化 性能分析器，用于统计时间开销：
# 第 1 个 Profile：数据预处理时间
# 第 2 个 Profile：模型推理时间
# 第 3 个 Profile：NMS（非极大值抑制）时间
    dt = Profile(), Profile(), Profile()  # profiling times
    # 初始化损失值 [box_loss, obj_loss, cls_loss]
    # 虽然是验证阶段，但 YOLOv5 仍然会计算 loss，用来评估模型质量
    loss = torch.zeros(3, device=device)
    # 初始化几个统计用变量：
# jdict：存储预测结果（用于 COCO JSON 格式评估）。
# stats：存储 (correct, conf, pcls, tcls) 信息，计算 PR 曲线用。
# ap：每个类别的 AP。
# ap_class：对应的类别 ID。
    jdict, stats, ap, ap_class = [], [], [], []
    # 触发回调函数（Hook）。
    callbacks.run('on_val_start')
    # 使用 tqdm 进度条包装 dataloader。
# desc=s：进度条前面会显示表头。
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    '''===6.1 开始验证前的预处理==='''
    # 遍历验证集 DataLoader：
# batch_i：当前 batch 的索引（从 0 开始）。
# im：图片张量，形状 (batch_size, 3, H, W)。
# targets：真实标注框，格式是 [image_index, class, x, y, w, h]（相对坐标）。
# paths：当前 batch 的图片路径列表。
# shapes：原始图片的大小（以及缩放比例、padding 信息）。
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        # 在每个 batch 推理前运行回调函数（比如日志、调试 hook）
        callbacks.run('on_val_batch_start')
        # dt[0] 是 Profile 计时器，这里统计 数据预处理阶段耗时。
        with dt[0]:
            if cuda:
                # 把图片 im 和标注 targets 移动到 GPU。
                # non_blocking=True：异步拷贝，提高数据加载速度
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
            # DataLoader 读入的图像通常是 uint8（0-255 的整型）
            im = im.half() if half else im.float()  # uint8 to fp16/32
            # 将像素值归一化到 [0, 1] 区间。
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 获取当前 batch 的图片信息：
# nb：batch size
# _：通道数（一般是 3）
# height：输入图像高度
# width：输入图像宽度
            nb, _, height, width = im.shape  # batch size, channels, height, width

        '''===6.2 前向推理==='''
        # Inference
        # dt[1] 专门用来统计 推理时间
        with dt[1]:
            # compute_loss 表示是否需要计算 loss（一般是 验证时训练态，比如 --save-json 或者 --task train 时会打开）
            # preds → 最终预测结果（经过解码的边框 + 置信度 + 分类结果）。
            # train_out → 训练用的输出（未解码的 raw 输出，用于计算 loss，比如 P3, P4, P5 特征层的卷积输出
            # 如果不需要计算 loss
            # preds = model(im, augment=augment)
            # （这里可能带 测试时增强 (TTA)：翻转、缩放等数据增强后再推理，最后融合结果。）
            # train_out = None（因为不需要 loss）
            preds, train_out = model(im) if compute_loss else (
                model(im, augment=augment), None)

        # Loss
        if compute_loss:
            # 调用 compute_loss(train_out, targets)，计算预测结果和真实标注的损失。
            # 这个函数返回一个元组 (total_loss, loss_items)：
            # total_loss → 整个 batch 的总损失。
            # loss_items → 各部分的损失（box loss、obj loss、cls loss）。
            # [1] 取的是 loss_items。
            # 所以这里 loss += ... 是把三个部分的损失（box、obj、cls）累加起来，便于后续日志统计
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        '''===6.4 NMS获得预测框==='''
        # NMS
        # to pixels
        # 将标注框从 归一化坐标 转换为 像素坐标。
# YOLOv5 的标注格式是 [image_idx, class, x_center, y_center, width, height]，归一化在 [0,1]。
# width, height 是当前 batch 图像的尺寸，用它乘以归一化坐标就得到实际像素值。
        targets[:, 2:] *= torch.tensor((width,
                                       height, width, height), device=device)
        # 如果开启 save_hybrid（通常用于自动标注/增强数据集）：
# 按 batch 中每张图分组，取 [class, x, y, w, h] 作为标签。
# targets[:, 0] == i → 筛选第 i 张图的标签
        lb = [targets[targets[:, 0] == i, 1:]
              for i in range(nb)] if save_hybrid else []  # for autolabelling
        # dt[2] 是第三个 Profile 对象，用于统计 NMS (Non-Max Suppression) 时间。
# 后面 non_max_suppression() 会在这里计时。
        with dt[2]:
            # preds → 模型预测输出，格式 [batch, xyxy, conf, cls]。
            # conf_thres → 置信度阈值，低于这个概率的框会被丢弃。
            # iou_thres → IoU 阈值，高于此值的框会被合并/删除。
            # labels=lb → 如果有 save_hybrid，会把真实标签也加入 NMS 处理，用于自动标注。
            # multi_label=True → 同一个框可以属于多个类别（罕见情况）。
            # agnostic=single_cls → 是否类别无关 NMS，单类时为 True。
            # max_det=max_det → 每张图片最多保留的检测框数。
            # 输出 preds 是 每张图像的 NMS 后预测框列表
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        '''===6.5 统计真实框、预测框信息==='''
        # Metrics、# 遍历当前 batch 中每张图像的预测结果：
# si → 当前图片在 batch 中的索引（0 到 nb-1）。
# pred → 当前图片经过 NMS 后的预测框 [x1, y1, x2, y2, conf, cls]。
        for si, pred in enumerate(preds):
            # 获取当前图片的真实标注框：
            # targets[:, 0] == si → 筛选 batch 中第 si 张图的标注。
            # [1:] → 取 [class, x, y, w, h]（像素坐标）
            labels = targets[targets[:, 0] == si, 1:]
            # number of labels, predictions
            # nl → 当前图片真实框数量。
# npr →当前图片预测框数量（NMS 后）。
            nl, npr = labels.shape[0], pred.shape[0]
            # 获取当前图片的路径和尺寸（缩放后的 shape，用于可视化或调试）
            # paths 是一个 图片路径列表，保存了当前 batch 中所有图片的文件路径。paths = [
            '/data/coco/images/val2017/000000000139.jpg',
            '/data/coco/images/val2017/000000000285.jpg',
            # paths[si]代表当前 batch 中的 第几个样本的索引
            # Path 来自 Python 标准库 pathlib，用来把字符串路径转换成 Path 对象，方便后续操作
            # 每张图片在加载时都会保存一个对应的 shapes 信息。
# 它是一个二维数组或元组，记录了该图片的：
# 原始尺寸 (h, w)
# 缩放比例和填充信息（用于从 letterbox 输入图像还原到原图尺寸时使用）
# 如shapes[si] = (
#     (h0, w0),   # 图片原始高度、宽度
#     ((ratio, ratio), (dw, dh))  # 缩放比例 & padding 偏移量
# )
# shapes[si][0]
# 只取第一个部分 (h0, w0)，也就是当前图片的 原始尺寸。
            path, shape = Path(paths[si]), shapes[si][0]
            # 初始化布尔张量 correct：用于标记每个预测框在不同 IoU 阈值下是否匹配真实框。
# 形状 (npr, niou)：
# npr → 当前图片预测框数
# niou → IoU 阈值数量（如 [0.5,0.55,...,0.95]，共 10 个）
# 初始值都是 False（还没匹配）
            correct = torch.zeros(
                npr, niou, dtype=torch.bool, device=device)  # init
            # 累计处理过的图片数量
            seen += 1
            # 如果 预测框为 0（模型没有预测任何目标）
            if npr == 0:
                # 如果真实框数量 nl>0
                if nl:
                    # 在 stats 中追加一条记录，表示这张图片的预测为空：
                    # correct → 所有预测都为 False
                    # torch.zeros((2, 0)) → 占位，表示置信度和预测类别为空
                    # labels[:, 0] → 真实类别
                    stats.append(
                        (correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        # 如果绘制混淆矩阵 (plots=True)，也会更新：
                        # 调用 confusion_matrix.process_batch()，没有检测框（detections=None），只提供真实类别
                        confusion_matrix.process_batch(
                            detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                # 如果你只训练/推理 单类别检测，就把所有预测框的类别 class_id 强制设为 0。
                pred[:, 5] = 0
            # 复制一份预测结果，避免后续修改 pred 本身。
            predn = pred.clone()
            # im[si].shape[1:]im[si] 是 dataloader 给模型的输入图片（通常是 缩放+padding 之后的 letterbox 图像）。
# .shape 返回 (height, width, channels)（如果是 numpy）或者 (channels, height, width)（如果是 tensor）。
# .shape[1:] 取的是 (height, width)，代表当前推理图像的大小。
# predn[:, :4]
# 取预测框的前四列 [x1, y1, x2, y2]，即边界框坐标。
# 注意这些坐标还是在 推理图像空间（可能被缩放过）。
# shape
# 原始图像的 (height, width)。
# 这是 dataloader 在读取图片时保存的，用来做坐标还原。
# shapes[si][1]
# shapes 里保存了 letterbox 时的缩放比例和 padding 信息。
# shapes[si] = (original_shape, (ratio, pad))，其中：
# ratio：输入图像相对原始图像的缩放因子。
# pad：上下/左右的填充像素。
# shapes[si][1] 就是 (ratio, pad)，用于把预测框从 letterbox 坐标映射回原图。
            scale_boxes(im[si].shape[1:], predn[:, :4], shape,
                        shapes[si][1])  # native-space pred

            # Evaluate
            # 把标签和预测框处理成统一格式，然后送入 process_batch 去算 mAP 等指标
            # 标签数量存在
            if nl:

                # labels[:, 1:5] 取出 [x, y, w, h] 部分。
                # xywh2xyxy 转成[x1, y1, x2, y2](左上角和右下角)，这样才能和预测框统一比较。
                tbox = xywh2xyxy(labels[:, 1:5])  # type: ignore # target boxes
                # 把标签框 tbox 从网络输入尺寸 (640×640) 映射回 原图尺寸，保证和 predn（预测框映射回原图后的结果）在同一坐标系。
                # labels[:, 0:1]：类别 ID。
# tbox：现在已经是 原图空间的 xyxy 坐标。
# 拼接后 labelsn 变成 [cls, x1, y1, x2, y2]
                scale_boxes(im[si].shape[1:], tbox, shape,
                            shapes[si][1])  # native-space labels
                # native-space labels
                # labels[:, 0:1]：类别 ID。
# tbox：现在已经是 原图空间的 xyxy 坐标。
# 拼接后 labelsn 变成 [cls, x1, y1, x2, y2]。
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                # predn：预测框 [x1, y1, x2, y2, conf, cls]，同样是在原图空间。
# labelsn：目标框 [cls, x1, y1, x2, y2]。
# iouv：IoU 阈值列表 (例如 [0.5, 0.55, …, 0.95])。
# process_batch 负责算：预测框和标签框的匹配关系（是否 TP / FP），返回布尔矩阵 correct。
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    # 更新混淆矩阵
                    confusion_matrix.process_batch(predn, labelsn)
            # (correct, conf, pcls, tcls)
            # correct：每个预测框在不同 IoU 阈值下是否匹配成功 (TP / FP)。
# pred[:, 4]：预测框的置信度 (confidence)。
# pred[:, 5]：预测框的类别 ID。
# labels[:, 0]：真实框的类别 ID。
# 这些信息被存到 stats 里，最后用来计算 P, R, mAP 等指标
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

            # Save/log
            if save_txt:
                # 保存预测信息到txt文件
                save_one_txt(predn, save_conf, shape,
                             file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                # append to COCO-JSON dictionary
                # 保存预测信息到json字典
                save_one_json(predn, jdict, path, class_map)
            # YOLOv5 在 val 过程中提供了 callback 钩子，方便外部插件或自定义逻辑在每张验证图像结束时执行。
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        '''===6.6 画出前三个batch图片的gt和pred框==='''
        # Plot images
        if plots and batch_i < 3:
            # plot_images：会把图片拼接成一个大图，并在图上绘制框\
            # 展示真实标签
            plot_images(im, targets, paths, save_dir /
                        f'val_batch{batch_i}_labels.jpg', names)  # labels
            # output_to_target(preds) 的作用是把 preds 转换为targets 格式一致
            # 展示预测框 (preds，先转成 target 格式)
            plot_images(im, output_to_target(preds), paths, save_dir /
                        f'val_batch{batch_i}_pred.jpg', names)  # pred
        # on_val_batch_end 表示 每个验证 batch 结束时触发
        callbacks.run('on_val_batch_end', batch_i, im,
                      targets, paths, shapes, preds)

    '''===6.7 计算指标==='''
    # Compute metrics
    # stats 是一个 list of tuples，每个 tuple 对应一个 batch，包含
    # correct: TP/FP 布尔矩阵
# pred[:,4]: 置信度
# pred[:,5]: 预测类别
# labels[:,0]: 真实类别
# zip(*stats)：把每个 batch 的同类数据分组，例如把所有 batch 的 correct 聚在一起
# torch.cat(x, 0)：按第 0 维拼接成一个大 tensor
# .cpu().numpy()：转成 numpy 数组方便后续计算
# stats[0] = all_correct
# stats[1] = all_conf
# stats[2] = all_pred_cls
# stats[3] = all_true_cls
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # 确保验证集里至少有一个预测框，否则跳过指标计算
    if len(stats) and stats[0].any():
        # ap_per_class 核心逻辑：
        # 按 置信度排序预测框
        # 逐个预测框匹配真实框，计算 TP/FP
        # 计算 Precision（P）和 Recall（R）
        # 画 PR 曲线
        # 积分得到 AP@0.5 和 AP@0.5:0.95
        # 返回每个类别的指标：
        # tp, fp：每个预测框对应的 TP/FP
        # p, r：每类 Precision/Recall
        # f1：每类 F1 分数
        # ap：每类 AP（多 IoU 阈值平均）
        # ap_class：类别 ID
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats, plot=plots, save_dir=save_dir, names=names)
        # ap 形状 [num_classes, num_iou_thresholds]
# ap[:,0] → IoU=0.5 时的 AP（AP@0.5）
# ap.mean(1) → 对 0.5~0.95 IoU 平均（AP@0.5:0.95）
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
#         mp → 平均 Precision（所有类别）
# mr → 平均 Recall
# map50 → 所有类别的 AP@0.5 平均
# map → 所有类别的 AP@0.5:0.95 平均
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    # number of targets per class
    # stats[3] = 所有真实标签的类别 ID
# np.bincount(..., minlength=nc) → 返回一个长度为 nc 的数组，每个元素是该类别真实目标数
# 用于在报告里显示每个类别有多少目标。
    nt = np.bincount(stats[3].astype(int), minlength=nc)

    '''===6.8 打印日志==='''
    # Print results
    # '%22s' → 占 22 个字符打印字符串（这里用于类名或 'all'）
# '%11i' * 2 → 两个整数，每个占 11 个字符
# '%11.3g' * 4 → 四个浮点数，每个占 11 个字符，保留 3 位有效数字
# 最终格式对应打印列
# Class | Images | Instances | P | R | mAP50 | mAP50-95
    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
    # 打印整体指标
    # 'all' → 表示统计所有类别的总体指标
# seen → 遍历过的图片数量
# nt.sum() → 验证集总目标数（所有类别加起来）
# mp → 平均 Precision
# mr → 平均 Recall
# map50 → mAP@0.5
# map → mAP@0.5:0.95
# ⚡ 作用：把验证集的整体性能指标输出到日志
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    if nt.sum() == 0:
        # 如果验证集中没有真实标签 (nt.sum() == 0)：
        # mAP / P / R 不能计算
        # 输出警告，提醒用户数据集可能配置错误或为空
        LOGGER.warning(
            f'WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels')

    # Print results per class
    # 打印每类指标
    # verbose → 用户指定详细输出
# nc < 50 and not training → 如果类别数少于 50 且是验证/测试模式，也打印每类指标
# nc > 1 → 确保有多类别
# len(stats) → 统计信息存在
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        # 遍历每个有 AP 的类别 ap_class
        # names[c] → 类别名称
        # seen → 图片总数
        # nt[c] → 该类别真实目标数量
        # p[i], r[i], ap50[i], ap[i] → 该类别的精度、召回率、mAP@0.5、mAP@0.5:0.95
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c],
                        p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    # 打印速度信息
    # dt 是三个 Profile 对象：
# dt[0] → 数据预处理耗时
# dt[1] → 推理耗时（模型 forward）
# dt[2] → NMS 耗时
# x.t / seen * 1E3 → 每张图的平均耗时（ms）
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        # 输出示例：Speed: 3.2ms pre-process, 12.5ms inference, 1.1ms NMS per image at shape (16, 3, 640, 640)

        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    '''===6.9 保存验证结果==='''
    # Plots
    if plots:
        # 绘制混淆矩阵
        #         # confusion_matrix → 在每个 batch 处理预测时已经累积了 TP/FP/类别匹配信息
        # plot(save_dir, names)：
        # 绘制混淆矩阵（类别预测 vs 真实标签）
        # 横轴：预测类别
        # 纵轴：真实类别
        # 颜色深浅表示数量多少
        # 保存路径：save_dir/confusion_matrix.png
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        # 触发验证结束回调
        callbacks.run('on_val_end', nt, tp, fp, p, r, f1,
                      ap, ap50, ap_class, confusion_matrix)

    # Save JSON
    # save_json=True → 用户希望把预测结果保存为 COCO 格式 JSON
# len(jdict) → jdict 不为空，即有预测结果
    if save_json and len(jdict):
        # weights 可能是列表或单个文件
        # Path(...).stem → 取文件名（不含路径和扩展名）
        # 用于生成预测 JSON 文件名
        # 验证时，通常只需要 一组权重生成一份预测 JSON，用于计算 mAP
        # 如果有多个权重：
        # 可以逐个权重跑一次验证
        # 每次生成独立的 JSON 文件
        # 所以这里默认取第一个权重文件作为输出文件名（best_predictions.json），避免覆盖或混乱
        w = Path(weights[0] if isinstance(
            weights, list) else weights).stem if weights is not None else ''  # weights
        # annotations
        # COCO 数据集标注路径
        # 指向 COCO 验证集的 真实标注文件
# pycocotools 用它来计算 mAP
        anno_json = str(
            Path('../datasets/coco/annotations/instances_val2017.json'))
        # 保存预测结果的 JSON 文件路径
        # JSON 格式是 COCO 风格：[{"image_id": 1, "category_id": 3, "bbox": [x, y, w, h], "score": 0.987}, ...]
        pred_json = str(save_dir / f'{w}_predictions.json')  # predictions
        # 提示用户正在使用 pycocotools 计算 mAP
# 并且会把预测结果保存到 pred_json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        '''用 pycocotools 官方评估 COCO mAP 的部分，它和 YOLOv5 内部自己算的 ap_per_class() 是两套平行的指标计算方式'''
        # 保存预测结果
        # jdict 前面已经收集了所有图片的预测框（转成 COCO JSON 格式），比如：[
#   {"image_id": 42, "category_id": 1, "bbox": [x, y, w, h], "score": 0.98},
#   {"image_id": 42, "category_id": 3, "bbox": [x, y, w, h], "score": 0.85},
#   ...
# ]这里把它写入 pred_json 文件，后续交给 pycocotools 来评估。
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            # 加载 COCO API
            # 确认 pycocotools 已安装
            # COCO 是官方的标注数据加载类
            # COCOeval 是官方的评估类，用来计算 mAP、Recall 等指标
            check_requirements('pycocotools>=2.0.6')
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            # 初始化标注和预测
            # 加载真值标注 (比如 COCO val2017.json)
            # anno_json 指向真实的标注文件
            anno = COCO(anno_json)  # init annotations api
            # 加载刚保存的预测 JSON
            # pred_json 是我们刚生成的预测结果
            pred = anno.loadRes(pred_json)  # init predictions api
            # 设定评估参数
            # COCOeval 第三个参数 'bbox' 表示评估 目标检测框（还可以评估 segmentation、keypoints 等）
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                # image IDs to evaluate
                # dataloader.dataset 表示它所加载的数据集对象。
                # im_files 是这个数据集中存储的 图片文件路径列表
                # 把文件名（不带扩展名的部分）转换成整数。
                # 比如 '000001' → 1，'000123' → 123。
                # iouType 可选值
                # 'segm' → 评估 实例分割 (segmentation masks)
                # 'bbox' → 评估 目标检测框 (bounding boxes)
                # 'keypoints' → 评估 关键点检测 (human pose / landmark keypoints)
                eval.params.imgIds = [int(Path(x).stem)
                                      for x in dataloader.dataset.im_files]
            # 执行评估
            #  匹配预测框和真值框
            eval.evaluate()
            # 累积各 IoU 阈值下的结果
            eval.accumulate()
            # 打印最终 mAP/Recall 等指标
            # 这里会打印标准的 12 个指标，比如
            # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.365
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
            eval.summarize()
            # update results (mAP@0.5:0.95, mAP@0.5)
            # eval.stats 是 pycocotools 计算的指标数组
# eval.stats[0] = mAP@0.5:0.95
# eval.stats[1] = mAP@0.5
# 赋值给 YOLOv5 内部变量，和之前 ap_per_class() 结果对齐
            map, map50 = eval.stats[:2]
        # 异常处理
        except Exception as e:
            # 如果用户没装 pycocotools，或者数据集不是 COCO 格式，就会报错，这里直接忽略。
            LOGGER.info(f'pycocotools unable to run: {e}')

    '''===6.10 返回结果==='''
    # Return results
    #  将模型转换为适用于训练的状态
    model.float()  # for training
    if not training:
        #  如果不是训练过程则将结果保存到对应的路径
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # 在控制台中打印保存结果
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # 初始化一个长度为类别数的数组，默认都填充 map（整体 mAP）
    maps = np.zeros(nc) + map
    # ap_class：哪些类别在验证集中实际出现过。
# ap[i]：类别 c 的 AP 值。
# 所以这一步会用 各类别 AP 覆盖 maps，保证 maps[c] 代表类别 c 的 AP，其余类别还是整体 mAP
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    # mp：mean precision
# mr：mean recall
# map50：mAP@0.5
# map：mAP@0.5:0.95
# loss：验证集上的平均 loss
# maps：每个类别的 AP 数组（长度 = nc）
# t：计时信息（数据时间、推理时间、NMS 时间等）
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


'''===============================================五、设置opt参数==================================================='''


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT /
                        'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--batch-size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size',
                        type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300,
                        help='maximum detections per image')
    # 设置测试的类型 有train, val, test, speed or study几种 默认val
    parser.add_argument('--task', default='val',
                        help='train, val, test, speed or study')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true',
                        help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    # 是否打印出每个类别的mAP 默认False
    parser.add_argument('--verbose', action='store_true',
                        help='report mAP by class')
    # 是否以txt文件的形式保存模型预测的框坐标, 默认False
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # 保存label+prediction杂交结果到对应.txt，默认False
    parser.add_argument('--save-hybrid', action='store_true',
                        help='save label+prediction hybrid results to *.txt')
    # 保存置信度
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估（需要同样coco的json格式的标签） 默认Fals
    parser.add_argument('--save-json', action='store_true',
                        help='save a COCO-JSON results file')
    # 测试保存的源文件 默认runs/val
    parser.add_argument('--project', default=ROOT /
                        'runs/val', help='save to project/name')
    # 测试保存的文件地址 默认exp  保存在runs/val/exp下
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    # 是否使用 OpenCV DNN对ONNX 模型推理
    parser.add_argument('--dnn', action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    # |或 左右两个变量有一个为True 左边变量就为True
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


'''==============================六、执行main（）函数======================================'''
def main(opt):
    # 检查运行 YOLOv5 所需的 Python 库是否齐全
    # 检查环境依赖，排除 tensorboard 和 thop
    check_requirements(exclude=('tensorboard', 'thop'))
    # 如果任务是训练、验证、测试
    if opt.task in ('train', 'val', 'test'):  # run normally
        # 1. 提醒置信度阈值过高的问题
        # 官方经验是：在验证/测试时，conf_thres 必须设置得很低（≤0.001），否则 mAP 结果会不准确（因为很多低置信度框没算进去）
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(
                f'WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results')
        # 2. 提醒保存混合标签会导致 mAP 虚高
        # --save-hybrid 参数会把 预测框和 GT 标签混合在一起保存，主要用于半监督学习。
# 但如果用它来验证，会导致 mAP 人为升高，所以这里有个警告
        if opt.save_hybrid:
            LOGGER.info(
                'WARNING ⚠️ --save-hybrid will return high mAP from hybrid labels, not from predictions alone')
        # 3. 调用 run(**vars(opt)) 进入具体任务逻辑
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(
            opt.weights, list) else [opt.weights]
        # FP16 for fastest results
        opt.half = torch.cuda.is_available() and opt.device != 'cpu'
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                # filename to save to
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'
                # x axis (image sizes), y axis
                x, y = list(range(256, 1536 + 128, 128)), []
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            subprocess.run(['zip', '-r', 'study.zip', 'study_*.txt'])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(
                f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
