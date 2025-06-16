# -*- coding:utf-8 -*-
"""
@file name  : resnet_ptq.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-09-25
@brief      : 肺炎Xray图像分类模型，resnet50 PTQ 量化
评估未量化前精度：
python resnet_ptq.py --mode evaluate
执行PTQ量化，并保存模型
python resnet_ptq.py --mode quantize --ptq-method max --num-data 512
python resnet_ptq.py --mode quantize --ptq-method entropy --num-data 512
python resnet_ptq.py --mode quantize --ptq-method mse --num-data 512
python resnet_ptq.py --mode quantize --ptq-method percentile --num-data 512

支持4种方法：max entropy mse percentile
https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html
"""
# quant_nn：量化神经网络层
from pytorch_quantization import nn as quant_nn
# 量化模块注册
from pytorch_quantization import quant_modules
# 量化校准工具，用于收集激活统计信息
from pytorch_quantization import calib
# tqdm：显示循环进度条，方便训练或推理时观察进度。
from tqdm import tqdm


import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import matplotlib

matplotlib.use('Agg')

import utils.my_utils as utils
from datasets.pneumonia_dataset import PneumoniaDataset
# PTQ（Post-Training Quantization，训练后量化

def collect_stats(model, data_loader, num_batches):
    """
    前向传播，获得统计数据，并进行量化
    :param model:
    :param data_loader:
    :param num_batches:
    :return:
    """
    # Enable calibrators
    # 遍历模型中的所有模块
    for name, module in model.named_modules():
        # 如果该模块是 quant_nn.TensorQuantizer 类型（即量化器模块）
        if isinstance(module, quant_nn.TensorQuantizer):
            # 如果它有校准器（_calibrator 不为 None）
            if module._calibrator is not None:
                # 关闭量化（使其不量化输入）
                module.disable_quant()
                # 打开统计模式，收集输入的数值范围、直方图等
                module.enable_calib()
            else:
                # 直接禁用该量化模块
                module.disable()

    # Feed data to the network for collecting stats
    # 遍历 data_loader 中的图像：
# 把图像传入模型执行前向传播（转到 cuda）
# 不关心标签 _，只需要图像输入用于统计。
# 执行 num_batches 个 batch 之后就停止（可以防止跑完整个数据集，提高效率）。
# 🔎 目的是：喂入一定数量的数据，让所有 TensorQuantizer 收集激活值分布信息。
# total=len(data_loader) 是传给 tqdm() 的参数，用于指定进度条的总长度（也就是总共有多少个 batch）
# tqdm() 是 Python 中常用的进度条库，默认情况下它会尝试自动估算总进度。但有时它无法准确获取 data_loader 的长度，或者估算不准，显示效果不好。
# 加上 total=len(data_loader) 可以明确告诉 tqdm：
# “这个 data_loader 总共有多少个 batch（步骤）。”
# 这样可以让 tqdm 正确地显示：
# 当前进度（已经处理了几个 batch）
# 百分比（完成了多少 %）
# 预计剩余时间（ETA）
# 每个 batch 的平均耗时
    for i, (image, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    # 关闭校准器、开启量化
    # 在统计完之后，重新配置 TensorQuantizer：
# 如果有 calibrator：
# 开启量化（之后就会使用 int8 等低精度计算）。
# 关闭校准（不再收集数据）。
# 否则直接启用模块。
# 🔎 目的是：模型已经完成了校准统计，现在进入“使用量化”的状态。
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    """
    根据统计值，计算amax，确定上限、下限。用于后续计算scale和Z值
    :param model:
    :param kwargs:
    :return:
    """
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            # 如果使用的是 `MaxCalibrator`（即统计每层激活的最大绝对值作为 `amax`）：
            # - 直接加载统计到的最大值作为 `amax`。
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    # 加载 MaxCalibrator 统计的最大值为 amax
                    module.load_calib_amax()
                else:
                    # 否则使用的是 HistogramCalibrator 或其他更高级的校准器（如百分位数）：
                    # 可能需要参数，比如 percentile=99.99。
                    # 调用 load_calib_amax(**kwargs)，自动根据直方图计算 amax，如取 top 99% 范围等
                    module.load_calib_amax(**kwargs)
    # 将模型放回 CUDA 上，准备后续继续训练、评估或导出 INT8 模型。
    model.cuda()

# 命令行参数解析器
def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument("--data-path", default=r"G:\deep_learning_data\chest_xray", type=str, help="dataset path")
    parser.add_argument("--ckpt-path", default=r"./Result/2023-09-26_01-47-40/checkpoint_best.pth", type=str, help="ckpt path")
    parser.add_argument("--model", default="resnet50", type=str,
                        help="model name; resnet50/convnext/convnext-tiny")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    # 支持：`quantize`（量化）、`evaluate`（评估）、`onnxexport`（导出 ONNX） |
    parser.add_argument("--mode", default="quantize", type=str, help="quantize\evaluate\onnxexport")
    # 量化校准阶段使用的 batch 数量，影响统计精度
    parser.add_argument("--num-data", default=512, type=int, help="量化校准数据batch数量")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    # 后训练量化方法，可选：`max`、`mse`、`entropy`、`percentile`（根据不同校准器） |
    parser.add_argument("--ptq-method", type=str, help="method for ptq; max; mse; entropy; percentile")

    return parser


def get_dataloader(args):
    data_dir = args.data_path
    normMean = [0.5]
    normStd = [0.5]
    input_size = (224, 224)
    normTransform = transforms.Normalize(normMean, normStd)
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(input_size, padding=4),
        transforms.ToTensor(),
        normTransform
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normTransform
    ])

    # chest_xray.zip 解压，获得 chest_xray/train, chest_xray/test
    # 数据可从 https://data.mendeley.com/datasets/rscbjbr9sj/2 下载
    # 构建数据集对象
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'test')
    train_set = PneumoniaDataset(train_dir, transform=train_transform)
    valid_set = PneumoniaDataset(valid_dir, transform=valid_transform)
    # 构建 DataLoader
    # 构建DataLoder
    train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_set, batch_size=8, num_workers=2)
    return train_loader, valid_loader


def get_model(args, logger, device):
    if args.model == 'resnet50':
        model = torchvision.models.resnet50()
    elif args.model == 'convnext':
        model = torchvision.models.convnext_base()
    elif args.model == 'convnext-tiny':
        model = torchvision.models.convnext_tiny()
    else:
        logger.error('unexpect model --> :{}'.format(args.model))

    model_name = model._get_name()

    if 'ResNet' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features  # 替换最后一层
        model.fc = nn.Linear(num_ftrs, 2)
    elif 'ConvNeXt' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        num_kernel = 128 if args.model == 'convnext' else 96
        model.features[0][0] = nn.Conv2d(1, num_kernel, (4, 4), stride=(4, 4))  # convnext base/ tiny
        # 替换最后一层
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 2)

    # ------------------------- 加载训练权重
    state_dict = torch.load(args.ckpt_path)
    model_sate_dict = state_dict['model_state_dict']
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    model.to(device)
    return model

# 后训练量化（PTQ, Post-Training Quantization）流程的主入口函数，主要完成：
# 读取数据 + 加载模型 → 校准统计 + 计算量化参数 → 评估精度 → 保存模型 + 导出 ONNX 模型
def ptq(args):
    """
    进行PTQ量化，并且保存模型
    :param args:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------------------ step1: dataset ------------------------------------
    train_loader, valid_loader = get_dataloader(args)
    # ------------------------------------ tep2: model ------------------------------------
    model = get_model(args, logger, device)
    # ------------------------------------ step3: 前向推理校准、量化 ------------------------------------
    with torch.no_grad():
        # 统计激活数据
        # 打开 TensorQuantizer 的 calibrator 开关
        # 运行训练集前向传播（不用反向传播），收集各层激活最大值或直方图等
        collect_stats(model, train_loader, num_batches=args.num_data)  # 设置量化模块开关，并推理，同时统计激活值
        # 计算量化 scale/zero_point（也就是 amax）
        # max：取最大值为量化上限
# mse：最小化误差
# entropy：KL散度
# percentile：排除最极端的激活（如 0.1%）
        if args.ptq_method == 'percentile':
            compute_amax(model, method='percentile', percentile=99.9)  # 计算上限、下限，并计算scale 、Z值
        else:
            compute_amax(model, method=args.ptq_method)                     # 计算上限、下限，并计算scale 、Z值
        logger.info('PTQ 量化完成')
    # ------------------------------------ step4: 评估量化后精度  ------------------------------------
    classes = ["NORMAL", "PNEUMONIA"]
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    # 用交叉熵评估验证集
# 结果包括 loss、准确率、混淆矩阵
    loss_m_valid, acc_m_valid, mat_valid = utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)
    logger.info('PTQ量化后模型ACC :{}，scale值计算方法是:{}'.format(acc_m_valid.avg, args.ptq_method))
    # ------------------------------------ step5: 保存ptq量化后模型 ------------------------------------
    dir_name = os.path.dirname(args.ckpt_path)
    ptq_ckpt_path = os.path.join(dir_name, "resnet50_ptq.pth")
    torch.save(model.state_dict(), ptq_ckpt_path)

    # 导出ONNX
    # 启用 fake quant 模式（模拟量化行为，方便导出 ONNX）
    # 作用：启用 Fake Quantization（假量化） 模式。
# ⚙️ 启用后，TensorQuantizer 不再做实际的离散化操作，而是使用一个模拟浮点行为的“假”量化节点。
# 📌 目的：
# 在导出 ONNX 时保留量化行为（scale、zero_point 模拟）
# ONNX 会包含 FakeQuant 节点，让部署引擎（如 TensorRT）知道这里需要量化
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    # 导出两个不同 batch size 的模型（1 和 32），方便部署时选择对应输入大小（比如推理时批处理 or 单张图像）
    for bs in [1, 32]:
        # 构造模型文件名
        # bs32: batch size
        # data-num512: 量化使用了多少 batch 校准数据
        # entropy: 量化方法
        # 92.15%: 量化后在验证集上的精度
        model_name = "resnet_50_ptq_bs{}_data-num{}_{}_{:.2%}.onnx".format(bs, args.num_data, args.ptq_method, acc_m_valid.avg / 100)
        # 使用训练 ckpt 文件所在目录作为 ONNX 的保存路径。
        onnx_path = os.path.join(dir_name, model_name)
        # 构造一个假的输入张量
# 大小为 [bs, 1, 224, 224]
# 1 是通道数（X-ray 为灰度图）
# 224x224 是模型输入尺寸
        dummy_input = torch.randn(bs, 1, 224, 224, device='cuda')
        # 导出模型为 ONNX 格式
#         # model	你已经量化好的 PyTorch 模型
# dummy_input	模拟输入张量，用于追踪计算图
# onnx_path	保存 ONNX 模型的路径
# opset_version=13	ONNX 版本（13 比较稳定且支持量化 op）
# do_constant_folding=False	是否提前折叠常量计算，量化模型建议关掉
# input_names	ONNX 中的输入名字（便于部署时使用）
# output_names	ONNX 中的输出名字
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=13, do_constant_folding=False,
                          input_names=['input'],  output_names=['output'])


def evaluate(args):
    """
    评估量化前模型精度
    :param args:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = args.output_dir
    # ------------------------------------  log ------------------------------------
    logger, log_dir = utils.make_logger(result_dir)
    # ------------------------------------ step1: dataset ------------------------------------
    train_loader, valid_loader = get_dataloader(args)
    # ------------------------------------ tep2: model ------------------------------------
    model = get_model(args, logger, device)
    # ------------------------------------ step3: evaluate ------------------------------------
    classes = ["NORMAL", "PNEUMONIA"]
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    loss_m_valid, acc_m_valid, mat_valid =\
        utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)

    logger.info('PTQ量化前模型ACC :{}'.format(acc_m_valid.avg))


def pre_t_model_export(args):
    """
    导出fp32的onnx模型，用于效率对比
    :param args:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, logger, device)
    dir_name = os.path.dirname(args.ckpt_path)

    for bs in [1, 32]:
        model_name = "resnet_50_fp32_bs{}.onnx".format(bs)
        onnx_path = os.path.join(dir_name, model_name)
        dummy_input = torch.randn(bs, 1, 224, 224, device='cuda')
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=13, do_constant_folding=False,
                          input_names=['input'],  output_names=['output'])
        print('模型保存完成: {}'.format(onnx_path))

def main(args):
    if args.mode == 'quantize':
        quant_modules.initialize()  # 替换torch.nn的常用层，变为可量化的层
        ptq(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'onnxexport':
        pre_t_model_export(args)
    else:
        print("args.mode is not recognize! got :{}".format(args.mode))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    result_dir = args.output_dir
    logger, log_dir = utils.make_logger(result_dir)

    # 不指定某一种ptq_method，则进行四种量化方法的对比实验
    if args.ptq_method:
        main(args)
    else:
        ptq_method_list = "max entropy mse percentile".split()
        for ptq_method in ptq_method_list:
            args.ptq_method = ptq_method
            main(args)


