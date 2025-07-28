# -*- coding:utf-8 -*-
"""
@file name  : resnet50_qat.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-02-04
@brief      : 肺炎Xray图像分类模型，resnet50 QAT 量化
"""
from datasets.pneumonia_dataset import PneumoniaDataset
import utils.my_utils as utils
import os
import time
import datetime
import torchvision
import torch
import torch.nn as nn
import matplotlib
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

matplotlib.use('Agg')


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument(
        "--data-path", default=r"G:\deep_learning_data\chest_xray", type=str, help="dataset path")
    parser.add_argument(
        "--ckpt-path", default=r"./Result/2023-09-26_01-47-40/checkpoint_best.pth", type=str, help="ckpt path")
    parser.add_argument("--model", default="resnet50", type=str,
                        help="model name; resnet50/convnext/convnext-tiny")
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=5, type=int,
                        metavar="N", help="number of total epochs to run")
    # DataLoader 读取数据的线程数
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")
    # 优化器
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    # 设置随机种子，保证结果可复现
    parser.add_argument("--random-seed", default=42,
                        type=int, help="random seed")
    parser.add_argument("--lr", default=0.01/100,
                        type=float, help="initial learning rate")
    # SGD 动量参数
    parser.add_argument("--momentum", default=0.9,
                        type=float, metavar="M", help="momentum")
    # 权重衰减系数（L2正则化）
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",)
    # 日志打印频率（每几轮 epoch 打印一次）
    parser.add_argument("--print-freq", default=20,
                        type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./Result",
                        type=str, help="path to save outputs")
    # 指定训练从第几轮开始（用于断点恢复）
    parser.add_argument("--start-epoch", default=0, type=int,
                        metavar="N", help="start epoch")

    return parser


def main(args):
    device = args.device
    data_dir = args.data_path
    result_dir = args.output_dir
    # ------------------------------------  log ------------------------------------
    logger, log_dir = utils.make_logger(result_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # ------------------------------------ step1: dataset ------------------------------------

    normMean = [0.5]
    normStd = [0.5]
    input_size = (224, 224)
    normTransform = transforms.Normalize(normMean, normStd)
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
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'test')
    train_set = PneumoniaDataset(train_dir, transform=train_transform)
    valid_set = PneumoniaDataset(valid_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=8, num_workers=args.workers)

    # ------------------------------------ tep2: model ------------------------------------
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif args.model == 'convnext':
        model = torchvision.models.convnext_base(pretrained=True)
    elif args.model == 'convnext-tiny':
        model = torchvision.models.convnext_tiny(pretrained=True)
    else:
        logger.error('unexpect model --> :{}'.format(args.model))

    model_name = model._get_name()

    if 'ResNet' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(
            2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features  # 替换最后一层
        model.fc = nn.Linear(num_ftrs, 2)
    elif 'ConvNeXt' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        num_kernel = 128 if args.model == 'convnext' else 96
        model.features[0][0] = nn.Conv2d(
            1, num_kernel, (4, 4), stride=(4, 4))  # convnext base/ tiny
        # 替换最后一层
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 2)

    # ------------------------- 加载训练权重
    state_dict = torch.load(args.ckpt_path)
    model_sate_dict = state_dict['model_state_dict']
    model.load_state_dict(model_sate_dict)  # 模型参数加载

    model.to(device)

    # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    # 使用 随机梯度下降（SGD）+ 动量项（momentum）+ 权重衰减（L2正则化） 来优化模型参数。
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)  # 选择优化器
    # 用 余弦退火（Cosine Annealing）策略动态调整学习率，在训练过程中逐渐降低。
    # T_max 是总轮数（周期长度），eta_min 是最小学习率。
    # 学习率像余弦曲线一样从高到低平滑下降，这样前期快速学习、后期精细调整，提高收敛质量。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr/100)  # 设置学习率下降策略

    # ------------------------------------ step4: iteration ------------------------------------
    logger.info(args)
    logger.info("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        loss_m_train, acc_m_train, mat_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                                               epoch, device, args, logger, classes)
        # 验证
        loss_m_valid, acc_m_valid, mat_valid = \
            utils.ModelTrainer.evaluate(
                valid_loader, model, criterion, device, classes)
        # 获取当前学习率（以列表形式返回，取第一个）
        # 有几个参数组就有几个学习率，这里只取第一个参数组的学习率
        lr_current = scheduler.get_last_lr()[0]
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'Valid Loss avg: {loss_valid.avg:>6.4f}  '
            'Train Acc@1 avg:  {top1_train.avg:>7.4f}   '
            'Valid Acc@1 avg: {top1_valid.avg:>7.4f}    '
            'LR: {lr}'.format(
                epoch, args.epochs, loss_train=loss_m_train, loss_valid=loss_m_valid,
                top1_train=acc_m_train, top1_valid=acc_m_valid, lr=lr_current))

        # 学习率调用
        # 每调用 step() 满 step_size 次更新
        scheduler.step()
        # 记录
        conf_mat_figure_train = utils.show_conf_mat(mat_train, classes, "train", log_dir, epoch=epoch,
                                                    verbose=epoch == args.epochs - 1, save=True)
        conf_mat_figure_valid = utils.show_conf_mat(mat_valid, classes, "valid", log_dir, epoch=epoch,
                                                    verbose=epoch == args.epochs - 1, save=True)
        writer.add_figure('confusion_matrix_train',
                          conf_mat_figure_train, global_step=epoch)
        writer.add_figure('confusion_matrix_valid',
                          conf_mat_figure_valid, global_step=epoch)
        writer.add_scalar('learning rate', lr_current, epoch)

    # ------------------------------------ 训练完毕模型保存 ------------------------------------
    # 这段代码的目的是将 QAT（Quantization-Aware Training）后的模型导出为 ONNX 格式，以便进行推理部署
    # Fake Quantization 是一种模拟量化的方式，它在训练时模拟量化误差，但权重仍然是 float32。
# 设置这个值可以确保导出 ONNX 时使用 fake quant 算子，而不是真实的 int8 值。
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    for bs in [1, 32]:
        model_name = "resnet_50_qat_bs{}_{:.2%}.onnx".format(
            bs, acc_m_valid.avg / 100)
        onnx_path = os.path.join(log_dir, model_name)
        dummy_input = torch.randn(bs, 1, 224, 224, device='cuda')
        # do_constant_folding=False：是否在导出时进行常量折叠优化（此处关闭，可能是为了保持 fake quant ops）
        torch.onnx.export(model, dummy_input, onnx_path, opset_version=13, do_constant_folding=False,
                          input_names=['input'], output_names=['output'])


classes = ["NORMAL", "PNEUMONIA"]


if __name__ == "__main__":
    #     quant_modules 一般来自 from pytorch_quantization import quant_modules
    # 这行代码会将 PyTorch 中的部分标准层（如 nn.Conv2d, nn.Linear, nn.ReLU 等）替换为它们的可量化版本，如 QuantConv2d、QuantLinear。
    # 替换后，你原始的模型结构可以直接加入量化节点而无需手动替换每个子模块。
    # 📌 目的：为 QAT（Quantization-Aware Training）做好模型结构的准备。
    # 只影响 后续构建 的模型；已构建模型不受影响
    # 恢复，只需调用 quant_modules.deactivate()。
    quant_modules.initialize()  # 替换torch.nn的常用层，变为可量化的层

    args = get_args_parser().parse_args()
    utils.setup_seed(args.random_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
