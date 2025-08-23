# -*- coding:utf-8 -*-
"""
@file name  : train_script.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-02-04
@brief      : 肺炎Xray图像分类训练脚本
"""
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
from code.chapter8.classification_01.datasets.pneumonia_dataset import PneumoniaDataset
import os
# os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
# print("NO_ALBUMENTATIONS_UPDATE =", os.getenv("NO_ALBUMENTATIONS_UPDATE"))
import torchvision
import torch
import torch.nn as nn
# Albumentations 是一个功能强大的开源图像增强库，常用于计算机视觉中的数据增强。
import albumentations as A
import matplotlib
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
# 从 Albumentations 库的 PyTorch 工具模块中导入 ToTensorV2，用于将图像转换为 PyTorch 的 tensor 格式
from albumentations.pytorch import ToTensorV2
matplotlib.use('Agg')
import utils.my_utils as utils
# 加载和预处理肺炎图像数据

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument(
        "--data-path", default=r"bigdata\chapter-8\1\ChestXRay2017\chest_xray", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet50", type=str, help="model name; resnet50 or convnext or convnext-tiny")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--random-seed", default=42, type=int, help="random seed")
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-step-size", default=20, type=int, help="decrease lr every step-size epochs")
    # 每次学习率下降时，乘以这个系数。
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    # 控制每隔多少个 batch 打印一次训练状态（如 loss、accuracy）
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./Result", type=str, help="path to save outputs")
    # 是否从之前保存的 checkpoint 继续训练。
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    #  是否使用 **AutoAugment** 自动增强策略。
# - 布尔类型，传入时为 `True`，不传为 `False`
    parser.add_argument('--autoaug', action='store_true', default=False, help='use torchvision autoaugment')
    # 不加这个参数时：args.useplateau 默认为 False，使用普通的 StepLR 调度器。
    # ReduceLROnPlateau	根据 val loss/acc 停滞情况自动降 lr
    parser.add_argument('--useplateau', action='store_true',
                        default=False, help='use ReduceLROnPlateau scheduler')


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

    if args.autoaug:
        # 使用 torchvision.transforms.AutoAugment 自动数据增强方法。
        # 并指定策略为 IMAGENET
        # 适用于 ImageNet 数据集或类似的大型图像数据集，包含一组变换操作组合，如旋转、颜色变换、剪切、翻转等。
        auto_aug_list = torchvision.transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
        train_transform = transforms.Compose([
            # 适用于 ImageNet 数据集或类似的大型图像数据集，包含一组变换操作组合，如旋转、颜色变换、剪切、翻转等。
            auto_aug_list,
            transforms.Resize(256),
            transforms.RandomCrop(input_size, padding=4),
            # 把 PIL 或 NumPy 图像转换为 torch.Tensor，并自动缩放到 [0, 1]
            transforms.ToTensor(),
            # 	用 ImageNet 均值和标准差对图像每个通道做标准化处理
            transforms.Normalize(normMean, normStd),
            # ToTensor()和Normalize顺序不能换
            #             Normalize 作用在 PIL.Image 上时会直接报错或无效，因为它期望的是 tensor。
            # 即使没报错，也不会除以 255，导致输入范围不对，标准化结果完全错乱。
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(normMean, normStd),
        ])
    else:
        # 使用 Albumentations 库定义的数据增强流水线，用于训练集（train_transform）和验证集（valid_transform）。相比于 torchvision.transforms，Albumentations 更强大、更灵活、运行更快，特别适合复杂的数据增强任务。
        train_transform = A.Compose([
            # 	随机缩放图像，高宽比例在 ±20% 内变化，有 50% 概率应用
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.Resize(256, 256),
            A.RandomCrop(224, 224),  # Randomly shift
            # 随机旋转，角度在 [-30, 30] 之间，有 50% 概率应用
            A.Rotate(limit=30, p=0.5),
            # 水平翻转图像，有 50% 概率应用
            A.HorizontalFlip(p=0.5),
            # 像素标准化：除以 255 之后再减去均值、除以标准差
            A.Normalize(normMean, normStd, max_pixel_value=255.),  # mean, std， 基于0-1，像素值要求0-255，并通过max_pixel_value=255，来实现整体数据变换为0-1
            # 将图像转换为 PyTorch tensor，不会自动除以 255，因为已经在上一步处理了
            ToTensorV2(),  # 仅数据转换，不会除以255
        ])

        valid_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(normMean, normStd, max_pixel_value=255.),  # mean, std， 基于0-1，像素值要求0-255，并通过max_pixel_value=255，来实现整体数据变换为0-1
            ToTensorV2(),  # 仅数据转换，不会除以255
        ])

    # chest_xray.zip 解压，获得 chest_xray/train, chest_xray/test
    # 数据可从 https://data.mendeley.com/datasets/rscbjbr9sj/2 下载
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'test')
    train_set = PneumoniaDataset(train_dir, transform=train_transform)
    valid_set = PneumoniaDataset(valid_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(dataset=valid_set, batch_size=8, num_workers=args.workers)

    # ------------------------------------ tep2: model ------------------------------------
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(weights=None)
    elif args.model == 'convnext':
        model = torchvision.models.convnext_base(weights=None)
    elif args.model == 'convnext-tiny':
        model = torchvision.models.convnext_tiny(weights=None)
    else:
        logger.error(f'unexpect model --> :{args.model}')
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

    model.to(device)

    # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 选择优化器
    # 学习率调度器
    if args.useplateau:
        # 监控某个指标（默认为验证集准确率或损失）
        # 当指标在 patience=10 个 epoch 内没有提升时，将学习率乘以 factor=0.2 缩小。
        # cooldown=5 表示降学习率后冷却 5 个 epoch 不再调整。
        # mode='max' 表示监控指标越大越好（比如准确率）。
        # 典型用法是：在验证集准确率不再提升时自动降低学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=10, cooldown=5, mode='max')
    else:
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)  # 设置学习率下降策略

    # ------------------------------------ step4: iteration ------------------------------------
    # 初始化记录最佳验证准确率和对应的epoch，用于后续保存最优模型或早停
    best_acc, best_epoch = 0, 0
    logger.info(args)
    # logger.info(train_loader, valid_loader)
    logger.info("Start training")
    start_time = time.time()
    # 创建一个AverageMeter对象用于统计每个epoch的耗时（假设AverageMeter是用于计算平均值和当前值的工具类）。
    epoch_time_m = utils.AverageMeter()
    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        loss_m_train, acc_m_train, mat_train = \
            utils.ModelTrainer.train_one_epoch(
                train_loader, model, criterion, optimizer, scheduler, epoch, device, args, logger, classes)
        # 验证
        loss_m_valid, acc_m_valid, mat_valid = \
            utils.ModelTrainer.evaluate(valid_loader, model, criterion, device, classes)

        epoch_time_m.update(time.time() - end)
        end = time.time()
        # 如果使用的是 ReduceLROnPlateau（args.useplateau=True），该调度器没有 get_last_lr() 方法，因此直接通过调度器绑定的优化器（scheduler.optimizer）读取当前学习率。
        lr_current = scheduler.optimizer.param_groups[0]['lr'] if args.useplateau else scheduler.get_last_lr()[0]
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'Valid Loss avg: {loss_valid.avg:>6.4f}  '
            'Train Acc@1 avg:  {top1_train.avg:>7.4f}   '
            'Valid Acc@1 avg: {top1_valid.avg:>7.4f}    '
            'LR: {lr}'.format(
                epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train, loss_valid=loss_m_valid,
                top1_train=acc_m_train, top1_valid=acc_m_valid, lr=lr_current))

        # 学习率更新
        if args.useplateau:
            # 需要把“监控指标”传给 .step()，此处是验证集准确率的平均值 acc_m_valid.avg。
            # ReduceLROnPlateau 会根据这个指标是否有提升来决定是否降低学习率。
            scheduler.step(acc_m_valid.avg)
        else:
            scheduler.step()
        # 记录
        writer.add_scalars('Loss_group', {
                           'train_loss': loss_m_train.avg, 'valid_loss': loss_m_valid.avg}, epoch)
        writer.add_scalars('Accuracy_group', {
                           'train_acc': acc_m_train.avg, 'valid_acc': acc_m_valid.avg}, epoch)
        # verbose=epoch == args.epochs - 1：如果是最后一个epoch才打印详细信息。
        # save=True：保存混淆矩阵图片文件。
        conf_mat_figure_train = utils.show_conf_mat(
            mat_train, classes, "train", log_dir, epoch=epoch, verbose=epoch == args.epochs - 1, save=True)
        conf_mat_figure_valid = utils.show_conf_mat(
            mat_valid, classes, "valid", log_dir, epoch=epoch, verbose=epoch == args.epochs - 1, save=True)
        writer.add_figure('confusion_matrix_train', conf_mat_figure_train, global_step=epoch)
        writer.add_figure('confusion_matrix_valid', conf_mat_figure_valid, global_step=epoch)
        writer.add_scalar('learning rate', lr_current, epoch)

        # ------------------------------------ 模型保存 ------------------------------------
        if best_acc < acc_m_valid.avg or epoch == args.epochs - 1:
            best_epoch = epoch if best_acc < acc_m_valid.avg else best_epoch
            best_acc = acc_m_valid.avg if best_acc < acc_m_valid.avg else best_acc
            # model_state_dict: 模型的所有权重参数。
# optimizer_state_dict: 优化器的状态（动量、历史梯度等），方便断点续训。
# lr_scheduler_state_dict: 学习率调度器的状态。
# epoch: 当前 epoch，方便恢复。
# args: 所有训练超参数。
# best_acc: 当前记录的最佳验证准确率。
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "best_acc": best_acc}
            pkl_name = "checkpoint_{}.pth".format(epoch) if epoch == args.epochs - 1 else "checkpoint_best.pth"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)
            logger.info(f'save ckpt done! best acc:{best_acc}, epoch:{epoch}')

    total_time = time.time() - start_time
    # ✅ total_time
# 是一个浮点数（单位是秒），比如 total_time = 3789.54 表示训练耗时 3789 秒。
# ✅ int(total_time)
# 转为整数秒：int(3789.54) → 3789（丢掉小数部分）。
# ✅ datetime.timedelta(seconds=...)
# 用 Python 的 datetime 库来创建一个 时间间隔对象（timedelta）。
# datetime.timedelta(seconds=3789) 就会变成一个 表示“1小时3分钟9秒” 的对象。
# ✅ str(...)
# 把 timedelta 对象转为字符串，如"1:03:09"
    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


classes = ["NORMAL", "PNEUMONIA"]


if __name__ == "__main__":
    # 加载 .env 文件
    load_dotenv()
    print("NO_ALBUMENTATIONS_UPDATE =", os.getenv("NO_ALBUMENTATIONS_UPDATE"))
    args = get_args_parser().parse_args()
    utils.setup_seed(args.random_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("开始：", datetime.now())
    main(args)
    print("结束：", datetime.now())





