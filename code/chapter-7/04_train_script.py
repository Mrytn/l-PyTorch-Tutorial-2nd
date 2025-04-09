# -*- coding:utf-8 -*-
"""
@file name  : 04_train_script.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-25
@brief      : 分类任务训练脚本
"""
import os
import time
import datetime
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import my_utils as utils
# argparse 模块的主要作用是让你可以方便地为 Python 脚本添加命令行参数，并且自动处理参数的解析和验证。用户可以通过命令行灵活地指定脚本运行所需的各种参数，而不需要修改代码中的硬编码值。同时，argparse 还会自动生成帮助信息，方便用户了解脚本的使用方法和各个参数的作用。
# argparse 是 Python 标准库中用于解析命令行参数和选项的模块。它能够让你方便地编写用户友好的命令行接口。程序定义它需要的参数，然后 argparse 将负责如何从 sys.argv 中解析出这些参数。
def get_args_parser(add_help=True):
    import argparse
    # argparse.ArgumentParser 是 argparse 模块的核心类，用于创建一个参数解析器对象。
# description 参数是一个字符串，用于在显示帮助信息时对程序的功能进行简要描述。
# add_help 参数是一个布尔值，用于指定是否添加默认的 -h 或 --help 选项，当用户在命令行输入这个选项时，会显示程序的使用说明和所有参数的帮助信息
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    # add_argument 方法用于向解析器中添加一个命令行参数。
# --data-path 是参数的长选项名，用户在命令行中可以使用 --data-path <value> 的形式来指定该参数的值。
# default 参数指定了该参数的默认值，当用户在命令行中没有指定该参数时，将使用这个默认值。
# type 参数指定了该参数的类型，这里是 str 表示字符串类型。
# help 参数是一个字符串，用于在显示帮助信息时对该参数的作用进行简要描述。
    parser.add_argument(
        "--data-path", default=r"data\datasets\cifar10-office", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet8", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
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
    parser.add_argument("--lr-step-size", default=80, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=80, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="Result",
                        type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
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
    normMean = [0.4948052, 0.48568845, 0.44682974]
    normStd = [0.24580306, 0.24236229, 0.2603115]
    normTransform = transforms.Normalize(normMean, normStd)
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normTransform
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    # root变量下需要存放cifar-10-python.tar.gz 文件
    # cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
    # 对图像数据进行标准化处理。
    # 时，download=True参数会触发下载动作。如果data_dir下不存在cifar-10-python.tar.gz文件，它会从指定链接https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz下载该文件 。下载完成后，torchvision.datasets.CIFAR10会自动解压这个压缩包，并按照数据集的结构组织文件，方便后续读取数据。
# 不手动解压也能读取数据原理：torchvision.datasets.CIFAR10类内部实现了处理压缩文件的逻辑。即使文件未解压，它也能在需要时从压缩包内读取数据。这是因为 Python 的压缩文件处理库（如tarfile）支持直接从压缩包中读取文件内容，torchvision.datasets.CIFAR10类利用了这一特性。在数据加载阶段，它会根据需求从压缩包内提取相应的数据批次，转化为模型训练和测试可用的格式，无需用户提前手动解压。

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, transform=valid_transform, download=True)
    # 构建DataLoder
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, num_workers=args.workers)
    # ------------------------------------ tep2: model ------------------------------------
    # 调用 utils.resnet8() 函数创建一个 ResNet - 8 模型。
# 使用 model.to(device) 将模型移动到指定的设备上进行训练。
    model = utils.resnet8()
    model.to(device)
    # ------------------------------------ step3: optimizer, lr scheduler ------------------------------------
    criterion = nn.CrossEntropyLoss()  # 选择损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 选择优化器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)  # 设置学习率下降策略
    # ------------------------------------ step4: iteration ------------------------------------
    # 初始化最佳准确率和最佳 epoch。
    best_acc, best_epoch = 0, 0
    logger.info(args)
    logger.info(train_loader, valid_loader)
    logger.info("Start training")
    start_time = time.time()
    epoch_time_m = utils.AverageMeter()
    end = time.time()
    # 开始训练循环，每个 epoch 依次进行训练和验证
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        # 一个 epoch 的训练，返回训练损失、准确率和混淆矩阵。
        loss_m_train, acc_m_train, mat_train = \
            utils.ModelTrainer.train_one_epoch(train_loader, model, criterion, optimizer, scheduler,
                                               epoch, device, args, logger, classes)
        # 验证
        loss_m_valid, acc_m_valid, mat_valid = \
            utils.ModelTrainer.evaluate(
                valid_loader, model, criterion, device, classes)
        epoch_time_m.update(time.time() - end)
        end = time.time()
        # 记录每个 epoch 的训练时间，并使用日志记录器输出训练信息。
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'Valid Loss avg: {loss_valid.avg:>6.4f}  '
            'Train Acc@1 avg:  {top1_train.avg:>7.4f}   '
            'Valid Acc@1 avg: {top1_valid.avg:>7.4f}    '
            'LR: {lr}'.format(
                epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train, loss_valid=loss_m_valid,
                top1_train=acc_m_train, top1_valid=acc_m_valid, lr=scheduler.get_last_lr()[0]))
        # 学习率更新
        scheduler.step()
        # 记录
        writer.add_scalars('Loss_group', {'train_loss': loss_m_train.avg,
                                          'valid_loss': loss_m_valid.avg}, epoch)
        writer.add_scalars('Accuracy_group', {'train_acc': acc_m_train.avg,
                                              'valid_acc': acc_m_valid.avg}, epoch)
        conf_mat_figure_train = utils.show_conf_mat(mat_train, classes, "train", log_dir, epoch=epoch,
                                        verbose=epoch == args.epochs - 1, save=False)
        conf_mat_figure_valid = utils.show_conf_mat(mat_valid, classes, "valid", log_dir, epoch=epoch,
                                        verbose=epoch == args.epochs - 1, save=False)
        writer.add_figure('confusion_matrix_train', conf_mat_figure_train, global_step=epoch)
        writer.add_figure('confusion_matrix_valid', conf_mat_figure_valid, global_step=epoch)
        writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)
        # ------------------------------------ 模型保存 ------------------------------------
        # 模型保存
        # 虽然第一轮验证时acc_m_valid.avg很可能大于 0，但如果后续轮次中出现更高的验证准确率，之前保存的模型就会被覆盖
        if best_acc < acc_m_valid.avg or epoch == args.epochs - 1:
            best_epoch = epoch if best_acc < acc_m_valid.avg else best_epoch
            best_acc = acc_m_valid.avg if best_acc < acc_m_valid.avg else best_acc
            # checkpoint 包含模型的状态字典、优化器的状态字典、学习率调度器的状态字典、当前 epoch、训练参数和最佳准确率
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
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    utils.setup_seed(args.random_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
