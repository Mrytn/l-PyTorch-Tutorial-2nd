# -*- coding:utf-8 -*-
"""
@file name  : 03_utils.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-14
@brief      : 训练所需的函数
"""
import random
import numpy as np
import os
import time

import torchmetrics
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from datetime import datetime
import logging
import matplotlib
# 它是设置 matplotlib 图像中的字体为黑体（SimHei），尤其用于：
# 支持中文显示：防止 matplotlib 画图时中文字符显示成乱码或方框（□ □ □）
# 统一字体样式：所有图像默认使用 'SimHei' 字体
matplotlib.rcParams['font.family'] = 'SimHei'


class LeNet5(nn.Module):
    def __init__(self):
        # super(LeNet5, self).__init__()	调用父类 nn.Module 的构造方法，完成模块注册
        # 可简写成super().__init__()
        super(LeNet5, self).__init__()
        # in_channels	3	输入通道数，比如 RGB 图像是 3 个通道（R, G, B）
# out_channels	6	卷积核个数（输出特征图的通道数），表示提取出 6 个特征
# kernel_size	5	卷积核大小为 5×5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # kernel_size	2	池化窗口为 2×2
# stride	2	每次移动 2 个像素（不重叠）
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _weights_init(m):
    classname = m.__class__.__name__  # 获取层的类名（例如：Linear、Conv2d）
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)  # 用 kaiming 正态分布初始化权重


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 表示你定义了一个 空的顺序模块（nn.Sequential），它 什么都不做，只是将输入原封不动返回。
        # 这是在构造 残差块（Residual Block） 时的一种特殊情况，表示 shortcut 路径不需要变换输入张量。
        self.shortcut = nn.Sequential()
    # self.expansion = 1这时，残差连接（shortcut）不能直接 + 原始输入和主路径输出，因为维度不一样。

# 所以我们就需要把输入变换一下，使其维度和主路径输出对齐。这行代码就是用于这种情况的一种简化实现。
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # 这是对通道维度进行padding（补0），让通道数变得和主路径对齐。
                # pad=(pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_c_front, pad_c_back)
# (0, 0, 0, 0, planes//4, planes//4)：
# 宽度方向不填充 → pad_w_left = 0, pad_w_right = 0
# 高度方向不填充 → pad_h_top = 0, pad_h_bottom = 0
# 通道维度前后各填充 planes//4 个通道 → 通道增加 planes//2
# 那为什么补 planes//4 就能让通道数匹配？
# CIFAR ResNet 原论文里，planes 总是 in_planes * 2 的形式

# 比如从 16→32，或 32→64，planes 总是 in_planes * 2

# planes // 4 = in_planes * 2 // 4 = in_planes // 2
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            # 作用：用1×1卷积改变输入特征图的通道数和（可能的）空间尺寸。
# 参数含义：
# in_planes：输入通道数。
# self.expansion * planes：输出通道数。self.expansion 是一个放大系数（比如BasicBlock里为1，Bottleneck里为4），乘以planes确定最终输出通道。
# kernel_size=1：1×1卷积，只改变通道数，不影响感受野。
# stride=stride：步长，可以是1或2，如果是2就会对空间尺寸做下采样（宽高减半）。
# bias=False：一般后面接 BatchNorm，卷积层不用偏置。
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1)  # 原版16
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2)  # 原版32
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2)  # 原版64
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        # strides = [2] + [1] * (3 - 1)
        #  = [2] + [1, 1]
        #  = [2, 1, 1]
        # [1] * (num_blocks - 1) 表示后续所有残差块的步长都设为1，即不改变特征图的尺寸。
        # 第1个残差块：stride=2 → 做下采样（空间尺寸减半）
        # 第2个残差块：stride=1 → 保持尺寸
        # 第3个残差块：stride=1 → 保持尺寸
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        #  后面两个 block 的作用：提取更深层的特征
# 虽然它们不再改变空间尺寸（H×W），但它们仍然执行：
# 卷积操作（比如 3×3 conv）
# 非线性激活（ReLU）
# 批归一化（BatchNorm）
# 残差连接（skip connection）
# 这些操作可以让网络学到：
# 更丰富的通道信息（通道维度）
# 更深的抽象表达
# 增强非线性能力
# 避免梯度消失（靠残差连接）
        for stride in strides:
            # block：残差块类型
            # planes：该层残差块中间层的通道数（不一定是最终输出通道数，视block而定）
            # num_blocks：残差块的数量（每层堆叠多少个block）
            # stride：第一个block是否做下采样（空间尺寸缩小）
            # 第一层输出16通道，第二层32通道，第三层64通道。
            # 对于BasicBlock，输出通道一般等于 planes。
            # 对于Bottleneck，输出通道是 planes * 4（扩展系数）
            # # planes 表示的是：
            # 每个 block 中间的“基准通道数”，最终的 输出通道数为 planes × block.expansion。
            # 所以：
            # planes 是中间的基础通道数；
            # in_planes 是输入通道数（从外部传进来的，作为 block 的第一个参数）；
            # 输出通道数 = planes × block.expansion（会变成下一个 block 的输入）。。
            layers.append(block(self.in_planes, planes, stride))
            # 输出由第一个卷积操作执行后完成输出数与预期输出值相等
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet8(num_classes=10):
    return ResNet(BasicBlock, [1, 1, 1], num_classes)


def resnet20():
    """
    https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
    """
    return ResNet(BasicBlock, [3, 3, 3])


def show_conf_mat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, perc=False, save=True):
    """
    混淆矩阵绘制并保存图片
    :param confusion_mat:  nd.array
    :param classes: list or tuple, 类别名称
    :param set_name: str, 数据集名称 train or valid or test?
    :param out_dir:  str, 图片要保存的文件夹
    :param epoch:  int, 第几个epoch
    :param verbose: bool, 是否打印精度信息
    :param perc: bool, 是否采用百分比，图像分割时用，因分类数目过大
    :return:
    """
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        # confusion_mat[i, :].sum() 返回的是一个标量（即单个数值）
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / \
            confusion_mat[i, :].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        # 这段代码的作用是：让图像尺寸根据类别数平滑变化，确保在类别少时图像不会太大，类别多时不会太拥挤。
        figsize = np.linspace(6, 30, 91)[cls_num-10]

    fig, ax = plt.subplots(figsize=(int(figsize), int(figsize*1.3)))

    # 获取颜色
    # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    cmap = plt.cm.get_cmap('Greys')
    plt_object = ax.imshow(confusion_mat_tmp, cmap=cmap)
    # 这两行代码的作用是给图加上颜色条（colorbar），并设置其刻度字体大小：
    # raction=0.03 控制颜色条宽度相对于主图的比例，0.03 表示颜色条宽度是主图宽度的 3%。
    cbar = plt.colorbar(plt_object, ax=ax, fraction=0.03)
    # 设置颜色条刻度的字体大小为 12。
    cbar.ax.tick_params(labelsize='12')

    # 设置文字
    xlocations = np.array(range(len(classes)))
    ax.set_xticks(xlocations)
    # 用于设置 x 轴的刻度标签（tick labels），并将它们 旋转 60 度显示，常用于标签文字较长或标签过多时避免重叠。
    ax.set_xticklabels(list(classes), rotation=60)  # , fontsize='small'
    ax.set_yticks(xlocations)
    ax.set_yticklabels(list(classes))
    ax.set_xlabel('Predict label')
    ax.set_ylabel('True label')
    ax.set_title("Confusion_Matrix_{}_{}".format(set_name, epoch))

    # 打印数字
    if perc:
        # 将原始混淆矩阵中每个元素除以对应列的总数，得到 按列归一化后的百分比矩阵。
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                # ax.text(...)	在图中添加文本（文字写在指定的位置）
                # x=j, y=i	文本位置是矩阵的第 i 行第 j 列（x是列，y是行）
                # s="{:.0%}".format(conf_mat_per[i, j])	显示内容是：conf_mat_per[i, j] 的百分比形式（保留整数百分比）
                # 例如 0.56 会显示为 "56%"
                # va='center'	垂直方向对齐方式为居中（vertical alignment）
                # ha='center'	水平方向对齐方式为居中（horizontal alignment）
                # color='red'	字体颜色为红色
                # fontsize=10	字体大小为10
                ax.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va='center', ha='center', color='red',
                        fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s=int(
                    confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    if save:
        fig.savefig(os.path.join(
            out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    # 关闭当前绘图窗口，释放内存资源
    plt.close()

    if verbose:
        for i in range(cls_num):
            # 打印每个类别在混淆矩阵中的统计信息，包括总数、正确数、召回率（Recall）、精确率（Precision）。
            # classes[i],                          # 类别名称
            # np.sum(confusion_mat[i, :]),         # 该类的总样本数（实际）
            # confusion_mat[i, i],                 # 预测正确数
            # confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),  # Recall
            # confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))   # Precision
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))))

    return fig


class ModelTrainer(object):
    # 表示这是一个静态方法，不依赖于类实例
    # data_loader：训练数据加载器
    # model：要训练的模型
    # loss_f：损失函数
    # optimizer：优化器（如 SGD、Adam）
    # scheduler：学习率调度器（本代码未使用）
    # epoch_idx：当前 epoch 索引
    # device：CPU or CUDA
    # args：命令行参数（含打印频率等配置）
    # logger：日志记录器
    # classes：类别名称列表
    @staticmethod
    def train_one_epoch(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, args, logger, classes):
        # 设置模型为训练模式（启用 dropout、BN 的训练行为）
        # 记录时间戳，用于计时。
        model.train()
        end = time.time()
        # 创建空的混淆矩阵（confusion matrix），统计每个类别的预测结果。
        class_num = len(classes)
        conf_mat = np.zeros((class_num, class_num))
        # AverageMeter 是一个统计工具，用于记录每个指标的累计值和平均值。
        loss_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        batch_time_m = AverageMeter()
        # 获取最后一个 batch 的编号
        last_idx = len(data_loader) - 1
        for batch_idx, data in enumerate(data_loader):
            # 拿到图像和标签并移动到 GPU（或 CPU）。
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # forward & backward
            # 模型得到预测 outputs
# 计算损失 loss
# 梯度清零 → 反向传播 → 参数更新
            outputs = model(inputs)
            optimizer.zero_grad()

            loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()

            # 计算accuracy
            # acc1: top-1 accuracy（预测中最可能的类别是否是正确的）
# acc5: top-5 accuracy（前5个最可能中是否包含正确的标签）
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            # torch.max(tensor, dim)，dim=1：在 第 1 个维度（也就是每一行）上取最大值。
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                # 在混淆矩阵中 conf_mat[真实类, 预测类] 的位置上加 1：
# 表示“真实是 cate_i，预测成了 pre_i”的样本数量+1
                conf_mat[cate_i, pre_i] += 1.

            # 记录指标
            # 因update里： self.sum += val * n， 因此需要传入batch数量
            loss_m.update(loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), outputs.size(0))
            top5_m.update(acc5.item(), outputs.size(0))

            # 打印训练信息
            batch_time_m.update(time.time() - end)
            end = time.time()
            # 如果 print_freq = 100，就会在 batch_idx = 99, 199, 299, ... 时打印。
            if batch_idx % args.print_freq == args.print_freq - 1:
                # train: [  99/499]  Time: 0.123 (0.120)  Loss: 1.2345 (1.5678)  Acc@1: 76.54 (74.32)  Acc@5: 95.67 (93.21)
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        "train", batch_idx, last_idx, batch_time=batch_time_m,
                        loss=loss_m, top1=top1_m, top5=top5_m))  # val是当次传进去的值，avg是整体平均值。
        return loss_m, top1_m, conf_mat

    @staticmethod
    def evaluate(data_loader, model, loss_f, device, classes):
        model.eval()

        class_num = len(classes)
        conf_mat = np.zeros((class_num, class_num))

        loss_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_f(outputs.cpu(), labels.cpu())

            # 计算accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 记录指标
            # 因update里： self.sum += val * n， 因此需要传入batch数量
            loss_m.update(loss.item(), inputs.size(0))
            top1_m.update(acc1.item(), outputs.size(0))
            top5_m.update(acc5.item(), outputs.size(0))

        return loss_m, top1_m, conf_mat

# ModelTrainerEnsemble 继承自 ModelTrainer
# 主要用于测试多个模型共同推理（集成推理）的效果，适用于 集成学习（Ensemble Learning） 场景。
class ModelTrainerEnsemble(ModelTrainer):
    # 作用：将多个模型的输出结果（概率分布）进行逐元素平均，作为最终预测输出。
    @staticmethod
    def average(outputs):
        """Compute the average over a list of tensors with the same size."""
        return sum(outputs) / len(outputs)

    @staticmethod
    def evaluate(data_loader, models, loss_f, device, classes):

        class_num = len(classes)
        conf_mat = np.zeros((class_num, class_num))

        loss_m = AverageMeter()
        # 它能自动累积多个 batch 的结果，在评估结束时通过 .compute() 得到最终准确率
        top1_m = torchmetrics.Accuracy().to(device)

        # top1 acc group
        top1_group = []
        for model_idx in range(len(models)):
            top1_group.append(torchmetrics.Accuracy().to(device))

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 遍历每个模型：
# 得到 softmax 概率 output_single
# 记录进列表 outputs
# 用 top1_group[i] 统计该模型的准确率
            outputs = []
            for model_idx, model in enumerate(models):
                output_single = F.softmax(model(inputs), dim=1)
                outputs.append(output_single)
                # 计算单个模型acc
                top1_group[model_idx](output_single, labels)
                # 计算单个模型loss

            # 计算acc 组
            # 将多个 softmax 输出平均，得到集成输出
# top1_m 是对整个集成预测的准确率
            output_avg = ModelTrainerEnsemble.average(outputs)
            top1_m(output_avg, labels)
            # 注意损失是在 CPU 上计算的，确保 loss_f 支持非 GPU。
# 更新 loss_m（均值）
            # loss 组
            loss = loss_f(output_avg.cpu(), labels.cpu())
            # loss.item()会返回一个 Python 标量数值
            loss_m.update(loss.item(), inputs.size(0))

        return loss_m, top1_m.compute(), top1_group, conf_mat


class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        # 如果 log_name 非空（即 path_log 不是以 / 结尾的目录路径），则使用它作为 self.log_name
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log
        # 获取日志文件所在目录路径
        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        # 获取一个名为 self.log_name 的 logger 实例。
        # 如果之前没有创建过这个名字的 logger，则创建一个新的。
        # 这个名字可以帮助区分不同模块或文件的日志
        logger = logging.getLogger(self.log_name)
        # 设置 logger 的日志级别为 INFO，即只输出 INFO 及以上级别的日志（比如 WARNING, ERROR, CRITICAL）。

# 可选等级（从低到高）为：DEBUG < INFO < WARNING < ERROR < CRITICAL
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        # 创建一个写入日志文件的 handler。
# self.out_path 是日志文件的完整路径。
# 'w' 模式表示每次启动都会覆盖原来的日志文件（写入模式）。
# 如果想追加日志，可使用 'a' 模式。
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        # 创建一个将日志输出到终端（标准输出/控制台）的 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(out_dir):
    """
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger用于记录信息
    :param out_dir: str
    :return:
    """
    # 获取当前时间（精确到秒），用于命名日志目录
    now_time = datetime.now()
    # 这个字符串会被用作日志子目录名，便于区分不同运行时刻的日志
    time_str = datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    # 创建一个 Logger 类的实例，传入日志文件路径 path_log。
# 这个 Logger 类是你前面定义过的那个自定义类，负责日志目录初始化等。
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir


def setup_seed(seed=42):
    # 设置 NumPy 的全局随机种子。
    # 控制 np.random.* 生成的随机数序列保持一致。
    np.random.seed(seed)
    # 设置 Python 内置 random 模块的随机种子。

# 控制例如 random.random()、random.shuffle() 等操作的结果一致
    random.seed(seed)
    # 设置 PyTorch 在 CPU 上的随机数生成器种子，如 torch.rand()、torch.randn() 等。
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        # 设置所有 GPU 的随机数种子（适用于多卡训练）。
        torch.cuda.manual_seed_all(seed)
        # 设置 cuDNN（NVIDIA 的加速库）为确定性模式。

# 有些 cuDNN 的操作为了加速会使用不确定算法，这会导致相同输入每次结果略有差异。

# 设置为 True 能保证相同输入得到相同输出，但可能会降低训练速度。
        torch.backends.cudnn.deterministic = True
        # 这个选项通常用于提升性能：它让 cuDNN 自动寻找最优算法。
# 但**⚠️这和 deterministic = True 是冲突的**。
        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark = False
    # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法


class AverageMeter:
    """Computes and stores the average and current value
    Hacked from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# output: 模型输出的 logits，形状是 [batch_size, num_classes]
# target: 真实标签，形状是 [batch_size]
# topk: 元组，如 (1,) 表示 top-1；(1, 5) 表示同时计算 top-1 和 top-5 准确率


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Hacked from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py"""
    # 得到要计算的最大 k 值（比如 top-5），但不能超过类别总数
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    # input	输入张量（如 logits）
# k	取前 k 个最大/最小值
# dim=1	沿着第 1 维（每行）进行操作，即对每个样本的所有类别
# largest=True	取最大值（False 就是取最小的）
# sorted=True	是否按降序排序输出结果
    _, pred = output.topk(maxk, 1, True, True)
    # 转置后 pred.shape == [maxk, batch_size]，方便后续广播比较
    pred = pred.t()
    # target.reshape(1, -1)
# 把 target 转换为形状 [1, batch_size]
# .expand_as(pred)
# 把 [1, batch_size] 扩展为 [k, batch_size]
# 复制成 k 行，让它的形状和 pred 一样，才能逐个对比
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # correct[:min(k, maxk)]取前 k 行（即 top-k 中的所有预测结果）
    # .reshape(-1)将其展平为一维
    # .float()将布尔值转换为浮点数（True → 1.0, False → 0.0）
    # .sum(0)对每列求和，得到每个样本的正确预测数
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
