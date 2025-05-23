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
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
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
    # m.__class__.__name__ 这行代码的作用是获取对象 m 所属类的类名
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # 使用 kaiming_normal_ 方法初始化当前层的权重
        init.kaiming_normal_(m.weight)
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
class BasicBlock(nn.Module):
    # expansion = 1：表示这个 block 不会改变输出通道数量。
    expansion = 1
    # in_planes 表示输入特征图的通道数。
# planes 表示输出特征图的通道数。
# stride 是卷积操作的步长，默认值为 1。
# option 是用于处理残差连接的选项，默认值为 'A'。
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        # self.conv1 和 self.conv2 是两个二维卷积层，kernel_size=3 表示卷积核的大小为 3x3，padding=1 保证卷积操作后特征图的尺寸不变（当 stride=1 时），bias=False 表示不使用偏置项。
        # self.bn1 和 self.bn2 是批量归一化层，用于加速模型的训练和提高模型的稳定性。
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.shortcut 是残差连接，默认初始化为一个空的 nn.Sequential 对象
        self.shortcut = nn.Sequential()
        # 如果 stride 不等于 1 或者 in_planes 不等于 planes，则需要对输入进行处理，以匹配输出的维度。
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # x[:, :, ::2, ::2] 会在第 2 维和第 3 维上进行下采样，也就是将输入张量的高度和宽度缩小一半
                # ::2 表示每隔两个元素取一个，相当于下采样（stride=2）
                # 原图大小是 32×32 → 下采样成 16×16
                #            没有用卷积，直接采样（这就是轻量的意思）
                # (0, 0, 0, 0, planes//4, planes//4)：这是填充的参数，填充顺序是从最后一个维度开始向前填充。对于一个 4 维张量（batch_size, channels, height, width），这里的填充顺序是 (左, 右, 上, 下, 前, 后)。所以 (0, 0, 0, 0, planes//4, planes//4) 表示在宽度和高度上不进行填充，在通道维度上前后各填充 planes//4 （32）个通道。原始通道数 = 64
# padding 前后各补 32
# 最终通道数 = 64 + 32 + 32 = 128
# "constant"：表示填充的方式为常数填充。
# 0：表示填充的常数值为 0。
# 从 [B, 64, 32, 32]
# 变成 [B, 128, 16, 16]
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            # 用 1×1 卷积调整通道 + 下采样
            # 这是标准的 projection shortcut，参数更多，但更强
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
    # 输入 x → conv1 → BN → ReLU
# → conv2 → BN
# 加上 shortcut 的输出 F(x)
# 最后再一次 ReLU
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  # 原版16
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  # 原版32
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # 原版64
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
    def _make_layer(self, block, planes, num_blocks, stride):
        # 如果stride = 2 num_blocks = 3
        # layer 由 3 个 block 组成
        # [2] + [1, 1]  # 最终变成 [2, 1, 1]
        # 第一个 block：stride=2，进行下采样
        #  后续 blocks：stride=1，保持尺寸一致
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # block(self.in_planes, planes, stride)：创建一个残差块实例，并将其添加到 layers 列表中。self.in_planes 表示输入的通道数，planes 表示输出的通道数，stride 表示该残差块的步长。
            layers.append(block(self.in_planes, planes, stride))
            # self.in_planes = planes * block.expansion：更新 self.in_planes 的值，使其等于当前残差块输出的通道数。block.expansion 是残差块类的一个属性，表示通道数的扩展因子。
            self.in_planes = planes * block.expansion
            # 使用 nn.Sequential 将 layers 列表中的所有残差块组合成一个顺序模块，并返回该模块。*layers 是 Python 的解包操作，将列表中的元素依次传递给 nn.Sequential。
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
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / \
            confusion_mat[i, :].sum()
    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num-10]
    fig, ax = plt.subplots(figsize=(int(figsize), int(figsize*1.3)))
    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt_object = ax.imshow(confusion_mat_tmp, cmap=cmap)
    cbar = plt.colorbar(plt_object, ax=ax, fraction=0.03)
    cbar.ax.tick_params(labelsize='12')
    # 设置文字
    xlocations = np.array(range(len(classes)))
    ax.set_xticks(xlocations)
    ax.set_xticklabels(list(classes), rotation=60)  # , fontsize='small'
    ax.set_yticks(xlocations)
    ax.set_yticklabels(list(classes))
    ax.set_xlabel('Predict label')
    ax.set_ylabel('True label')
    ax.set_title("Confusion_Matrix_{}_{}".format(set_name, epoch))
    # 打印数字
    if perc:
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va='center', ha='center', color='red',
                         fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                ax.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    if save:
        fig.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()
    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i]))))
    return fig
class ModelTrainer(object):
    # @staticmethod这是 Python 里的一个装饰器，其作用是把一个方法转换为静态方法。静态方法属于类，而非类的实例，调用时无需创建类的实例
    @staticmethod
    def train_one_epoch(data_loader, model, loss_f, optimizer, scheduler, epoch_idx, device, args, logger, classes):
        model.train()
        end = time.time()
        # 初始化各种监控指标
        # AverageMeter用于计算并存储平均值和当前值
        class_num = len(classes)
        conf_mat = np.zeros((class_num, class_num))
        loss_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        batch_time_m = AverageMeter()
        last_idx = len(data_loader) - 1
        for batch_idx, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # forward & backward
            # 前向传播 + 反向传播 + 优化
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()
            # 计算accuracy前5准确率
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                # 混淆矩阵
                conf_mat[cate_i, pre_i] += 1.
            # 记录指标
            loss_m.update(loss.item(), inputs.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
            top1_m.update(acc1.item(), outputs.size(0))
            top5_m.update(acc5.item(), outputs.size(0))
            # 打印训练信息
            batch_time_m.update(time.time() - end)
            end = time.time()
            if batch_idx % args.print_freq == args.print_freq - 1:
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
            loss_m.update(loss.item(), inputs.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
            top1_m.update(acc1.item(), outputs.size(0))
            top5_m.update(acc5.item(), outputs.size(0))
        return loss_m, top1_m, conf_mat
# 输入一组模型 models（比如多个训练好的模型）
# 对每个模型分别计算输出的 Softmax（即预测概率）
# 把多个模型输出平均，作为集成预测结果
# 同时记录：
# 每个模型的准确率
# 集成结果的准确率
# 集成结果的 loss
class ModelTrainerEnsemble(ModelTrainer):
    @staticmethod
    def average(outputs):
        """Compute the average over a list of tensors with the same size."""
        return sum(outputs) / len(outputs)
    @staticmethod
    def evaluate(data_loader, models, loss_f, device, classes):
        class_num = len(classes)
        conf_mat = np.zeros((class_num, class_num))
        loss_m = AverageMeter()
        top1_m = torchmetrics.Accuracy(
            task='multiclass', num_classes=class_num).to(device)  # 记录ensemble的acc
        # top1 acc group
        top1_group = []
        for model_idx in range(len(models)):
            # # 每个模型的acc
            top1_group.append(torchmetrics.Accuracy(
                task='multiclass', num_classes=class_num).to(device))
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = []
            for model_idx, model in enumerate(models):
                # # 先过模型，再做 softmax
                output_single = F.softmax(model(inputs), dim=1)
                outputs.append(output_single)
                # 计算单个模型acc
                top1_group[model_idx](output_single, labels)
                # 计算单个模型loss
            # 计算acc 组
            output_avg = ModelTrainerEnsemble.average(outputs)
            top1_m(output_avg, labels)
            # loss 组
            loss = loss_f(output_avg.cpu(), labels.cpu())
            loss_m.update(loss.item(), inputs.size(0))
        return loss_m, top1_m.compute(), top1_group, conf_mat
class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log
        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)
        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        # 配置屏幕Handler
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
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir
def setup_seed(seed=42):
    # 设置 NumPy 的随机数种子。影响如 np.random.rand() 之类的操作
    np.random.seed(seed)
    # 设置 Python 自带的 random 模块的随机数种子。比如影响 random.randint()
    random.seed(seed)
    # 设置 PyTorch 的 CPU 随机数种子，影响如 torch.randn()、torch.randperm() 等。
    torch.manual_seed(seed)     # cpu
    if torch.cuda.is_available():
        # 设置所有 GPU 上的随机种子。因为你可能有多个 GPU 卡，确保每张卡都统一种子
        torch.cuda.manual_seed_all(seed)
        # 使用固定的算法实现（deterministic），强制 CUDNN 以可重复的方式执行
        torch.backends.cudnn.deterministic = True
        # 启用 cudnn 的自动算法搜索优化（benchmark 模式），提升性能
        torch.backends.cudnn.benchmark = True       # 训练集变化不大时使训练加速，是固定cudnn最优配置，如卷积算法
# 计算并存储某个值的当前值、总和、数量以及平均值
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
# 计算模型预测结果在指定的前 k 个预测中的准确率
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    Hacked from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py"""
    # 取 topk 中的最大值和 output 中类别数量的较小值，确保不会超出类别数量。
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    # 获取张量中指定维度上的前 maxk 个最大值及其索引
    # 1：指定在维度 1 上进行操作，即按行获取最大值。
    # True（第一个）：largest 参数，是一个布尔值。如果设置为 True，则返回最大的 k 个元素；如果设置为 False，则返回最小的 k 个元素。这里设置为 True，表示返回最大的 maxk 个元素。
# True（第二个）：sorted 参数，也是一个布尔值。如果设置为 True，则返回的元素会按照降序排列；如果设置为 False，则不保证返回的元素是有序的。这里设置为 True，表示返回的元素会按照从大到小的顺序排列。
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # 将真实标签 target 重塑为形状 (1, batch_size) 的张量
    # expand_as(pred)：将重塑后的真实标签张量扩展为与 pred 相同的形状。
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # sum(0)：对第一个维度求和，得到每个样本在前 maxk 个预测中的正确预测数量。
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]