# -*- coding: utf-8 -*-
"""
# @file name  : 02_fintune-freeze.py
# @author     :  TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-24
# @brief      : finetune方法之冻结特征提取层
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from my_utils import *
BASEDIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))
label_name = {"ants": 0, "bees": 1}
class AntsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {"ants": 0, "bees": 1}
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.data_info)
    def get_img_info(self, data_dir):
        data_info = list()
        # 使用 os.walk 遍历根目录
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(
                    filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = self.label_name[sub_dir]
                    data_info.append((path_img, int(label)))
        if len(data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return data_info
# 参数设置
max_epoch = 25
BATCH_SIZE = 16
LR = 0.001
# 每训练 log_interval 个 batch 记录一次日志（如 loss）。
log_interval = 10
# 每训练 val_interval 个 epoch 进行一次验证。
val_interval = 1
# 分类类别数，本例为 2（蚂蚁、蜜蜂）
classes = 2
# 	起始 epoch，默认是 -1，通常用于断点恢复训练。
start_epoch = -1
# 每训练 7 个 epoch，学习率衰减一次（适用于调度器）。
lr_decay_step = 7
# 每训练 print_interval 个 batch 打印一次控制台输出。
print_interval = 2
# ------------------------------------  log ------------------------------------
result_dir = os.path.join("Result")
now_time = datetime.now()
#  # 格式化时间作为日志目录名
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
# # 最终日志路径
log_dir = os.path.join(result_dir, time_str)
# # 如果日志路径不存在就创建
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)
# ============================ step 1/5 数据 ============================
# https://download.pytorch.org/tutorial/hymenoptera_data.zip
data_dir = r"data\datasets\hymenoptera_data"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "val")
# # 图像归一化的均值
norm_mean = [0.485, 0.456, 0.406]
# # 图像归一化的标准差
norm_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),      # 随机裁剪到 224x224
    transforms.RandomHorizontalFlip(),      # 水平翻转（增强数据）
    transforms.ToTensor(),                  # 转成 Tensor，值范围 [0, 1]
    transforms.Normalize(norm_mean, norm_std),  # 标准化
])
valid_transform = transforms.Compose([
    # Resize(256)保持图像原始宽高比（aspect ratio）不变，将较短边调整为 256 像素，较长边按比例缩放。
    transforms.Resize(256),                 # 调整图像大小
    transforms.CenterCrop(224),            # 中心裁剪到 224x224
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
# 构建MyDataset实例
train_data = AntsDataset(data_dir=train_dir,    transform=train_transform)
valid_data = AntsDataset(data_dir=valid_dir, transform=valid_transform)
# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
# ============================ step 2/5 模型 ============================
# 1/3 构建模型
resnet18_ft = models.resnet18()
# 2/3 加载参数
# download resnet18-f37072fd.pth from:
# https://download.pytorch.org/models/resnet18-f37072fd.pth
path_pretrained_model = r"data\model_zoo\resnet18-f37072fd.pth"
# torch.load 加载 .pth 格式的参数字典（通常是用 torch.save(model.state_dict()) 保存的）
state_dict_load = torch.load(path_pretrained_model)
# 将这些参数加载进模型
resnet18_ft.load_state_dict(state_dict_load)
# 法1: 冻结卷积层
# 设置所有参数的 requires_grad=False，表示训练时不更新这些参数（不参与反向传播）
# 适用于只用预训练网络提取图像特征，只训练最后的全连接层（分类器）
# 节省计算资源、加快训练速度
for param in resnet18_ft.parameters():
    param.requires_grad = False
# 出的是 conv1 层第一个卷积核的参数（第一个输入通道）
# 通常用于确认模型是否成功加载、或验证冻结是否成功
print("conv1.weights[0, 0, ...]:\n {}".format(
    resnet18_ft.conv1.weight[0, 0, ...]))
# 替换fc层
# in_features 是全连接层的一个属性，它代表输入到该全连接层的特征数量
num_ftrs = resnet18_ft.fc.in_features
# nn.Linear(num_ftrs, classes) 创建了一个新的全连接层，该层的输入特征数量为 num_ftrs，输出特征数量为 classes。这里的 classes 通常代表分类任务中的类别数量。
resnet18_ft.fc = nn.Linear(num_ftrs, classes)
# 把模型放到你之前定义的 device 上（如 'cuda:0' 或 'cpu'）
# 必须保证你的数据和模型都在同一个 device 上！
resnet18_ft.to(device)
# ============================ step 3/5 损失函数 ============================
# 选择损失函数
criterion = nn.CrossEntropyLoss()
# =========================== step 4/5 优化器 ============================
# 如果你冻结了卷积层，记得训练时只传入 requires_grad=True 的参数
optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)               # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=lr_decay_step, gamma=0.1)     # 设置学习率下降策略
# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()
# 支持从 start_epoch 断点恢复（设为 -1 表示从第 0 轮开始）
for epoch in range(start_epoch + 1, max_epoch):
    # 训练
    conf_mat = np.zeros((classes, classes))  # 混淆矩阵
    loss_sigma = []                       # 收集每个 batch 的 loss
    loss_avg = 0
    acc_avg = 0
    path_error = []                       # 用于记录错误样本路径（可选）
    label_list = []                       # 用于记录所有标签（选）
    # 模型训练
    resnet18_ft.train()
    for i, data in enumerate(train_loader):
        # forward
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # forward, backward, update weights
        # 前向 + 反向传播
        optimizer.zero_grad()
        outputs = resnet18_ft(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 统计分类情况
        # 统计预测信息
        loss_sigma.append(loss.item())
        loss_avg = np.mean(loss_sigma)
        #  # 取最大概率作为预测结果
        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(labels)):
            # 真实标签
            cate_i = labels[j].cpu().numpy()
            # 预测标签
            pre_i = predicted[j].cpu().numpy()
            # 对应标签位置加1
            conf_mat[cate_i, pre_i] += 1.
        # 计算准确率
        # trace() 是 numpy 数组的一个方法，它的作用是计算矩阵的迹。矩阵的迹指的是矩阵主对角线元素之和，在混淆矩阵里，主对角线元素代表的是模型正确分类的样本数量。
        acc_avg = conf_mat.trace() / conf_mat.sum()
        # 打印训练信息
        # 因为从第 0 轮开始训练，所以print_interval - 1打印的是第 2, 4, 6, ... 个 batch（从 0 开始数）
        # 如果print_interval == 0，打印的就是第 1, 3, 5, ... 个 batch（从 0 开始数）
        if i % print_interval == print_interval - 1:
            # {:0>3}：这是格式化指令，0 表示用 0 来填充，> 表示右对齐，3 表示宽度为 3 个字符。它会把传入的值格式化为 3 位宽度的字符串，如果不足 3 位则在左边补 0。这里用于格式化 epoch + 1、max_epoch、i + 1 和 len(train_loader)。
            # {:.4f}：将传入的浮点数格式化为小数点后保留 4 位的字符串，用于格式化 loss_avg（平均损失）。
            # {:.2%}：把传入的小数格式化为百分比形式，且保留两位小数，用于格式化 acc_avg（平均准确率）。
            # 这段代码会输出训练过程中的信息，包含当前的训练轮数（Epoch）、当前迭代次数、总迭代次数、平均损失和平均准确率。
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                  format(epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, acc_avg))
            # 输出当前训练轮数以及 ResNet - 18 模型第一层卷积层的部分权重信息，有助于你观察模型在训练过程中权重的变化情况
            print("epoch:{} conv1.weights[0, 0, ...] :\n {}".format(
                epoch, resnet18_ft.conv1.weight[0, 0, ...]))
    scheduler.step()  # 更新学习率
    # 记录训练loss
    # TensorBoard 写入
    writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
    # 记录learning rate
    # [0]：这是列表索引操作，其作用是取出列表中的第一个元素。在大多数情况下，优化器只有一个参数组，因此使用 [0] 来获取当前的学习率。
    writer.add_scalar('learning rate', scheduler.get_last_lr()[0], epoch)
    # 记录Accuracy
    writer.add_scalars('Accuracy_group', {'train_acc': acc_avg}, epoch)
    # 自定义函数 show_conf_mat 用于生成 matplotlib 图像，并写入 TensorBoard
# verbose=True 表示最后一轮保存图像文件
    conf_mat_figure = show_conf_mat(conf_mat, list(label_name.keys()), "train", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    writer.add_figure('confusion_matrix_train',
                      conf_mat_figure, global_step=epoch)
    # validate the model
    # 验证模型
    class_num = classes
    conf_mat = np.zeros((class_num, class_num))
    loss_sigma = []
    loss_avg = 0
    acc_avg = 0
    path_error = []
    label_list = []
    resnet18_ft.eval()
    with torch.no_grad():
        for j, data in enumerate(valid_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet18_ft(inputs)
            loss = criterion(outputs, labels)
            # 统计预测信息
            loss_sigma.append(loss.item())
            loss_avg = np.mean(loss_sigma)
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.
            acc_avg = conf_mat.trace() / conf_mat.sum()
    print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
    # 记录Loss, accuracy
    writer.add_scalars('Loss_group', {'valid_loss': loss_avg}, epoch)
    writer.add_scalars('Accuracy_group', {'valid_acc': acc_avg}, epoch)
    # 保存混淆矩阵图
    conf_mat_figure = show_conf_mat(conf_mat, list(label_name.keys()), "valid", log_dir, epoch=epoch, verbose=epoch == max_epoch - 1)
    writer.add_figure('confusion_matrix_valid', conf_mat_figure, global_step=epoch)
print('Finished Training')
