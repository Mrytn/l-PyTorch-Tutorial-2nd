# -*- coding:utf-8 -*-
"""
@file name  : 07_model_ensemble.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-30
@brief      : 模型集成
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchensemble.fusion import FusionClassifier
from torchensemble.voting import VotingClassifier
from torchensemble.bagging import BaggingClassifier
from torchensemble.gradient_boosting import GradientBoostingClassifier
from torchensemble.snapshot_ensemble import SnapshotEnsembleClassifier
from torchensemble.soft_gradient_boosting import SoftGradientBoostingClassifier
from torchensemble.utils.logging import set_logger
# 记录模型结果
# 将多个模型的测试结果（方法名、训练时间、评估时间、准确率）以美观的格式写入日志中，方便统一查看和比较
def display_records(records, logger):
    # 这是一段格式化字符串，定义了输出日志的格式：
    # {: < 28} 表示左对齐，占 28 个字符的宽度（用来放方法名 method）。
    # {: .2f} 表示浮点数保留两位小数
    msg = (
        "{:<28} | Testing Acc: {:.2f} % | Training Time: {:.2f} s |"
        " Evaluating Time: {:.2f} s"
    )
    print("\n")
    for method, training_time, evaluating_time, acc in records:
        logger.info(msg.format(method, acc, training_time, evaluating_time))
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


if __name__ == "__main__":
    # Hyper-parameters
    # n_estimators: 如果你用的是集成方法，比如 AdaBoost 或 Bagging，这个表示子模型的数量。
    # lr: 学习率（Learning Rate），控制每次参数更新的步长。
    # weight_decay: 权重衰减（L2正则化），防止过拟合。
    # epochs: 总训练轮数。
    n_estimators = 5
    lr = 1e-3
    weight_decay = 5e-4
    epochs = 100
    # Utils
    data_dir = r"D:\ai\pytorch\l-PyTorch-Tutorial-2nd\data\datasets\cifar10-office"
    batch_size = 128
    records = []
    torch.manual_seed(0)
    # Load data
    train_transform = transforms.Compose(
        [
            # 随机水平翻转图像，增强泛化能力。
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    # root变量下需要存放cifar-10-python.tar.gz 文件
    # cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
    train_set = datasets.CIFAR10(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, transform=valid_transform, download=True)
    # 构建DataLoder
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, num_workers=4)
    logger = set_logger("classification_cifar10_cnn", use_tb_logger=True)
    #
    # ============================= FusionClassifier =============================
    # 创建一个名为 FusionClassifier 的集成模型实例
    model = FusionClassifier(
        # estimator=LeNet5: 基模型是 LeNet5（卷积神经网络，经典用于图像分类）。
        # n_estimators=n_estimators: 子模型数量（前面设置的是 5）。
        # cuda=True: 如果有 GPU，就使用 GPU 加速。
        # 5个子模型 结构上一样，参数初始化可能不同，训练过程中也可能各自学习不同的东西。
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )
    # Set the optimizer
    # 设置优化器
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)
    # Training
    #  模型训练
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    # 记录训练前后的时间，用来计算训练耗时。
    # training_time 就是模型训练一共用了多少秒。
    training_time = toc - tic
    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic
    # 将本次训练的结果加入 records 列表。
# 这个 records 就是为后续 display_records(records, logger) 准备的。
    records.append(
        ("FusionClassifier", training_time, evaluating_time, testing_acc)
    )
    # ============================= VotingClassifier =============================
    model = VotingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )
    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)
    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic
    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic
    records.append(
        ("VotingClassifier", training_time, evaluating_time, testing_acc)
    )
    # ============================= BaggingClassifier =============================
    model = BaggingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )
    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)
    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic
    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic
    records.append(
        ("BaggingClassifier", training_time, evaluating_time, testing_acc)
    )
    # ============================= GradientBoostingClassifier =============================
    model = GradientBoostingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )
    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)
    # Training
    tic = time.time()
    # model.fit(train_loader, epochs=epochs)
    model.fit(train_loader, epochs=1)
    toc = time.time()
    training_time = toc - tic
    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic
    records.append(
        (
            "GradientBoostingClassifier",
            training_time,
            evaluating_time,
            testing_acc,
        )
    )
    # ============================= SnapshotEnsembleClassifier =============================
    model = SnapshotEnsembleClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )
    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)
    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic
    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic
    records.append(
        (
            "SnapshotEnsembleClassifier",
            training_time,
            evaluating_time,
            testing_acc,
        )
    )
    # ============================= SoftGradientBoostingClassifier =============================
    model = SoftGradientBoostingClassifier(
        estimator=LeNet5, n_estimators=n_estimators, cuda=True
    )
    # Set the optimizer
    model.set_optimizer("Adam", lr=lr, weight_decay=weight_decay)
    # Training
    tic = time.time()
    model.fit(train_loader, epochs=epochs)
    toc = time.time()
    training_time = toc - tic
    # Evaluating
    tic = time.time()
    testing_acc = model.evaluate(valid_loader)
    toc = time.time()
    evaluating_time = toc - tic
    records.append(
        (
            "SoftGradientBoostingClassifier",
            training_time,
            evaluating_time,
            testing_acc,
        )
    )
    # Print results on different ensemble methods
    display_records(records, logger)
