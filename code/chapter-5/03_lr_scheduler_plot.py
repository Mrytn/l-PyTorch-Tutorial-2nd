# -*- coding:utf-8 -*-
"""
@file name  : 03_lr_scheduler_plot.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-01
@brief      : scheduler 绘图
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
# 设置随机种子，保证实验的可复现性。
torch.manual_seed(1)
LR = 0.1  # 初始学习率
iteration = 10  # 每个 epoch 内部迭代的次数
max_epoch = 200  # 总的训练 epoch 数
# ------------------------------ fake data and optimizer  ------------------------------
# 创建一个需要梯度更新的参数。
weights = torch.randn((1), requires_grad=True)
# 目标值设为 0（可以看作简单的优化目标）
target = torch.zeros((1))
# 使用 SGD 优化器，并设置 momentum=0.9 来加速收敛。
optimizer = optim.SGD([weights], lr=LR, momentum=0.9)
# ------------------------------ 1 Step LR ------------------------------
scheduler_lr = optim.lr_scheduler.StepLR(
    optimizer, step_size=50, gamma=0.1)  # 设置学习率下降策略
lr_list, epoch_list = list(), list()
for epoch in range(max_epoch):
    # 获取当前lr，新版本用 get_last_lr()函数，旧版本用get_last_lr()函数，具体看UserWarning
    lr_list.append(scheduler_lr.get_last_lr())
    epoch_list.append(epoch)
    for i in range(iteration):
        # 均方误差损失函数
        loss = torch.pow((weights - target), 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    #  # 每个 epoch 结束后更新学习率
    scheduler_lr.step()
plt.plot(epoch_list, lr_list, label="Step LR Scheduler")
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.show()
# ------------------------------ 2 Multi Step LR ------------------------------
# 只在这些 epoch 发生学习率衰减
milestones = [50, 125, 160]
scheduler_lr = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=milestones, gamma=0.1)
lr_list, epoch_list = list(), list()
for epoch in range(max_epoch):
    lr_list.append(scheduler_lr.get_last_lr())
    epoch_list.append(epoch)
    for i in range(iteration):
        loss = torch.pow((weights - target), 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler_lr.step()
plt.plot(epoch_list, lr_list,
         label="Multi Step LR Scheduler\nmilestones:{}".format(milestones))
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.show()
# ------------------------------ 3 Exponential LR ------------------------------
#  每个 epoch 乘以 gamma（0.95）
gamma = 0.95
scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
lr_list, epoch_list = list(), list()
for epoch in range(max_epoch):
    lr_list.append(scheduler_lr.get_last_lr())
    epoch_list.append(epoch)
    for i in range(iteration):
        loss = torch.pow((weights - target), 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler_lr.step()
plt.plot(epoch_list, lr_list,
         label="Exponential LR Scheduler\ngamma:{}".format(gamma))
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.show()
# ------------------------------ 4 Cosine Annealing LR ------------------------------
# 代表 50 个 epoch 内从最大值衰减到 eta_min，然后重新开始
# 学习率缓慢下降，接近 0 后重新回升（适用于周期性训练任务）
# 初始最大学习率为0.1
t_max = 50
scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=t_max, eta_min=0.)
lr_list, epoch_list = list(), list()
for epoch in range(max_epoch):
    lr_list.append(scheduler_lr.get_last_lr())
    epoch_list.append(epoch)
    for i in range(iteration):
        loss = torch.pow((weights - target), 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler_lr.step()
plt.plot(epoch_list, lr_list,
         label="CosineAnnealingLR Scheduler\nT_max:{}".format(t_max))
plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.legend()
plt.show()
# ------------------------------ 5 Reduce LR On Plateau ------------------------------
# 适用于 loss 下降不稳定的情况，学习率衰减依赖于 loss 而非固定周期
loss_value = 0.5  # 初始损失
factor = 0.1  # 当触发学习率调整时，学习率乘以该因子（即缩小到 10%）
mode = "min"  # 监测 loss，目标是最小化它
patience = 10  # 若 loss 在 10 个 epoch 内未改善，则降低学习率
cooldown = 10  # 降低学习率后，等待 10 个 epoch 才能再次降低
min_lr = 1e-4  # 最小学习率，不会降低到这个值以下
verbose = True  # 输出学习率变化信息
# 这个调度器会在 loss_value 停止下降 patience 轮后降低学习率。
# mode="min" 表示当 loss_value 下降时，认为训练有进展，不会调整学习率。
scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, mode=mode, patience=patience,
                                                    cooldown=cooldown, min_lr=min_lr, verbose=verbose)
# 代码模拟了一个 loss_value 下降的情况（epoch = 5 时 loss_value 变小）。
# scheduler_lr.step(loss_value) 会检查 loss 是否下降了，如果 patience 轮内没有下降，就降低学习率
for epoch in range(max_epoch):
    for i in range(iteration):
        # train(...)
        optimizer.step()
        optimizer.zero_grad()
    if epoch == 5:
        loss_value = 0.4
    scheduler_lr.step(loss_value)
# ------------------------------ 6 lambda ------------------------------
lr_init = 0.1
weights_1 = torch.randn((6, 3, 5, 5))
weights_2 = torch.ones((5, 5))
# weights_1 和 weights_2 代表神经网络的权重参数，分别初始化。
# 这两个参数的学习率都从 0.1 开始。
optimizer = optim.SGD([
    {'params': [weights_1]},
    {'params': [weights_2]}], lr=lr_init)


def lambda1(epoch): return 0.1 ** (epoch // 20)# # 每 20 轮衰减 10 倍
def lambda2(epoch): return 0.95 ** epoch# # 每轮衰减 0.95 倍

# 这里 lr_lambda 传入了两个函数，分别作用于 weights_1 和 weights_2。
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=[lambda1, lambda2])
lr_list, epoch_list = list(), list()
for epoch in range(max_epoch):
    for i in range(iteration):
        # train(...)
        optimizer.step()
        optimizer.zero_grad()
    # 每轮都会 scheduler.step() 让学习率更新
    scheduler.step()
    # get_last_lr() 返回的是所有参数组的学习率，因此 lr_list[i] 是一个包含两个学习率的列表：
    lr_list.append(scheduler.get_last_lr())
    epoch_list.append(epoch)
    print('epoch:{:5d}, lr:{}'.format(epoch, scheduler.get_last_lr()))
plt.plot(epoch_list, [i[0] for i in lr_list], label="lambda 1")
plt.plot(epoch_list, [i[1] for i in lr_list], label="lambda 2")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("LambdaLR")
plt.legend()
plt.show()
