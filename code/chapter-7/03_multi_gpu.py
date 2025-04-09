# -*- coding: utf-8 -*-
"""
# @file name  : 03_multi_gpu.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-25
# @brief      : 多gpu示例。单机多卡情况。
"""
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FooNet(nn.Module):
    def __init__(self, neural_num, layers=3):
        super(FooNet, self).__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])

    def forward(self, x):
        # 每次前向传播会打印 batch 大小（用于验证 DataParallel 是否做了拆分）
        print("\nbatch size in forward: {}".format(x.size()[0]))
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
        return x


if __name__ == "__main__":
    batch_size = 16
    # data
    inputs = torch.randn(batch_size, 3)
    labels = torch.randn(batch_size, 3)
    inputs, labels = inputs.to(device), labels.to(device)
    # model
    net = FooNet(neural_num=3, layers=3)
    # 用 DataParallel 包装模型，使其在多 GPU 上运行。
# DataParallel 会自动把输入数据按 batch 拆分成多个子 batch，分发到多个 GPU 上执行。
    net = nn.DataParallel(net)
    net.to(device)
    # training
    for epoch in range(1):
        # 训练 loop
        outputs = net(inputs)
        # 打印个数
        print("model outputs.size: {}".format(outputs.size()))
    print("device_count :{}".format(torch.cuda.device_count()))
