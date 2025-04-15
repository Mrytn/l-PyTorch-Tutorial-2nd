# -*- coding: utf-8 -*-
"""
# @file name  : 03_model_load_in_gpu.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2022-06-25
# @brief      : DataParallel的保存与加载
"""
import os
import torch
import torch.nn as nn
class FooNet(nn.Module):
    def __init__(self, neural_num, layers=3):
        super(FooNet, self).__init__()
        self.linears = nn.ModuleList(
            [nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])

    def forward(self, x):
        print("\nbatch size in forward: {}".format(x.size()[0]))
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
        return x


# =================================== 加载至cpu
# # 控制开关：0 表示不用 GPU，1 表示使用 GPU
flag = 0
# flag = 1
if flag:
    # 这行代码定义了一个列表 gpu_list，列表中包含一个元素 0。在 CUDA 中，GPU 设备是通过编号来标识的，编号从 0 开始。这里的 0 表示使用编号为 0 的 GPU 设备。如果需要使用多个 GPU 设备，可以将多个设备编号添加到列表中，例如 gpu_list = [0, 1, 2] 表示使用编号为 0、1 和 2 的三个 GPU 设备。
    gpu_list = [0]
    # map(str, gpu_list)：map 是 Python 的内置函数，它会将 str 函数应用到 gpu_list 中的每个元素上，将列表中的整数元素转换为字符串类型
    gpu_list_str = ','.join(map(str, gpu_list))
    # os.environ：这是 Python 的 os 模块中的一个字典，它包含了当前进程的环境变量。
# setdefault(key, default)：这是字典的方法，用于获取指定键的值，如果键不存在，则将键和默认值添加到字典中，并返回默认值。
# "CUDA_VISIBLE_DEVICES"：这是一个 CUDA 相关的环境变量，用于指定哪些 GPU 设备对于当前进程是可见的。通过设置这个环境变量，可以限制当前进程只能使用指定的 GPU 设备。
# gpu_list_str：作为 setdefault 方法的默认值，如果 "CUDA_VISIBLE_DEVICES" 环境变量已经存在，则不会改变其值；如果不存在，则将其设置为 gpu_list_str。
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FooNet(neural_num=3, layers=3)
    net.to(device)
    # save
    net_state_dict = net.state_dict()
    path_state_dict = "code/chapter-7/model_in_gpu_01.pkl"
    torch.save(net_state_dict, path_state_dict)
    # load
    # state_dict_load = torch.load(path_state_dict)
    #  # 推荐写法：无论在哪保存，加载都映射到 CPU
    state_dict_load = torch.load(path_state_dict, map_location="cpu")
    print("state_dict_load:\n{}".format(state_dict_load))
# =================================== 多gpu 保存
# flag = 0
flag = 1
if flag:
    # 用于获取当前系统中可用的 GPU 数量
    if torch.cuda.device_count() < 2:
        print("gpu数量不足，请到多gpu环境下运行")
        print(torch.cuda.device_count())
        import sys
        # 如果可用 GPU 数量小于 2，会打印提示信息并使用 sys.exit(0) 终止程序运行
        sys.exit(0)
    gpu_list = [0, 1, 2, 3]
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FooNet(neural_num=3, layers=3)
    # 使用 nn.DataParallel 对模型进行包装，使其支持多 GPU 并行计算。
    net = nn.DataParallel(net)
    net.to(device)
    # save
    net_state_dict = net.state_dict()
    path_state_dict = "code/chapter-7/model_in_multi_gpu02.pkl"
    torch.save(net_state_dict, path_state_dict)
# =================================== 多gpu 加载
flag = 0
# flag = 1
if flag:
    net = FooNet(neural_num=3, layers=3)
    path_state_dict = "code/chapter-7/model_in_multi_gpu.pkl"
    # 加载模型参数（state_dict），并使用 map_location="cpu" 将其映射到 CPU 上，避免没有 GPU 的情况下出错
    state_dict_load = torch.load(path_state_dict, map_location="cpu")
    print("state_dict_load:\n{}".format(state_dict_load))
    # net.load_state_dict(state_dict_load)
    # remove module.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict_load.items():
        namekey = k[7:] if k.startswith('module.') else k
        new_state_dict[namekey] = v
    print("new_state_dict:\n{}".format(new_state_dict))
    net.load_state_dict(new_state_dict)
