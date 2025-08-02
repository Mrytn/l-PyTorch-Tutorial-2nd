# -*- coding:utf-8 -*-
"""
@file name  : 01_resnet_export_onnx.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-06-02
@brief      : resnet50 onnx导出
"""
import os.path

import torchvision
import torch
import torch
import torchvision
import torch.nn as nn

ckpt_path = r"./Result/2023-09-25_22-09-35/checkpoint_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载一个未预训练的 ResNet50 模型。
# 默认的 ResNet50 结构是：
# 输入通道数为 3（彩色图像）
# 最后全连接层输出为 1000 类（ImageNet）
model = torchvision.models.resnet50(pretrained=False)
# 修改输入层：从3通道改为1通道
# 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# 获取原先的 fc（全连接层）输入特征数量（num_ftrs = 2048）
num_ftrs = model.fc.in_features  # 替换最后一层
# 将最后的输出层改为 Linear(2048, 2)，即输出为 2 类，用于二分类任务
model.fc = nn.Linear(num_ftrs, 2)

# 从本地路径 ckpt_path 加载模型权重文件
# map_location=device 确保权重加载到当前使用的 CPU 或 GPU 上
state_dict = torch.load(ckpt_path, map_location=device)
# 从 checkpoint 中提取 'model_state_dict'，并加载到模型中
model_sate_dict = state_dict['model_state_dict']
model.load_state_dict(model_sate_dict)  # 模型参数加载
# 设置为 eval 模式 增强鲁棒性
# 切换模型到 eval() 模式，影响如下层行为：
# Dropout 层会关闭（不再随机丢弃神经元）
# BatchNorm 层会使用移动平均和方差而非当前批次数据
model.eval()

if __name__ == '__main__':
    # opset_version: ONNX 操作集版本（建议 11~17，13 通用性好）
    op_set = 13
    # 准备 dummy 输入张量（假数据）
    dummy_data = torch.randn((1, 3, 224, 224))

    # 固定 batch = 1
    # 导出模型为 ONNX 格式
    out_dir = os.path.dirname(ckpt_path)
    path_out = os.path.join(out_dir, "resnet50_bs_1.onnx")
    torch.onnx.export(model, (dummy_data), path_out,
                      opset_version=op_set, input_names=['input'],  output_names=['output'])



