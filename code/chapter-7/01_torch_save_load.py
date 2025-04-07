# -*- coding:utf-8 -*-
"""
@file name  : 01_torch_save_load.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-022
@brief      : 模型保存与加载
"""
import torch
import torchvision.models as models
from torchinfo import summary
if __name__ == "__main__":
    # ========================================= torch.save ==============================================
    # path_state_dict = "resnet50_state_dict_2025.pth"
    # resnet_50 = models.resnet50(pretrained=False)
    # # 模拟训练，将模型参数进行修改
    # # 打印 conv1（第一个卷积层）的第一个卷积核的所有参数。
    # print("训练前: ", resnet_50.conv1.weight[0, ...])
    # for p in resnet_50.parameters():
    #     p.data.fill_(2025)
    # print("训练后: ", resnet_50.conv1.weight[0, ...])
    # # 保存模型参数
    # net_state_dict = resnet_50.state_dict()
    # torch.save(net_state_dict, path_state_dict)
    # # ========================================= torch.load ==============================================
    # resnet_50_new = models.resnet50(pretrained=False)
    # print("初始化: ", resnet_50_new.conv1.weight[0, ...])
    # # 加载权重，
    # state_dict = torch.load(path_state_dict)
    # # 权重应用到新模型中
    # resnet_50_new.load_state_dict(state_dict)
    # print("加载后: ", resnet_50_new.conv1.weight[0, ...])
    # ========================================= torchvision scripts ==============================================
    # https://github.com/pytorch/vision/blob/fa347eb9f38c1759b73677a11b17335191e3f602/references/classification/train.py
    # 这是一个字典，保存了训练状态的多个组件
    # 键名	内容	说明
    # "model"	模型的权重参数	model_without_ddp 是去掉了 DDP 封装的模型（通常用于多GPU训练）
    # "optimizer"	优化器的状态（如动量、学习率）	便于训练恢复后继续优化器状态
    # "lr_scheduler"	学习率调度器的状态	保证学习率变化曲线连续性
    # "epoch"	当前训练到的 epoch 数（整数）	下次训练从下一个 epoch 开始
    checkpoint = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
    }
    # 将字典序列化保存为文件，比如：model_20.pth
    path_save = "model_{}.pth".format(epoch)
    torch.save(checkpoint, path_save)
    # ========================================= resume ==============================================
    # resume
    # 加载 checkpoint，指定 map_location="cpu" 意味着无论是否在 GPU 上保存，都会加载到 CPU（可之后再转到 GPU）
    checkpoint = torch.load(path_save, map_location="cpu")
    # 分别恢复：
# 模型参数
# 优化器状态
# 学习率调度器状态
# start_epoch: 让训练从下一个 epoch 继续
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    start_epoch = checkpoint["epoch"] + 1
