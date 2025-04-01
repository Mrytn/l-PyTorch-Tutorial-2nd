# -*- coding:utf-8 -*-
"""
@file name  : 01_tensorboard.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-06-06
@brief      : tensorboard 安装与使用
"""
import torch
import torch.nn as nn
if __name__ == "__main__":
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    from torch.utils.tensorboard import SummaryWriter
    log_dir = BASE_DIR  # 即test_tensorboard.py文件所在目录
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_test_tensorboard")
    # writer = SummaryWriter(comment="test01", filename_suffix="_test_tensorboard")
    x = range(100)
    for i in x:
        # 记录两条标量数据 (add_scalar)
        writer.add_scalar('y=3x', i * 3, i)
        writer.add_scalar('y=pow(2, x)', 2 ** i, i)
    # 关闭记录器 (writer.close())，确保数据写入完成
    writer.close()
