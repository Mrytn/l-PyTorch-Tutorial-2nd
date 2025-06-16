# -*- coding:utf-8 -*-
"""
@file name  : resnet50_speed_test
@author     : TingsongYu
@date       : 2023-05-24
@brief      : resnet50 推理速度评估脚本
"""
import time
# 保存和加载 Python 对象（如模型日志等）
import pickle
import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np
import matplotlib
# 检查操作系统类型
import platform

if platform.system() == 'Linux':
    matplotlib.use('Agg')


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)
    # 表示是否启用 混合精度训练（fp16）。
# 如果添加该参数，则为 True；否则为 False。
    parser.add_argument('--half', action='store_true', default=False)

    return parser

def main(args):
    # ------------------------------------ step2: model init ------------------------------------
    # weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    # 这是一个枚举类型，表示 加载官方提供的预训练权重。
    # ResNet50_Weights 是 torchvision.models 中的一个权重配置类。
    # IMAGENET1K_V1 表示这些权重是在 ImageNet-1K 数据集（1000类）上训练得到的第一版本
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    # 如果用户传入了 --half，则将模型转为半精度（float16）
    if args.half:
        model.half()
    model.to(device)
    model.eval()
    # ------------------------------------ step3: inference ------------------------------------
    # 构造不同的 batch size 列表： [4, 8, 16, 32, 64, 128]，用于测试不同输入规模下的性能。
    bs_list = list(map(lambda n: 2**n, range(2, 8)))
    # 推理时间
    speed_list = []
    # 吞吐量
    throughput_list = []
    # 每个 batch size 都会 重复推理 10 次，用于测量平均值，减少偶然波动。
    repeat_num = 10
    for bs in bs_list:
        img_tensor_batch = torch.randn(bs, 3, 224, 224)
        img_tensor_batch = img_tensor_batch.to(device)
        if args.half:
            img_tensor_batch = img_tensor_batch.half()
        # 推理并计时
        with torch.no_grad():
            s = time.time()
            for i in range(repeat_num):
                _ = model(img_tensor_batch)
            time_c = time.time() - s
            # 单张图片的平均推理时间，单位：ms
            speed = time_c/(bs*repeat_num)*1000  # ms
            # 总推理时间 / 图片总数 × 1000 → 得到每张图像平均耗时（毫秒）。
            throughput = (bs*repeat_num)/time_c
            print('bs: {} speed: {:.4f} s/batch, speed:{:.4f} ms/frame Throughput: {:.0f} frame/s'.format(
                bs, time_c/repeat_num, speed, throughput))

        speed_list.append(speed)
        throughput_list.append(throughput)

    # 绘图,可视化不同 batch size 下的推理速度和吞吐量
    plt.subplot(2, 1, 1)
    plt.plot(bs_list, speed_list, marker='o', linestyle='--')
    for a, b in zip(bs_list, speed_list):
        # 在点的正上方绘制文本标签，显示速度数值，保留两位小数。
        # va='bottom' 表示文本在点的上方对齐。
        plt.text(a, b, f'{b:.2f}', ha='center', va='bottom', fontsize=10)
    plt.title('Speed ms/frame')

    # 绘制第二幅图
    plt.subplot(2, 1, 2)
    plt.plot(bs_list, throughput_list, marker='o', linestyle='--')
    for a, b in zip(bs_list, throughput_list):
        plt.text(a, b, f'{b:.2f}', ha='center', va='bottom', fontsize=10)
    plt.title('Throughput frame/s')
    # 整体图表的主标题（super title），包含 GPU 名称、输入图像大小、是否使用半精度信息
    plt.suptitle(f'Resnet50 speed test in {gpu_name} imgsize 224-is half {args.half}')
    # 整体图表的主标题（super title），包含 GPU 名称、输入图像大小、是否使用半精度信息
    plt.subplots_adjust(hspace=0.5)
    # plt.show()
    # 保存图像为 PNG 文件，文件名包含是否使用半精度的标志
    plt.savefig(f'resnet50-speed-test-half-is-{args.half}.png')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name()
    print('gpu name: {}'.format(gpu_name))
    main(args)
