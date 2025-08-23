# -*- coding:utf-8 -*-
"""
@file name  : inference_main.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2023-03-02
@brief      : 推理脚本
"""
import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import platform
# 作用：在 Linux 下将 matplotlib 的图形后端改为 'Agg'，防止在无图形界面（如服务器、容器中）运行时出错。
# 'Agg' 是一个非交互式后端，用于生成图像而不弹出窗口
if platform.system() == 'Linux':
    matplotlib.use('Agg')


def get_args_parser(add_help=True):
    import argparse
    # argparse.ArgumentParser：构造命令行解析器。
# description="..."：帮助文档中的描述。
# add_help=True：是否添加 -h/--help 参数，默认为 True。
    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training", add_help=add_help)
    # --img-path：要分类的图像路径。
# 默认值是一个样例图像路径。
# 类型为字符串。
    parser.add_argument(
        "--img-path", default=  # r"bigdata/chapter-8/1/ChestXRay2017/chest_xray/test/NORMAL/IM-0001-0001.jpeg",
        # r"bigdata/chapter-8/1/ChestXRay2017/chest_xray/test/PNEUMONIA/person37_virus_82.jpeg",
        # r"bigdata/chapter-8/1/ChestXRay2017/chest_xray/test/PNEUMONIA/person78_bacteria_380.jpeg",
        # 同一个人108两张图片不同结果
        # r"bigdata/chapter-8/1/ChestXRay2017/chest_xray/train/PNEUMONIA/person108_virus_199.jpeg",
        r"bigdata/chapter-8/1/ChestXRay2017/chest_xray/train/PNEUMONIA/person108_virus_201.jpeg",
        type=str, help="dataset path")
    # --ckpt-path：模型的权重文件（checkpoint）路径。
# 默认值指向一个已保存的模型。
    parser.add_argument(
        "--ckpt-path", default=r"./Result/2025-08-02_22-42-38/checkpoint_best.pth", type=str, help="ckpt path")
    # --model：模型名称，支持：
# resnet50
# convnext
# convnext-tiny
# 用于根据名称选择模型架构。
    parser.add_argument("--model", default="convnext-tiny", type=str,
                        help="model name; resnet50/convnext/convnext-tiny")
    # --device：推理时使用的设备，cuda 或 cpu。
# 默认是 GPU。
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Use cuda or cpu Default: cuda)")
    # --output-dir：输出结果（如预测图、日志）的保存路径。
    parser.add_argument("--output-dir", default="./Result",
                        type=str, help="path to save outputs")

    return parser


def main(args):
    device = args.device
    path_img = args.img_path
    result_dir = args.output_dir
    # ------------------------------------ step1: img preprocess ------------------------------------

    normMean = [0.5]
    normStd = [0.5]
    input_size = (224, 224)
    normTransform = transforms.Normalize(normMean, normStd)

    valid_transform = transforms.Compose([
        # 缩放图像
        transforms.Resize(input_size),
        # 转为 Tensor，范围 [0, 1]。
        transforms.ToTensor(),
        # 归一化为 [-1, 1]。
        normTransform
    ])

# 加载图片并转为灰度图（单通道）
    img = Image.open(path_img).convert('L')
    img_tensor = valid_transform(img)
    img_tensor = img_tensor.to(device)

    # ------------------------------------ step2: model init ------------------------------------
    # 根据参数选择预训练模型
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(weights=None)
    elif args.model == 'convnext':
        model = torchvision.models.convnext_base(weights=None)
    elif args.model == 'convnext-tiny':
        model = torchvision.models.convnext_tiny(weights=None)
    else:
        print('unexpect model --> :{}'.format(args.model))

    model_name = model._get_name()

    if 'ResNet' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        model.conv1 = nn.Conv2d(1, 64, (7, 7), stride=(
            2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features  # 替换最后一层
        model.fc = nn.Linear(num_ftrs, 2)
    elif 'ConvNeXt' in model_name:
        # 替换第一层： 因为预训练模型输入是3通道，而本案例是灰度图，输入是1通道
        num_kernel = 128 if args.model == 'convnext' else 96
        # model.features[0][0] 是 ConvNeXt 中的第一层卷积（默认是 Conv2d(3, 96/128, 4, 4)）
        model.features[0][0] = nn.Conv2d(
            1, num_kernel, (4, 4), stride=(4, 4))  # convnext base/ tiny
        # 替换最后一层
        # onvNeXt 的 classifier 是一个 nn.Sequential 模块
        # 其中 [2] 是最后的全连接层，原始输出 1000 维；
# 改为输出 2个类，适用于你自己的分类任务
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, 2)

    state_dict = torch.load(args.ckpt_path)
    model_sate_dict = state_dict['model_state_dict']
    model.load_state_dict(model_sate_dict, strict=False)  # 模型参数加载

    model.to(device)
    # 设置为评估模式
    model.eval()
    # ------------------------------------ step3: inference ------------------------------------
    # ，关闭梯度计算，提高速度并节省显存。
    with torch.no_grad():
        ss = time.time()
        for i in range(20):
            s = time.time()
            # img_tensor.unsqueeze(dim=0) 是 PyTorch 中用于在指定维度插入一个维度的操作，通常用于将图像张量从 单张图像 转换成 批量输入
            img_tensor_batch = img_tensor.unsqueeze(dim=0)
            bs = 128
            # .repeat(a, b, c, d) 表示在每个维度上复制几次：
# 原始维度	含义	repeat(bs, 1, 1, 1) 的作用
# 1	batch size	将这 1 张图像复制成 128 张图像
# C	通道数	不复制（1次）
# H	高度	不复制
# W	宽度	不复制
            img_tensor_batch = img_tensor_batch.repeat(
                bs, 1, 1, 1)  # 128 or 100 or 1
            outputs = model(img_tensor_batch)
            # dim=1 表示在类别维度上进行 softmax（常见模型输出形状是 [batch_size, num_classes]）
            outputs_prob = torch.nn.functional.softmax(outputs, dim=1)
            # torch.max(..., 1) 沿着 dim=1 取最大值（即每行中最大值对应的索引）
            _, predicted = torch.max(outputs_prob.data, 1)
            # 把 predicted 从 GPU 移到 CPU（如果模型在 GPU 上）；
# 用 .data.numpy() 转为 NumPy 数组；
# [0] 取出第一个元素
            pred_idx = predicted.cpu().data.numpy()[0]
            time_c = time.time() - s
            # \r：回车符
# 作用：将光标移到当前行的开头，实现覆盖同一行打印的效果。
# classes[pred_idx]	预测的类别标签（如 'cat'、'dog'）
# time_c	当前 batch 的推理耗时（单位：秒）
# 1*bs/time_c	当前吞吐量：batch 中有 bs 张图像，除以耗时得到每秒处理帧数 FPS
            print('\r', 'model predict: {},  speed: {:.4f} s/batch, Throughput: {:.0f} frame/s'.format(
                classes[pred_idx], time_c, 1*bs/time_c), end='')
        print('\n', time.time()-ss)

    # ------------------------------------ step4: visualization ------------------------------------
    plt.imshow(img, cmap='Greys_r')
    plt.title("predict:{}".format(classes[pred_idx]))
    # {:.1%}	转换为百分比并保留 1 位小数，比如 0.876 → 87.6%
    # outputs_prob[0, pred_idx]	预测的类别的概率（属于该类的置信度）
    # bbox=dict(fc='yellow')	给文字添加一个黄色的背景框，便于在图像上识别
    # dict(...) 是传入一个样式字典；
# fc 是 facecolor 的缩写，表示背景色（填充色）；
# 'yellow' 是背景色为黄色，也可以写成 'white', 'black', 'red' 等
    plt.text(50, 50, "predict: {}, probability: {:.1%}".format(
        classes[pred_idx], outputs_prob.cpu().data.numpy()[0, pred_idx]), bbox=dict(fc='yellow'))
    plt.show()


classes = ["NORMAL", "PNEUMONIA"]

if __name__ == "__main__":
    # get_args_parser() 是你自定义的函数（应该返回一个 argparse.ArgumentParser() 实例）；
    # .parse_args() 会从命令行读取参数（如 --batch-size 64 --model convnext）并返回一个包含参数的对象；
    # args 就是命令行参数集合
    args = get_args_parser().parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name()
    print('gpu name: {}'.format(gpu_name))
    # 调用你定义的 main(args) 函数，正式启动主逻辑；
# 把上面处理好的命令行参数传入，便于在主函数中统一使用
    main(args)
