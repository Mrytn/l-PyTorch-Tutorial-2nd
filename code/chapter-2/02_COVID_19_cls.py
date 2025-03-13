# -*- coding:utf-8 -*-
"""
@file name  : 02_COVID_19_cls.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2021-12-28
@brief      : 新冠肺炎X光分类 demo，极简代码实现深度学习模型训练，为后续核心模块讲解，章节内容讲解奠定框架性基础。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


def main():
    # 思考：如何实现你的模型训练？第一步干什么？第二步干什么？...第n步...
    # step 1/4 : 数据模块：构建dataset, dataloader，实现对硬盘中数据的读取及设定预处理方法
    # step 2/4 : 模型模块：构建神经网络，用于后续训练
    # step 3/4 : 优化模块：设定损失函数与优化器，用于在训练过程中对网络参数进行更新
    # step 4/4 : 迭代模块: 循环迭代地进行模型训练，数据一轮又一轮的喂给模型，不断优化模型，直到我们让它停止训练
    # step 1/4 : 数据模块
    class COVID19Dataset(Dataset):
        def __init__(self, root_dir, txt_path, transform=None):
            """
            获取数据集的路径、预处理的方法
            """
            self.root_dir = root_dir
            self.txt_path = txt_path
            self.transform = transform
            self.img_info = []  # [(path, label), ... , ]
            self.label_array = None
            self._get_img_info()

        def __getitem__(self, index):
            """
            输入标量index, 从硬盘中读取数据，并预处理，to Tensor
            :param index:
            :return:
            """
            # # 获取图片路径和标签
            path_img, label = self.img_info[index]
            # Image.open(path_img): 读取图片。
            # .convert('L'): 转换为灰度图像。
            img = Image.open(path_img).convert('L')
            # 如果 transform 不是 None，则对 img 进行数据预处理，如调整大小、转换为 Tensor
            if self.transform is not None:
                img = self.transform(img)
            return img, label
        # 如果数据集为空，抛出异常并打印错误信息。
# 返回 self.img_info 的长度，即样本数量。

        def __len__(self):
            if len(self.img_info) == 0:
                raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                    self.root_dir))  # 代码具有友好的提示功能，便于debug
            return len(self.img_info)

        def _get_img_info(self):
            """
            实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
            path, label
            :return:
            """
            # 读取txt，解析txt
            # "r"只读
            with open(self.txt_path, "r") as f:
                txt_data = f.read().strip()
                txt_data = txt_data.split("\n")
            # i.split()：对每一行 i 按空格分割成列表。例如，如果 i = "image1.jpg 123 0"，那么 i.split() 结果是 ["image1.jpg", "123", "0"]。
            # i.split()[0]：取出文件名（image1.jpg）。
            # i.split()[2]：取出标签值（如 0），并转换为 int 类型。
            # os.path.join(self.root_dir, i.split()[0])：将 self.root_dir（图片的根目录）和 i.split()[0]（文件名）拼接，形成完整的图片路径。
            self.img_info = [(os.path.join(self.root_dir, i.split()[0]), int(i.split()[2]))
                             for i in txt_data]
    # you can download the datasets from
    # https://pan.baidu.com/s/18BsxploWR3pbybFtNsw5fA  code：pyto
    # path to datasets——covid-19-demo
    # r 代表 原始字符串（raw string），用于避免 转义字符 造成的问题。就像java里的\一样
    # 在 Python 中，普通字符串 会解析转义字符（如 \n 表示换行，\t 表示制表符）。原始字符串（raw string） 在字符串前加上 r 或 R，可以防止转义字符生效，使其按字面意思解析。
    root_dir = r"data/datasets/covid-19-demo"
    img_dir = os.path.join(root_dir, "imgs")
    path_txt_train = os.path.join(root_dir, "labels", "train.txt")
    path_txt_valid = os.path.join(root_dir, "labels", "valid.txt")
    transforms_func = transforms.Compose([
        # 将图片缩放为 8x8 大小。
        transforms.Resize((8, 8)),
        # 转换为 PyTorch Tensor，并归一化到 [0,1]
        transforms.ToTensor(),
    ])
    # 使用 COVID19Dataset 读取训练集，并应用 transforms_func 预处理数据。
    train_data = COVID19Dataset(
        root_dir=img_dir, txt_path=path_txt_train, transform=transforms_func)
    valid_data = COVID19Dataset(
        root_dir=img_dir, txt_path=path_txt_valid, transform=transforms_func)
    #  用于批量加载数据并自动打乱,batch_size=2：每次训练迭代读取 2 个样本。
    train_loader = DataLoader(dataset=train_data, batch_size=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=2)
    # step 2/4 : 模型模块
    # 这行代码明确声明 TinnyCNN 继承自 torch.nn.Module。

    class TinnyCNN(nn.Module):
        def __init__(self, cls_num=2):
            # super(ClassName, self)
            # super(TinnyCNN, self) 代表调用 TinnyCNN 的父类（即 nn.Module） 的方法
            super(TinnyCNN, self).__init__()
            # 定义了一个 Conv2d 卷积层（输入通道数 1，输出通道数 1，卷积核大小 3x3）
            # in_channels：输入通道数（输入图像的深度）
            # out_channels：输出通道数（卷积核个数，决定输出的深度）
            # kernel_size：卷积核的大小（(height, width)）
            self.convolution_layer = nn.Conv2d(1, 1, kernel_size=(3, 3))
            # 定义了一个 Linear 全连接层（输入 36，输出 cls_num，默认为 2
            # nn.Linear 是 PyTorch 中的 全连接层（又称线性变换层），用于对输入数据执行线性变换
            self.fc = nn.Linear(36, cls_num)

        def forward(self, x):
            # 卷积操作
            x = self.convolution_layer(x)
            # x.size(0) 获取 批量大小（batch size）
# view() 调整 x 的形状（相当于 reshape）
            x = x.view(x.size(0), -1)
            # 全连接输出
            out = self.fc(x)
            return out
        # 2 传递给 cls_num，即2分类任务
    model = TinnyCNN(2)
    # step 3/4 : 优化模块
    # 交叉熵损失函数
    loss_f = nn.CrossEntropyLoss()
    # 选择优化器为 随机梯度下降（SGD）。该优化器会优化 model.parameters()（即模型的权重和偏置）。
# lr=0.1：设置学习率为 0.1。
# momentum=0.9：设置动量为 0.9，动量有助于加速 SGD 收敛。
# model.parameters() 返回的是 模型中所有可训练的参数
# weight_decay 是权重衰减（weight decay）的参数，也称为 L2 正则化。权重衰减的作用是防止模型过拟合，它会在损失函数中添加一个正则化项，使得模型的参数值不会过大。在这个例子中，权重衰减的值被设置为 5e-4
# optim.SGD 不直接支持 L1 正则化
# 动量能使优化算法在优化过程中有一定的惯性，避免了过于细致的调整，从而减少了在局部最小值附近的震荡。
# 在很多情况下，动量可以显著加速训练的收敛速度，尤其是在复杂的深度神经网络中。
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    # StepLR：使用学习率调度器。在每个 step_size 轮次后，学习率会按 gamma 缩小。
# gamma=0.1：每 50 个 epoch 后，学习率减少为原来的 10%。
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=50)
    # step 4/4 : 迭代模块
    for epoch in range(100):
        # 训练集训练
        # model.train()：设置模型为训练模式，这样 dropout 和 batch norm 等层会启用
        model.train()
        for data, labels in train_loader:
            # forward & backward
            # 将输入数据传递给模型，得到模型的输出 outputs。这是模型的预测结果。
            outputs = model(data)
            # 清空优化器的梯度缓存。每次反向传播前，必须先清空上一轮的梯度
            optimizer.zero_grad()
            # loss 计算计算 损失
            loss = loss_f(outputs, labels)
            # 进行反向传播，计算梯度
            # # backward() 过程：
            # 计算 loss 对 outputs 的梯度。
            # 计算 outputs 对模型参数的梯度。
            # 将计算得到的梯度存储在 param.grad。。。
            # param.grad 是 PyTorch 中的一个 Tensor，它存储了某个模型参数（如权重 W 或偏置 b）的 梯度（gradient）。
            loss.backward()
            # 通过梯度更新模型的权重。
            # 内部实现param.data = param.data - lr * param.grad
            optimizer.step()
            # 计算分类准确率
            # 张量的第 1 维（列维度）上寻找最大值，并返回最大值和对应的索引。
            _, predicted = torch.max(outputs.data, 1)
            # 计算预测正确的样本数量，同样使用布尔值的 sum() 方法
            correct_num = (predicted == labels).sum()
            # 计算验证集的准确率，验证集的准确率也是正确样本数除以总样本数。
            acc = correct_num / labels.shape[0]
            # 打印验证集信息，输出当前 epoch 的验证损失和准确率。
            # :.2f 代表 格式化为浮点数，保留2位小数
            # :.0% 代表 以百分比形式显示，并且 不保留小数（0 位）
            print("Epoch:{} Train Loss:{:.2f} Acc:{:.0%}".format(epoch, loss, acc))
            # print(predicted, labels)
        # 验证集验证
        # model.eval()：将模型切换为评估模式。此时，dropout 和 batch norm 等层会被禁用
        model.eval()
        for data, label in valid_loader:
            # forward
            # 通过模型进行前向传播，得到预测输出。
            outputs = model(data)
            # loss 计算
            loss = loss_f(outputs, labels)
            # 计算分类准确率
            _, predicted = torch.max(outputs.data, 1)
            correct_num = (predicted == labels).sum()
            acc_valid = correct_num / labels.shape[0]
            print("Epoch:{} Valid Loss:{:.2f} Acc:{:.0%}".format(
                epoch, loss, acc_valid))
        # 添加停止条件
        # 如果验证集准确率达到了 100%（即 acc_valid == 1），则终止训练循环，避免过拟合。
        if acc_valid == 1:
            break
        # 学习率调整
        # 每经过一定的 epoch 数（step_size=50），调整学习率。此处会将学习率缩小 gamma=0.1 倍
        scheduler.step()


# 这一行代码是 Python 中的标准写法，用于确保 Python 脚本在被直接运行时执行 main()，而在被导入时不会执行 main()。
if __name__ == "__main__":
    main()
