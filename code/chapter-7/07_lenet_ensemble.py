# -*- coding:utf-8 -*-
"""
@file name  : 07_lenet_ensemble.py
@author     : TingsongYu https://github.com/TingsongYu
@date       : 2022-07-06
@brief      : 编写自定义的模型集成类
"""
import os
import torch
import torchvision
import torchmetrics
import torch.nn as nn
import my_utils as utils
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchensemble.utils import set_module
from torchensemble.voting import VotingClassifier
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training", add_help=add_help)
    parser.add_argument(
        "--data-path", default=r"F:\pytorch-tutorial-2nd\data\datasets\cifar10-office", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet8",
                        type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str,
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=128, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=200, type=int,
                        metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="SGD", type=str, help="optimizer")
    parser.add_argument("--random-seed", default=42,
                        type=int, help="random seed")
    parser.add_argument("--lr", default=0.1, type=float,
                        help="initial learning rate")
    parser.add_argument("--momentum", default=0.9,
                        type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-step-size", default=80, type=int,
                        help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float,
                        help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq", default=80,
                        type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./Result",
                        type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str,
                        help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int,
                        metavar="N", help="start epoch")
    return parser
def main(args):
    device = args.device
    data_dir = args.data_path
    result_dir = args.output_dir
    # ------------------------------------  log ------------------------------------
    logger, log_dir = utils.make_logger(result_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # ------------------------------------ step1: dataset ------------------------------------
    normMean = [0.4948052, 0.48568845, 0.44682974]
    normStd = [0.24580306, 0.24236229, 0.2603115]
    normTransform = transforms.Normalize(normMean, normStd)
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normTransform
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    # root变量下需要存放cifar-10-python.tar.gz 文件
    # cifar-10-python.tar.gz可从 "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" 下载
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, transform=train_transform, download=True)
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, transform=valid_transform, download=True)
    # 构建DataLoder
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, num_workers=args.workers)
    # ------------------------------------ tep2: model ------------------------------------
    model_base = utils.resnet20()
    # model_base = utils.LeNet5()
    model = MyEnsemble(estimator=model_base, n_estimators=3, logger=logger, device=device, args=args,
                       classes=classes, writer=writer, save_dir=log_dir)
    model.set_optimizer(args.opt, lr=args.lr, weight_decay=args.weight_decay)
    model.fit(train_loader, test_loader=valid_loader, epochs=args.epochs)
# 自定义模型集成类分类器
class MyEnsemble(VotingClassifier):
    def __init__(self, **kwargs):
        # logger, device, args, classes, writer
        super(VotingClassifier, self).__init__(
            kwargs["estimator"], kwargs["n_estimators"])
        self.logger = kwargs["logger"]
        self.writer = kwargs["writer"]
        self.device = kwargs["device"]
        self.args = kwargs["args"]
        self.classes = kwargs["classes"]
        self.save_dir = kwargs["save_dir"]
    @staticmethod
    def save(model, save_dir, logger):
        """Implement model serialization to the specified directory."""
        # 如果没有提供路径，就默认保存在当前目录；若路径不存在，就创建它
        if save_dir is None:
            save_dir = "./"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # Decide the base estimator name
        # 判断是类还是对象
        if isinstance(model.base_estimator_, type):
            base_estimator_name = model.base_estimator_.__name__
        else:
            base_estimator_name = model.base_estimator_.__class__.__name__
        # {Ensemble_Model_Name}_{Base_Estimator_Name}_{n_estimators}
        # 拼接文件名：
        filename = "{}_{}_{}_ckpt.pth".format(
            type(model).__name__,
            base_estimator_name,
            model.n_estimators,
        )
        # The real number of base estimators in some ensembles is not same as
        # `n_estimators`.
        # 保存内容打包：
        state = {
            "n_estimators": len(model.estimators_),
            "model": model.state_dict(),
            "_criterion": model._criterion,
        }
        # 真的保存到文件：
        save_dir = os.path.join(save_dir, filename)
        logger.info("Saving the model to `{}`".format(save_dir))
        # Save
        torch.save(state, save_dir)
        return
    # 集成学习训练框架中 fit() 方法的完整实现。它的作用是训练多个子模型（即基模型），并保存验证集上表现最好的集成模型
# train_loader: 训练数据加载器
# epochs: 训练轮数
# log_interval: 日志打印间隔（虽然代码中并未使用）
# test_loader: 验证集加载器
# save_model: 是否保存模型
# save_dir: 模型保存路径

    def fit(self, train_loader, epochs=100, log_interval=100, test_loader=None, save_model=True, save_dir=None,):
        # 模型、优化器、学习率调整器、评估器 列表创建
        estimators = []
        # 创建多个基模型
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())
        # 分别为每个基模型设置独立的优化器和学习率调度器。
        optimizers = []
        schedulers = []
        for i in range(self.n_estimators):
            # 为第 i 个估计器（模型）设置优化器，然后将其添加到 optimizers 列表中。
            optimizers.append(set_module.set_optimizer(
                estimators[i], self.optimizer_name, **self.optimizer_args))
            # MultiStepLR 按设定 epoch 降低学习率，例如在第 100 和 150 个 epoch 减小。
            scheduler_ = torch.optim.lr_scheduler.MultiStepLR(
                optimizers[i], milestones=[100, 150], gamma=self.args.lr_gamma)  # 设置学习率下降策略
            # scheduler_ = torch.optim.lr_scheduler.StepLR(optimizers[i], step_size=self.args.lr_step_size,
            #                                             gamma=self.args.lr_gamma)  # 设置学习率下降策略
            schedulers.append(scheduler_)
        acc_metrics = []
        for i in range(self.n_estimators):
            acc_metrics.append(torchmetrics.Accuracy(
                task='multiclass', num_classes=10))
        self._criterion = nn.CrossEntropyLoss()
        # epoch循环迭代
        best_acc = 0.
        for epoch in range(epochs):
            # training
            for model_idx, (estimator, optimizer, scheduler) in enumerate(zip(estimators, optimizers, schedulers)):
                # 调用你自己封装的训练函数返回损失和准确率。
                loss_m_train, acc_m_train, mat_train = \
                    utils.ModelTrainerEnsemble.train_one_epoch(
                        train_loader, estimator, self._criterion, optimizer, scheduler, epoch,
                        self.device, self.args, self.logger, self.classes)
                # 学习率更新
                scheduler.step()
                # 记录
                self.writer.add_scalars('Loss_group', {'train_loss_{}'.format(model_idx):
                                                       loss_m_train.avg}, epoch)
                self.writer.add_scalars('Accuracy_group', {'train_acc_{}'.format(model_idx):
                                                           acc_m_train.avg}, epoch)
                self.writer.add_scalar(
                    'learning rate', scheduler.get_last_lr()[0], epoch)
            # validate
            # 验证集评估整个集成模型
            # 用多个子模型评估验证集，可能是投票结果或者平均预测。
            loss_valid_meter, acc_valid, top1_group, mat_valid = \
                utils.ModelTrainerEnsemble.evaluate(
                    test_loader, estimators, self._criterion, self.device, self.classes)
            # 日志
            self.writer.add_scalars('Loss_group', {'valid_loss':
                                                   loss_valid_meter.avg}, epoch)
            self.writer.add_scalars('Accuracy_group', {'valid_acc':
                                                       acc_valid*100}, epoch)
            self.logger.info(
                'Epoch: [{:0>3}/{:0>3}]  '
                'Train Loss avg: {loss_train:>6.4f}  '
                'Valid Loss avg: {loss_valid:>6.4f}  '
                'Train Acc@1 avg:  {top1_train:>7.2f}%   '
                'Valid Acc@1 avg: {top1_valid:>7.2%}    '
                'LR: {lr}'.format(
                    epoch, self.args.epochs, loss_train=loss_m_train.avg, loss_valid=loss_valid_meter.avg,
                    top1_train=acc_m_train.avg, top1_valid=acc_valid, lr=schedulers[0].get_last_lr()[0]))
            for model_idx, top1_meter in enumerate(top1_group):
                self.writer.add_scalars('Accuracy_group', {'valid_acc_{}'.format(
                    model_idx): top1_meter.compute()*100}, epoch)
            if acc_valid > best_acc:
                best_acc = acc_valid
                self.estimators_ = nn.ModuleList()
                self.estimators_.extend(estimators)
                if save_model:
                    self.save(self, self.save_dir, self.logger)
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    utils.setup_seed(args.random_seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
