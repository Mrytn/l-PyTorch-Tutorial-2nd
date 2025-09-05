import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os

from model import Net

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
args = parser.parse_args()

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loader
root = args.data_dir
query_dir = os.path.join(root,"query")
gallery_dir = os.path.join(root,"gallery")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
queryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(query_dir, transform=transform),
    batch_size=64, shuffle=False
)
galleryloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(gallery_dir, transform=transform),
    batch_size=64, shuffle=False
)

# net definition
net = Net(reid=True)
assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
print('Loading from checkpoint/ckpt.t7')
checkpoint = torch.load("./checkpoint/ckpt.t7")
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict, strict=False)
net.eval()
net.to(device)

# compute features
# 建立空的 Tensor 来存放所有 query / gallery 的特征和标签
query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
gallery_features = torch.tensor([]).float()
gallery_labels = torch.tensor([]).long()

with torch.no_grad():
    # 遍历 queryloader，送入模型 net 得到特征。
# features shape 一般是 (batch_size, feature_dim)。
# 累加到 query_features 里。
# 同步保存 labels。
    for idx,(inputs,labels) in enumerate(queryloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        query_features = torch.cat((query_features, features), dim=0)
        query_labels = torch.cat((query_labels, labels))
    # 提取 Gallery 特征
    for idx,(inputs,labels) in enumerate(galleryloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        gallery_features = torch.cat((gallery_features, features), dim=0)
        gallery_labels = torch.cat((gallery_labels, labels))
# 所有 gallery 标签统一减去 2。
# 👉 这说明你的 dataset label 编码里，gallery 的 ID 起始值比 query 高 2，这里是做 对齐处理。
gallery_labels -= 2

# save features
features = {
    "qf": query_features,
    "ql": query_labels,
    "gf": gallery_features,
    "gl": gallery_labels
}
# 把 query 和 gallery 的特征与标签打包成字典。
# 保存成 features.pth，方便后续检索 / 评估使用。
torch.save(features,"features.pth")