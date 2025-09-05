import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net
'''你这段代码实现了一个 ReID 特征提取器 (Extractor)，主要功能是加载预训练的 ReID 网络模型，然后对输入图像进行预处理，最后输出特征向量'''
class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        # 模型加载
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        # 预处理
        # 输入是原始 BGR 或 RGB 图像 crop。
# 转 float32，缩放到 [0, 1]。
# resize 到 (64, 128)（与 Market1501 数据集一致）
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])



    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            # 你这里用了 cv2.resize 返回的是 np.float32，再传给 transforms.ToTensor() 时会报错（因为 ToTensor() 期望 PIL.Image 或 numpy.uint8）。
# 更常见的写法是 直接用 torchvision.transforms.Resize。
# ToTensor() + cv2.resize 不太兼容
# 你这里 _resize 之后得到的是 np.float32 的 [H,W,C] 数组，范围 [0,1]，再传 ToTensor() 会多做一次除以 255 的缩放（得到 [0, 0.0039] 范围），结果就错了
# 改成im = cv2.resize(im, size)  # 直接 resize，保持 0~255
    # return im[:, :, ::-1]  # 如果输入是 BGR 转 RGB（PyTorch 默认 RGB）
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        # 前向推理
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    # cv2.imread 默认读取的是 BGR 格式
# cv2.imread("demo.jpg")[:, :, (2,1,0)] 就是把通道顺序调换成 RGB等价于
# cv2.imread("demo.jpg")[:, :, ::-1]
# 第 2 个通道 → R
# 第 1 个通道 → G
# 第 0 个通道 → B
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

