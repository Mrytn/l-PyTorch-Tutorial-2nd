import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


def image_write(path_A, path_B, path_AB):
    # 参数 1 表示以彩色模式读取图像
    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    # 在指定轴上拼接数组
    # axis=1 → 表示按列（水平）拼接
    # 结果 im_AB 是一个新的 ndarray，宽度 = im_A.width + im_B.width，高度 = im_A.height（要求两张图高度相同，否则会报错）
    im_AB = np.concatenate([im_A, im_B], 1)
    # 将 ndarray 图像写入磁盘文件
    cv2.imwrite(path_AB, im_AB)


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true',default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))

splits = os.listdir(args.fold_A)
# 如果没有禁用多进程，则创建一个 multiprocessing.Pool 对象（CPU 核心池）。
# 可以让多个 CPU 核心并行处理图片，加快速度。
if not args.no_multiprocessing:
    pool=Pool()

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    # 获取 A 目录下的所有图片列表
    img_list = os.listdir(img_fold_A)
    # 若 use_AB 为真，则只保留文件名中包含 _A. 的图像
    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(args.fold_AB, sp)
    # 限制实际处理数量，防止太多。
# 如果输出目录不存在，则创建
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        # 获取当前第 n 张图片的路径
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        # 如果使用 _A/_B 命名规则，则将 0001_A.jpg → 0001_B.jpg。
# 否则直接假定文件名一致（两边同名）
        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)
        # 确保 A、B 两张图都存在。
# 输出文件名去掉 _A 后缀，比如 0001.jpg。
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_AB = os.path.join(img_fold_AB, name_AB)
            # 若启用多进程，使用 pool.apply_async() 异步地调用 image_write()。
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_AB))
            else:
                # 否则直接在主进程执行相同逻辑。
                im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
                im_AB = np.concatenate([im_A, im_B], 1)
                cv2.imwrite(path_AB, im_AB)
if not args.no_multiprocessing:
    # 关闭进程池（等待所有任务结束）
    # close()：不再接受新的任务。
# join()：等待所有任务执行完毕。
    pool.close()
    pool.join()
