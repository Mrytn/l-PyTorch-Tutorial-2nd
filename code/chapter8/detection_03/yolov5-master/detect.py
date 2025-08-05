# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
import argparse
import os
import platform
import sys
from pathlib import Path
from tqdm import tqdm

import torch
# 绝对路径
FILE = Path(__file__).resolve()
# 父目录
ROOT = FILE.parents[0]  # YOLOv5 root directory
# 如果当前路径不在python运行环境里（sys.path）
if str(ROOT) not in sys.path:
    # 添加根目录到系统路径
    sys.path.append(str(ROOT))  # add ROOT to PATH
# 将 ROOT 转换为相对于当前工作目录（Path.cwd()）的相对路径，然后再转成一个 Path 对象。
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 在推理时关闭梯度计算，节省显存、加快速度；但在训练时保留梯度
@smart_inference_mode()
# ---------1.载入参数---------
def run(
        # 模型权重文件的路径，默认为YOLOv5s的权重文件路径
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        # 输入图像或视频的路径或URL，或者使用数字0指代摄像头，默认为YOLOv5自带的测试图像文件夹。
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        # 数据集文件的路径，默认为COCO128数据集的配置文件路径。
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        # 输入图像的大小，默认为640x640
        imgsz=(640, 640),  # inference size (height, width)
        # 置信度阈值，默认为0.25。
        conf_thres=0.25,  # confidence threshold
        # 非极大值抑制的IoU阈值，默认为0.45。
        iou_thres=0.45,  # NMS IOU threshold
        # 每张图像的最大检测框数，默认为1000。
        max_det=1000,  # maximum detections per image
        # 使用的设备类型，默认为空，表示自动选择最合适的设备。
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # 是否在屏幕上显示检测结果，默认为False。
        view_img=False,  # show results
        # 是否将检测结果保存为文本文件，默认为False。
        save_txt=False,  # save results to *.txt
        # 是否在保存的文本文件中包含置信度信息，默认为False。
        save_conf=False,  # save confidences in --save-txt labels
        # 是否将检测出的目标区域保存为图像文件，默认为False。
        save_crop=False,  # save cropped prediction boxes
        # 是否不保存检测结果的图像或视频，默认为False。
        nosave=False,  # do not save images/videos
        # 指定要检测的目标类别，默认为None，表示检测所有类别。
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        # 是否使用类无关的非极大值抑制，默认为False。
        agnostic_nms=False,  # class-agnostic NMS
        # 是否使用增强推理，默认为False。
        augment=False,  # augmented inference
        # 是否可视化特征图，默认为False。
        visualize=False,  # visualize features
        # 是否自动更新模型权重文件，默认为False。是否清除优化器相关
        update=False,  # update all models
        # 结果保存的项目文件夹路径，默认为“runs/detect”。
        project=ROOT / 'runs/detect',  # save results to project/name
        #  结果保存的文件名，默认为“exp”。
        name='exp',  # save results to project/name
        # 如果结果保存的文件夹已存在，是否覆盖，默认为False，即不覆盖。
        exist_ok=False,  # existing project/name ok, do not increment
        # 检测框的线条宽度，默认为3。
        line_thickness=3,  # bounding box thickness (pixels)
        # 是否隐藏标签信息，默认为False，即显示标签信息。
        hide_labels=False,  # hide labels
        # 是否隐藏置信度信息，默认为False，即显示置信度信息。
        hide_conf=False,  # hide confidences
        # 是否使用FP16的半精度推理模式，默认为False。
        half=False,  # use FP16 half-precision inference
        # 是否使用OpenCV DNN作为ONNX推理的后端，默认为False。
        dnn=False,  # use OpenCV DNN for ONNX inference
        # 采样帧的间隔
        vid_stride=1,  # video frame-rate stride
):
    # ---------2.初始化配置---------
    source = str(source)
    # 是否保存图片和txt文件，如果nosave(传入的参数)为false且source的结尾不是txt则保存图片
    save_img = not nosave and not source.endswith(
        '.txt')  # save inference images
    # # 判断source是不是视频/图像文件路径
    # Path()提取文件名。suffix：最后一个组件的文件扩展名。若source是"D://YOLOv5/data/1.jpg"， 则Path(source).suffix是".jpg"， Path(source).suffix[1:]是"jpg"
    # 而IMG_FORMATS 和 VID_FORMATS两个变量保存的是所有的视频和图片的格式后缀。
    # .suffix 返回的是带点的字符串，比如 .mp4、.jpg
# [1:] 是从索引 1 开始切片，去掉开头的点
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    #  # 判断source是否是链接
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    #   # 判断是source是否是摄像头
    # .isnumeric()是否是由数字组成，返回True or False
    webcam = source.isnumeric() or source.endswith(
        '.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        # # 返回文件。如果source是一个指向图片/视频的链接,则下载输入数据
        source = check_file(source)  # download

    # Directories
    # ---------3.保存结果---------
    # # save_dir是保存运行结果的文件夹名，是通过递增的方式来命名的。第一次运行时路径是“runs\detect\exp”，第二次运行时路径是“runs\detect\exp1”
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    # # 根据前面生成的路径创建文件夹
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Load model
    # --------4.载入模型---------
    # 获取设备 CPU/CUD
    device = select_device(device)
    # # DetectMultiBackend定义在models.common模块中，是我们要加载的网络，其中weights参数就是输入时指定的权重文件（比如yolov5s.pt）
    '''
        stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
        names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...]
        pt: 加载的是否是pytorch模型（也就是pt格式的文件）
        jit：当某段代码即将第一次被执行时进行编译，因而叫“即时编译”
        onnx：利用Pytorch我们可以将model.pt转化为model.onnx格式的权重，在这里onnx充当一个后缀名称，
              model.onnx就代表ONNX格式的权重文件，这个权重文件不仅包含了权重值，也包含了神经网络的网络流动信息以及每一层网络的输入输出信息和一些其他的辅助信息。
    '''
    model = DetectMultiBackend(
        weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # ---------5.加载数据---------
    bs = 1  # batch_size
    if webcam:  # 使用摄像头作为输入
        # 检测cv2.imshow()方法是否可以执行，不能执行则抛出异常
        view_img = check_imshow(warn=True)
        #  cudnn.benchmark = True  # set True to speed up constant image size inference  该设置可以加速预测
        # 加载输入数据流
        dataset = LoadStreams(source, img_size=imgsz,
                              stride=stride, auto=pt, vid_stride=vid_stride)
        '''
         source：输入数据源；image_size 图片识别前被放缩的大小；stride：识别时的步长，
         auto的作用可以看utils.augmentations.letterbox方法，它决定了是否需要将图片填充为正方形，如果auto=True则不需要
        '''
        bs = len(dataset)  # batch_size 批大小
    elif screenshot:  # 从屏幕截图中获取图像
        # 当前屏幕截图中截取图像作为输入。
        # 一般用于演示或监控场景，可以实时对屏幕内容进行目标检测。
        dataset = LoadScreenshots(
            source, img_size=imgsz, stride=stride, auto=pt)
    else:  # 直接从source文件下读取图片
        dataset = LoadImages(source, img_size=imgsz,
                             stride=stride, auto=pt, vid_stride=vid_stride)
    # 保存视频的路径
    # 前者是视频路径,后者是一个cv2.VideoWriter对象
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ---------6.推理部分---------
    # Run inference
    # 对模型进行预热（warmup）。预热是指在正式推理前用一个假数据跑一次，以便让模型把各层的初始化、显存分配、GPU kernel 编译等工作提前完成，从而加快之后的第一次真实推理速度。
    # pt: 布尔变量，是否使用的是 PyTorch 格式的模型（.pt 文件）。
# model.triton: 是否使用了 Triton 推理后端。
# bs: batch size，当前每次推理输入的图片数量。
# imgsz: 是一个元组 (h, w)，表示图像输入尺寸。
# 👇 所以 imgsz=(N, 3, H, W) 被构造为：
# N 是 batch size，通常在 pt 或 Triton 情况下设为 1（即 (1, 3, 640, 640)）；
# 3 是图像通道数（RGB）；
# *imgsz 解包高度和宽度，如 imgsz=(640, 640)。
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # seen	已处理的图像数量，用于累计记录有多少张图被检测
# windows	用于显示图像时的窗口列表（OpenCV imshow 用）
# dt	是 3 个 Profile() 实例组成的元组，用于分别统计以下三个阶段的耗时：
# dt[0]: 用来统计 预处理 时间（如 letterbox, to(device), normalize 等）；
# dt[1]: 用来统计 模型推理（inference） 时间；
# dt[2]: 用来统计 后处理 时间（如 NMS, annotator.box_label() 等）。
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # 遍历图片，进行计数
    for path, im, im0s, vid_cap, s in tqdm(dataset, total=len(dataset)):
        '''
         在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
          path：文件路径（即source）
          im: resize后的图片（经过了放缩操作）
          im0s: 原始图片
          vid_cap=none
          s： 图片的基本信息，比如路径，大小
        '''
        # 数据预处理
        with dt[0]:
            #  # 将图片放到指定设备(如GPU)上识别。#torch.size=[3,640,480]
            im = torch.from_numpy(im).to(model.device)
            # uint8 to fp16/32 # 把输入从整型转化为半精度/全精度浮点数。
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            # 0 - 255 to 0.0 - 1.0 归一化，所有像素点除以255
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                # # expand for batch dim 添加一个第0维。缺少batch这个尺寸，所以将它扩充一下，变成[1，3,640,480]
                # 等价于im = im[np.newaxis, ...]  # NumPy 写法
                # 或
                # im = im.unsqueeze(0)      # PyTorch 写法
                im = im[None]  # expand for batch dim

        # Inference
        # 前向推理
        with dt[1]:
            # 如果 visualize=True，就调用 increment_path(...) 来生成保存路径，并赋值给 visualize
            # # 可视化文件路径。如果为True则保留推理过程中的特征图，保存在runs文件夹
            # Python 是动态类型语言
            # x = True       # 布尔
            # x = "path/"    # 字符串
            visualize = increment_path(
                save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 推理结果，pred保存的是所有的bound_box的信息，
            # 模型预测出来的所有检测框，torch.size=[1,18900,85]
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        # 后处理
        with dt[2]:
            # 执行非极大值抑制，返回值为过滤后的预测框
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # 把所有的检测框画到原图中
        for i, det in enumerate(pred):  # per image每次迭代处理一张图片
            '''
            i：每个batch的信息
            det:表示5个检测框的信息
            '''
            seen += 1  # seen是一个计数的功能
            if webcam:  # batch_size >= 1
                # 如果输入源是webcam则batch_size>=1 取出dataset中的一张图片
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                '''
                大部分我们一般都是从LoadImages流读取本地文件中的照片或者视频 所以batch_size=1
                   p: 当前图片/视频的绝对路径 如 F:\yolo_v5\yolov5-U\data\images\bus.jpg
                   s: 输出信息 初始为 ''
                   im0: 原始图片 letterbox + pad 之前的图片
                   frame: 视频流,此次取的是第几张图片
                '''

            p = Path(p)  # to Path
            # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\fire.jpg
            save_path = str(save_dir / p.name)  # im.jpg
            # 设置保存框坐标的txt文件路径，每张图片对应一个框坐标信息
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 设置输出图片信息。图片shape (w, h)
            s += '%gx%g ' % im.shape[2:]  # print string
            # normalization gain whwh
            # 得到原图的宽和高
            # [1, 0, 1, 0] 是索引列表，用于提取宽、高、宽、高的值；
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            # 保存截图。如果save_crop的值为true，则将检测到的bounding_box单独保存成一张图片
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名
            annotator = Annotator(
                im0, line_width=line_thickness, example=str(names))
            # 判断有没有框
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息映射到原图
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）此时坐标格式为xyxy
                # im.shape[2:] 取的是图像的高和宽。
                # det[:, :4]是模型输出的边界框坐标（x1, y1, x2, y2）
                # 将检测框的坐标从 输入图像的尺度 im.shape[2:] 映射回原始图像尺寸 im0.shape。
                # scale_boxes(from_shape, boxes, to_shape) 用于把 boxes 从模型输入分辨率映射回原始图像分辨率。
                # .round() 是四舍五入为整数像素
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # 统计每个类别的目标数量，并格式化成字符串描述，用于打印或记录检测结果。
                # det 是一个检测结果的数组（通常是一个二维的 numpy 数组），其中每一行表示一个检测框，det[:, 5] 表示所有检测结果的第6列（索引5），也就是类别索引（class id）。
# .unique() 返回该列中不重复的类别索引，即一共检测到了哪些类别。
                for c in det[:, 5].unique():
                    # 统计当前类别 c 出现了多少次（即检测到了多少个该类别的目标）
                    n = (det[:, 5] == c).sum()  # detections per class
                    # add to string
                    # names[int(c)] 是类别的名字，比如如果 c=0，就可能是 'person'。
# {'s' * (n > 1)} 是为了英文复数拼写，如果 n > 1 就在后面加上一个 's'，比如：
# 1 person,
# 2 persons,
# 最后加上 ", " 是为了美化输出格式（多个类别之间用逗号分隔）。
# *重复前面的字符串
# ture一次false0次
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                 # 保存预测结果：txt/图片画框/crop-image
                 # [x1, y1, x2, y2, conf, cls]
                # *xyxy：解包前四个值 [x1, y1, x2, y2] 为边界框的坐标
                # conf：第五个值，代表该目标的置信度（即模型认为这个检测结果是正确的可能性）。
                # cls：第六个值，代表检测到的类别索引（整数，需通过标签表映射成具体类名）。
                # reversed(det)：为了让置信度更高的框先被绘制，通常使用 reversed（因为前面已经根据置信度排序过了）
                # det 默认是按置信度降序排列 ✅
# reversed(det) ⇒ 按置信度升序遍历 ✅
# 作用：后画高置信度框 → 避免被低置信度框盖住
                for *xyxy, conf, cls in reversed(det):
                    # 每次迭代处理一# 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
                    if save_txt:  # Write to file保存txt文件
                        # torch.tensor(xyxy).view(1, 4)
                        # 将列表 xyxy（长度为4）转换成一个形状为 [1, 4] 的 tensor，即： [[x1, y1, x2, y2]]
                        # xyxy2xywh(...)这是一个函数，通常来自 YOLO 的工具函数，其作用是将边界框从 (x1, y1, x2, y2) 格式转换为 (x_center, y_center, width, height) 格式
                        # / gn
                        # 对坐标进行归一化。gn 通常是一个形如 [image_width, image_height, image_width, image_height] 的张量，用于将坐标映射到 [0, 1] 区间
                        # .view(-1).tolist()
                        # 将结果张量展平成一维数组，然后转换为 Python 列表
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        # line的形式是： ”类别 x y w h“，若save_conf为true，则line的形式是：”类别 x y w h 置信度“
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        # 'a' 模式表示 追加模式（append），意味着如果文件已存在，就在文件末尾追加内容；如果文件不存在，就创建一个新文件。
# with 语句是 Python 的上下文管理器语法，用于自动打开和关闭文件（即使写入过程中抛出异常，也会自动关闭文件，防止资源泄露）。
# f 是文件对象，可以通过 f.write(...) 来写入数据。
                        with open(f'{txt_path}.txt', 'a') as f:
                            # # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
                            # %g ' * len(line)
                            # 如果 line 有 3 个元素，就变成了：'%g %g %g '
                            # .rstrip()把右侧多余的空格去掉'%g %g %g'
                            # % line使用 line 列表中的元素填充 %g %g %g 如'0.1234 56.789 1'
                            # + '\n'加上换行
                            # '0.1234 56.789 1\n'
                            # f.write(...)写入文件中，作为一行
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 在原图上画框+将预测到的目标剪切出来保存成图片，保存在save_dir/crops下，在原图像画图或者保存结果
                    # 如果需要保存图像（save_img）、保存裁剪图（save_crop）、或展示图像（view_img），就执行下面的操作
                    if save_img or save_crop or view_img:  # Add bbox to image
                        # 把当前预测框的类别 cls（通常是 float32）转换为整数，表示类别 ID，用于查找类别名或颜色。
                        c = int(cls)  # integer class
                        # 决定要不要画出标签，以及标签内容是什么。
# 如果 hide_labels = True，则不显示标签 → label = None
# 如果 hide_labels = False 且 hide_conf = True，只显示类别名，如 person
# 如果 hide_conf = False，显示类别+置信度，如 person 0.92
                        label = None if hide_labels else (
                            names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # annotator 对象（如 Annotator 类）在图像上绘制边框和标签
                        # xyxy：边框坐标
                        # label：标签内容（如 person 0.92）
                        # colors(c, True)：根据类别 c 选取颜色（保证不同类别颜色不同）
                        # bgr=False（默认）：返回 RGB 格式，如 (255, 56, 56)
# bgr=True：返回 BGR 格式，如 (56, 56, 255)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:  # 如果启用了 --save-crop 参数，就把检测框对应的区域裁剪出来保存。
                        # save_one_box 函数的作用是将图像中对应 xyxy 的部分裁剪下来并保存
                        # xyxy：框的坐标
                        # imc：原图（可能是 im0.copy()）
                        # file：保存路径，如 runs/detect/exp0/crops/person/img001.jpg
                        # BGR=True：是否以 OpenCV 的 BGR 色彩格式保存（与 RGB 相反
                        save_one_box(xyxy, imc, file=save_dir / 'crops' /
                                     names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # im0是绘制好的图片
            im0 = annotator.result()
            if view_img:  # 如果设置展示，则show图片 / 视频
                # 在 Linux 系统下，为图像窗口开启调整大小的功能，并根据图像尺寸设置窗口大小
                # windows 是一个保存已创建窗口名的列表
                # 如果当前窗口名 p 不在已创建窗口列表中，才进行窗口创建与调整。
                if platform.system() == 'Linux' and p not in windows:
                    # windows.append(p)
                    windows.append(p)
                    # allow window resize (Linux)
                    # 创建一个可以被用户手动调整大小的窗口（通过 WINDOW_NORMAL）。
# WINDOW_KEEPRATIO 表示在调整窗口大小时保持图像宽高比。
# 用 | 进行按位或运算后得到一个新值，OpenCV 内部就可以知道你想同时使用这两个窗口设置。
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO)
                    # 显式设置窗口的初始尺寸为图像 im0 的尺寸。
# im0.shape[1] 是图像的宽度，im0.shape[0] 是图像的高度。
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # 显示图像窗口，窗口名为 p，图像内容为 im0
                cv2.imshow(str(p), im0)
                # 等待键盘事件
                # 参数 1 表示等待 1 毫秒后继续执行程序（非阻塞），如果在这 1 毫秒内按了键，则返回按键的 ASCII 码，否则返回 -1
                # waitKey 是刷新图像窗口和捕获键盘事件的必要调用，缺少会导致图像窗口不更新
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 设置保存图片/视频
            if save_img:  # 保存绘制完的图片
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream' 如果是视频或者"流"
                    '''
                    # vid_path：保存当前各个视频输出路径的列表。
                    # vid_writer：视频写入器对象列表，每个对应一个输出视频。
                    # vid_cap：当前输入视频的 cv2.VideoCapture 对象，可能为 None（例如网络流）。
                    # im0：当前处理的图像帧，格式为 numpy ndarray。
                    # save_path：当前帧对应的视频文件保存路径。
                    # i：索引，表示当前处理的是第几个视频/流。
                    # 如果当前帧的保存路径和之前记录的不一样，说明是新的视频文件，需要重新初始化视频写入器。
                    # 更新 vid_path[i] 记录当前路径这样后续帧就会知道当前写入的目标文件
                    '''
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        # 判断 vid_writer[i] 是否是一个有效的 cv2.VideoWriter 对象。
# vid_writer 是一个列表，存储每个视频的写入器实例。
# 如果 vid_writer[i] 是已存在的视频写入器，则先调用 .release() 方法关闭它。
# 这一步非常重要，避免文件句柄泄露和确保之前的视频文件正常保存关闭。
# 一开始 vid_writer[i] 可能是 None，或者是某个占位符（比如空列表、空字符串等），还没创建写入器。
# 只有在它已经是一个 cv2.VideoWriter 实例时，才能调用 .release()
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            # release previous video writer
                            vid_writer[i].release()
                        # 判断 vid_cap（视频读取对象）是否存在，即当前是否有视频文件输入（而非网络流或图片流）
                        if vid_cap:  # video
                            # 如果有视频输入，从 vid_cap 读取视频参数：
                            # fps：帧率
                            # w：视频宽度（像素）
                            # h：视频高度（像素）
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            # 如果没有 vid_cap（比如是直播流、摄像头流、或者图片序列）
                            # 则手动设置帧率为 30 fps（默认值），
                            # 视频宽高直接取当前帧图像 im0 的尺寸
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # force *.mp4 suffix on results videos
                        # 使用 Path 类强制把视频保存路径的后缀改为 .mp4
                        # 不论传入的路径后缀是什么，最终都保存为 MP4 格式，确保视频格式统一
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        # 创建新的 cv2.VideoWriter 对象
                        # save_path：输出文件路径
                        # cv2.VideoWriter_fourcc(*'mp4v')：指定编码格式为 mp4v（MP4 格式编码）
                        # fps：帧率
# (w, h)：视频尺寸，宽高元组
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    # 把当前帧图像 im0 写入到对应的视频文件
                    # 这一步完成单帧保存
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # 检测结束后在控制台或日志中输出检测结果、检测耗时的功能。
        # {'' if len(det) else '(no detections), '}
        # 如果 len(det) 大于 0，说明检测到了对象 → 加 ''（不加任何内容）。
# 否则 → 输出 "(no detections), " 字样，说明没检测到任何目标。
# dt[1] 是一个 Profile() 对象（一个小工具类，记录时间消耗）。
# .dt 是记录的时间（单位为秒）。
# 乘以 1E3（也就是 1000）表示 换算成毫秒。
# :.1f 表示保留 一位小数。
# 所以这部分最终格式可能是 3.4ms、18.7ms，表示检测时间
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # ---------7.在终端里打印出运行的结果---------
    # dt = (Profile(), Profile(), Profile())：之前代码中用 Profile() 记录三个阶段的耗时：
# 图像预处理（如缩放、标准化）
# 推理（模型前向）
# NMS（非极大值抑制）
    # x.t 是某个阶段总耗时
    # seen 是处理的图像总数。
    # * 1E3 把单位转换成 毫秒/图像。
    # 最终 t 是一个三元组，比如 (2.3, 12.5, 1.7)。
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # 如(1, 3, *imgsz) 是模型输入张量的形状（batch=1，3通道，尺寸为 imgsz）
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:  # 如果设置了保存预测框为 .txt 或保存带框图像为 .jpg
        # 如果保存了 .txt 标签文件（save_txt=True），就：
        # 统计保存的标签文件数量如：10 labels saved to runs/detect/exp/labels
        # 否则 s 为空。
        # 输出保存结果路径和标签保存信息（如果有）。
        # colorstr('bold', save_dir) 用于让路径加粗高亮显示（控制台可见
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:  # update=True 是推理脚本里用于「清理权重文件中的训练信息」，不是更新参数，而是为了部署更轻量。
        # update model (to fix SourceChangeWarning)
        # strip_optimizer(weights_path) 的作用是：
        # 从保存的 .pt 模型中移除优化器信息
        strip_optimizer(weights[0])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / 'best.engine', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT /
                        'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT /
                        'data/mydrone.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default=ROOT /
                        'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3,
                        type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False,
                        action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False,
                        action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    opt = parser.parse_args()
    # len(opt.imgsz) == 1：表示用户只提供了一个尺寸（正方形图像）
# opt.imgsz *= 2：将列表扩展一倍，变成 [640, 640]
# 变成两个相同的值 [640] * 2 → [640, 640]，表示高和宽都是 640
# else 1：如果本来就是两个元素，就保持不变
# 确保 imgsz 是 (height, width) 这样的两个元素形式。
# 如果原本就是 [640, 480] 这样两个值，就不变（乘以 1）
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # 打印参数信息
    print_args(vars(opt))
    return opt

# 接受一个参数 opt，它通常是一个 argparse.Namespace 对象，用于包含命令行参数。
# 假如命令行输入python detect.py --weights yolov5s.pt --img 640 --conf 0.25
# 则opt 就是：Namespace(weights='yolov5s.pt', img=640, conf=0.25)


def main(opt):
    # 检查项目所需的依赖库是否已安装，并提醒用户
    # 忽略检查这两个库，即使没安装也不报错。
    # 这通常用于减少依赖，例如你只运行推理（inference）代码时，不一定需要用到 tensorboard 或 thop（参数量、FLOPs 分析工具）。
    check_requirements(exclude=('tensorboard', 'thop'))
    # vars(opt) 会把 Namespace 对象转换成字典。例如：
    # **vars(opt) 是参数解包，等价于 run(weights='yolov5s.pt', img=640, conf=0.25)
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
