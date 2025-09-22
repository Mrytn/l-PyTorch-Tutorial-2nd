from .deep_sort import DeepSort

# __all__ 声明模块公开的接口，外部 from xxx import * 时，只能导入 DeepSort 和 build_tracker
__all__ = ['DeepSort', 'build_tracker']

# build_tracker 接收 cfg（通常是 .yaml 或 .json 的配置）
# 解析里面的参数 → 调用 DeepSort(...) → 返回一个可用的 跟踪器实例
# 就能得到 DeepSORT 跟踪器，然后直接 tracker.update(detections) 使用。
def build_tracker(cfg, use_cuda):
    return DeepSort(cfg.DEEPSORT.REID_CKPT,
                max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)










