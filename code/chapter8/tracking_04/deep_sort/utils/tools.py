from functools import wraps
from time import time


def is_video(ext: str):
    """
    Returns true if ext exists in
    allowed_exts for video files.

    Args:
        ext:

    Returns:

    """

    allowed_exts = ('.mp4', '.webm', '.ogg', '.avi', '.wmv', '.mkv', '.3gp')
    return any((ext.endswith(x) for x in allowed_exts))

# 统计函数执行时间和每秒处理帧数（FPS），常用于性能监控或调试
def tik_tok(func):
    """
    keep track of time for each process.
    Args:
        func:

    Returns:

    """
    # 保留原函数的名称、文档字符串等信息
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time()
        try:
            # 调用原函数
            return func(*args, **kwargs)
        finally:
            end_ = time()
            print("time: {:.03f}s, fps: {:.03f}".format(end_ - start, 1 / (end_ - start)))

    return _time_it
