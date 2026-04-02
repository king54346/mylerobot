"""
队列工具模块 (Queue Utilities Module)
==================================

本模块提供了用于多进程队列操作的工具函数。

主要函数:
    - get_last_item_from_queue: 获取队列中的最后一项，丢弃旧数据
"""

import platform
from contextlib import suppress
from queue import Empty
from typing import Any

from torch.multiprocessing import Queue


def get_last_item_from_queue(queue: Queue, block=True, timeout: float = 0.1) -> Any:
    if block:
        try:
            item = queue.get(timeout=timeout)
        except Empty:
            return None
    else:
        item = None

    # Drain queue and keep only the most recent parameters
    if platform.system() == "Darwin":
        # On Mac, avoid using `qsize` due to unreliable implementation.
        # There is a comment on `qsize` code in the Python source:
        # Raises NotImplementedError on Mac OSX because of broken sem_getvalue()
        try:
            while True:
                item = queue.get_nowait()
        except Empty:
            pass

        return item

    # Details about using qsize in https://github.com/huggingface/lerobot/issues/1523
    while queue.qsize() > 0:
        with suppress(Empty):
            item = queue.get_nowait()

    return item
