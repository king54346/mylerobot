"""实时分块 (RTC) 的动作队列管理。

RTC 动作队列模块
======================

本模块提供 ActionQueue，一个用于实时控制场景中管理动作块的线程安全队列。
它支持 RTC 启用和非 RTC 模式，处理动作合并和剩余追踪。

主要功能:
- 线程安全的动作队列管理
- 原始动作追踪（用于 RTC 计算）
- 处理后动作存储（用于机器人执行）
- 推理延迟补偿
"""

import logging
from threading import Lock

import torch
from torch import Tensor

from lerobot.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """用于实时控制中管理动作块的线程安全队列。

    该队列处理两种类型的动作序列:
    - 原始动作: 用于 RTC 计算前一块的剩余动作
    - 处理后动作: 后处理完毕、准备供机器人执行的动作

    队列有两种操作模式:
    1. RTC 启用: 用新动作替换整个队列，考虑推理延迟
    2. RTC 禁用: 将新动作追加到队列，保持连续性

    参数:
        cfg (RTCConfig): 实时分块行为的配置。

    属性:
        queue (Tensor | None): 用于机器人展开的处理后动作 (time_steps, action_dim)。
        original_queue (Tensor | None): 用于 RTC 计算的原始动作 (time_steps, action_dim)。
        last_index (int): 队列中当前消费索引。
    """

    def __init__(self, cfg: RTCConfig):
        """初始化动作队列。

        参数:
            cfg: 控制队列行为的 RTC 配置。
        """
        self.queue = None  # 用于机器人展开的处理后动作
        self.original_queue = None  # 用于 RTC 的原始动作
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        """从队列获取下一个动作。

        返回:
            Tensor | None: 下一个动作 (action_dim,)，如果队列为空则返回 None。
                          返回克隆以防止外部修改。
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def qsize(self) -> int:
        """获取队列中剩余动作的数量。

        返回:
            int: 未消费的动作数量。
        """
        if self.queue is None:
            return 0
        length = len(self.queue)
        return length - self.last_index

    def empty(self) -> bool:
        """检查队列是否为空。

        返回:
            bool: 如果没有剩余动作返回 True，否则返回 False。
        """
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index <= 0

    def get_action_index(self) -> int:
        """获取当前动作消费索引。

        返回:
            int: 下一个要消费的动作的索引。
        """
        return self.last_index

    def get_left_over(self) -> Tensor | None:
        """获取 RTC prev_chunk_left_over 的剩余原始动作。

        这些是当前块中未消费的动作，将被 RTC 用于
        计算下一块的校正。

        返回:
            Tensor | None: 剩余的原始动作 (remaining_steps, action_dim)，
                          如果没有原始队列则返回 None。
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :]

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = 0,
    ):
        """将新动作合并到队列中。

        此方法根据 RTC 模式有不同的操作:
        - RTC 启用: 替换队列，考虑推理延迟
        - RTC 禁用: 追加到队列，保持连续性

        参数:
            original_actions: 来自策略的未处理动作 (time_steps, action_dim)。
            processed_actions: 用于机器人的后处理动作 (time_steps, action_dim)。
            real_delay: 推理延迟的时间步数。
            action_index_before_inference: 推理开始前的索引，用于验证。
        """
        with self.lock:
            self._check_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(original_actions, processed_actions, real_delay)
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(self, original_actions: Tensor, processed_actions: Tensor, real_delay: int):
        """用新动作替换队列 (RTC 模式)。

        丢弃前 `real_delay` 个动作，因为它们对应于推理期间的时间，
        此时机器人正在执行前一个动作。

        参数:
            original_actions: 来自策略的未处理动作。
            processed_actions: 用于机器人的后处理动作。
            real_delay: 由于推理延迟需要跳过的时间步数。
        """
        self.original_queue = original_actions[real_delay:].clone()
        self.queue = processed_actions[real_delay:].clone()

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}")

        self.last_index = 0

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        """将新动作追加到队列 (非 RTC 模式)。

        移除已消费的动作并追加新动作，保持队列连续性而不替换。

        参数:
            original_actions: 来自策略的未处理动作。
            processed_actions: 用于机器人的后处理动作。
        """
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_delays(self, real_delay: int, action_index_before_inference: int | None = None):
        """验证计算的延迟是否符合预期。

        比较从推理延迟计算的延迟与推理期间实际消费的动作数量。

        参数:
            real_delay: 从推理延迟计算的延迟。
            action_index_before_inference: 推理开始时的动作索引。
        """
        if action_index_before_inference is None:
            return

        indexes_diff = self.last_index - action_index_before_inference
        if indexes_diff != real_delay:
            # 检查动作索引差（基于动作队列计算的真实延迟）
            # 是否与基于推理延迟计算的延迟相同
            logger.warning(
                f"[ACTION_QUEUE] Indexes diff is not equal to real delay. "
                f"Indexes diff: {indexes_diff}, real delay: {real_delay}"
            )
