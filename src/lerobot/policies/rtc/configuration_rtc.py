"""实时分块 (RTC) 和双向解码 (BID) 配置类。

RTC 配置模块
======================

基于:
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
"""

from dataclasses import dataclass

from lerobot.configs.types import RTCAttentionSchedule


@dataclass
class RTCConfig:
    """实时分块 (RTC) 推理配置。

    RTC 通过将块生成视为修复问题来改善实时推理，
    使用前缀注意力策略性地处理动作块之间的重叠时间步。
    """

    # 基础设施
    enabled: bool = False

    # 核心 RTC 设置
    # TODO: 改为 exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    # 调试设置
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """验证 RTC 配置参数。"""
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")
