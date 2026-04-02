"""
核心类型定义模块 (Core Types Module)
==============================

定义了数据处理系统中使用的核心类型：
- PolicyAction: 策略输出的动作（torch.Tensor）
- RobotAction: 机器人执行的动作（字典格式）
- EnvAction: 环境接收的动作（numpy数组）
- RobotObservation: 机器人的观测数据（字典格式）
- EnvTransition: 环境转移元组，包含观测、动作、奖励等
"""

from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias, TypedDict

import numpy as np
import torch


class TransitionKey(str, Enum):
    """
    环境转移元组的键枚举类。
    
    用于访问 EnvTransition 字典中的各个组件，
    确保键名的一致性和类型安全。
    """

    # TODO(Steven): 使用常量
    OBSERVATION = "observation"          # 观测数据
    ACTION = "action"                    # 动作数据
    REWARD = "reward"                    # 奖励值
    DONE = "done"                        # 终止标志（回合是否结束）
    TRUNCATED = "truncated"              # 截断标志（回合是否被截断）
    INFO = "info"                        # 附加信息
    COMPLEMENTARY_DATA = "complementary_data"  # 补充数据（如遥操作数据）


# 类型别名定义
# PolicyAction: 策略网络输出的动作张量，形状通常为 (batch_size, action_dim) 或 (action_dim,)
PolicyAction: TypeAlias = torch.Tensor

# RobotAction: 发送给机器人的动作字典，键为电机名/关节名，值为目标位置
RobotAction: TypeAlias = dict[str, Any]

# EnvAction: Gym环境使用的numpy数组动作
EnvAction: TypeAlias = np.ndarray

# RobotObservation: 机器人观测数据字典，包含位置、图像等信息
RobotObservation: TypeAlias = dict[str, Any]


# EnvTransition: 环境转移元组，用于强化学习和数据收集
# 包含一次环境交互的所有信息：观测、动作、奖励、终止标志等
EnvTransition = TypedDict(
    "EnvTransition",
    {
        TransitionKey.OBSERVATION.value: RobotObservation | None,  # 当前观测
        TransitionKey.ACTION.value: PolicyAction | RobotAction | EnvAction | None,  # 执行的动作
        TransitionKey.REWARD.value: float | torch.Tensor | None,  # 获得的奖励
        TransitionKey.DONE.value: bool | torch.Tensor | None,  # 是否终止
        TransitionKey.TRUNCATED.value: bool | torch.Tensor | None,  # 是否截断
        TransitionKey.INFO.value: dict[str, Any] | None,  # 附加信息
        TransitionKey.COMPLEMENTARY_DATA.value: dict[str, Any] | None,  # 补充数据
    },
)
