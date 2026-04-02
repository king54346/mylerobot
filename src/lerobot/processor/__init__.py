"""
数据处理器模块 (Processor Module)
================================

该模块提供了机器人学习系统中数据处理的核心组件。
处理器用于在不同数据格式之间进行转换，包括：
- 策略动作 (PolicyAction): 神经网络输出的张量格式动作
- 机器人动作 (RobotAction): 发送给机器人执行的字典格式动作
- 环境动作 (EnvAction): Gym环境使用的numpy数组格式动作
- 机器人观测 (RobotObservation): 从传感器获取的观测数据

主要组件：
- ProcessorStep: 处理步骤的基类
- DataProcessorPipeline: 数据处理管道，串联多个处理步骤
- PolicyProcessorPipeline: 策略专用处理管道
- RobotProcessorPipeline: 机器人专用处理管道

核心处理器：
- NormalizerProcessorStep: 数据归一化处理器
- DeviceProcessorStep: 设备转移处理器（CPU/GPU）
- TokenizerProcessorStep: 分词器处理器（用于VLA模型）
"""

# 批处理维度处理器 - 用于添加/移除批次维度
from .batch_processor import AddBatchDimensionProcessorStep

# 数据转换工具 - 在不同数据结构之间转换
from .converters import (
    batch_to_transition,      # 批次数据转换为转移元组
    create_transition,        # 创建新的转移元组
    transition_to_batch,      # 转移元组转换为批次数据
)

# 核心类型定义
from .core import (
    EnvAction,               # 环境动作类型（numpy数组）
    EnvTransition,           # 环境转移元组类型
    PolicyAction,            # 策略动作类型（torch张量）
    RobotAction,             # 机器人动作类型（字典）
    RobotObservation,        # 机器人观测类型（字典）
    TransitionKey,           # 转移元组的键枚举
)

# 增量动作处理器 - 处理相对位置控制
from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep

# 设备处理器 - 处理CPU/GPU数据转移
from .device_processor import DeviceProcessorStep

# 工厂函数 - 创建默认处理器
from .factory import (
    make_default_processors,                    # 创建所有默认处理器
    make_default_robot_action_processor,        # 创建默认机器人动作处理器
    make_default_robot_observation_processor,   # 创建默认机器人观测处理器
    make_default_teleop_action_processor,       # 创建默认遥操作动作处理器
)

# Gym动作处理器 - Numpy/Torch格式转换
from .gym_action_processor import (
    Numpy2TorchActionProcessorStep,   # Numpy转Torch
    Torch2NumpyActionProcessorStep,   # Torch转Numpy
)

# 人机协同学习(HIL)处理器 - 用于在线学习和干预
from .hil_processor import (
    AddTeleopActionAsComplimentaryDataStep,   # 添加遥操作动作作为补充数据
    AddTeleopEventsAsInfoStep,                # 添加遥操作事件信息
    GripperPenaltyProcessorStep,              # 夹爪惩罚处理器
    GymHILAdapterProcessorStep,               # Gym HIL适配器
    ImageCropResizeProcessorStep,             # 图像裁剪缩放处理器
    InterventionActionProcessorStep,          # 干预动作处理器
    RewardClassifierProcessorStep,            # 奖励分类器处理器
    TimeLimitProcessorStep,                   # 时间限制处理器
)

# 归一化处理器 - 数据标准化
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep, hotswap_stats

# 观测处理器 - 处理观测数据
from .observation_processor import VanillaObservationProcessorStep

# 管道核心组件 - 处理流水线
from .pipeline import (
    ActionProcessorStep,              # 动作处理步骤基类
    ComplementaryDataProcessorStep,   # 补充数据处理步骤
    DataProcessorPipeline,            # 通用数据处理管道
    DoneProcessorStep,                # 终止标志处理步骤
    IdentityProcessorStep,            # 恒等处理步骤（不做任何处理）
    InfoProcessorStep,                # 信息处理步骤
    ObservationProcessorStep,         # 观测处理步骤基类
    PolicyActionProcessorStep,        # 策略动作处理步骤
    PolicyProcessorPipeline,          # 策略处理管道
    ProcessorKwargs,                  # 处理器参数类型
    ProcessorStep,                    # 处理步骤基类
    ProcessorStepRegistry,            # 处理步骤注册表
    RewardProcessorStep,              # 奖励处理步骤
    RobotActionProcessorStep,         # 机器人动作处理步骤
    RobotProcessorPipeline,           # 机器人处理管道
    TruncatedProcessorStep,           # 截断标志处理步骤
)

# 策略-机器人桥接处理器 - 在策略动作和机器人动作之间转换
from .policy_robot_bridge import (
    PolicyActionToRobotActionProcessorStep,   # 策略动作转机器人动作
    RobotActionToPolicyActionProcessorStep,   # 机器人动作转策略动作
)

# 重命名处理器 - 重命名观测字段
from .rename_processor import RenameObservationsProcessorStep

# 分词器处理器 - 用于VLA模型的动作分词
from .tokenizer_processor import ActionTokenizerProcessorStep, TokenizerProcessorStep

__all__ = [
    "ActionProcessorStep",
    "AddTeleopActionAsComplimentaryDataStep",
    "AddTeleopEventsAsInfoStep",
    "ComplementaryDataProcessorStep",
    "batch_to_transition",
    "create_transition",
    "DeviceProcessorStep",
    "DoneProcessorStep",
    "EnvAction",
    "EnvTransition",
    "GymHILAdapterProcessorStep",
    "GripperPenaltyProcessorStep",
    "hotswap_stats",
    "IdentityProcessorStep",
    "ImageCropResizeProcessorStep",
    "InfoProcessorStep",
    "InterventionActionProcessorStep",
    "make_default_processors",
    "make_default_teleop_action_processor",
    "make_default_robot_action_processor",
    "make_default_robot_observation_processor",
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "NormalizerProcessorStep",
    "Numpy2TorchActionProcessorStep",
    "ObservationProcessorStep",
    "PolicyAction",
    "PolicyActionProcessorStep",
    "PolicyProcessorPipeline",
    "ProcessorKwargs",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RobotAction",
    "RobotActionProcessorStep",
    "RobotObservation",
    "RenameObservationsProcessorStep",
    "RewardClassifierProcessorStep",
    "RewardProcessorStep",
    "DataProcessorPipeline",
    "TimeLimitProcessorStep",
    "AddBatchDimensionProcessorStep",
    "RobotProcessorPipeline",
    "TokenizerProcessorStep",
    "ActionTokenizerProcessorStep",
    "Torch2NumpyActionProcessorStep",
    "RobotActionToPolicyActionProcessorStep",
    "PolicyActionToRobotActionProcessorStep",
    "transition_to_batch",
    "TransitionKey",
    "TruncatedProcessorStep",
    "UnnormalizerProcessorStep",
    "VanillaObservationProcessorStep",
]
