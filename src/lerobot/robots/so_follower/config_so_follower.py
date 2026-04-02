"""
SO 跟随者机械臂配置模块 (SO Follower Configuration Module)
====================================================

本模块定义了 SO 系列跟随者机械臂的配置类。

支持的型号:
    - SO-100
    - SO-101
"""

from dataclasses import dataclass, field
from typing import TypeAlias

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SOFollowerConfig:
    """
    SO 跟随者机器人的基础配置类。
    
    属性:
        port: 连接机械臂的串口
        disable_torque_on_disconnect: 断开连接时是否禁用力矩
        max_relative_target: 最大相对目标位置，用于安全限制
        cameras: 相机配置字典
        use_degrees: 是否使用角度制
    """
    # 连接机械臂的端口
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True


@RobotConfig.register_subclass("so101_follower")
@RobotConfig.register_subclass("so100_follower")
@dataclass
class SOFollowerRobotConfig(RobotConfig, SOFollowerConfig):
    pass


SO100FollowerConfig: TypeAlias = SOFollowerRobotConfig
SO101FollowerConfig: TypeAlias = SOFollowerRobotConfig
