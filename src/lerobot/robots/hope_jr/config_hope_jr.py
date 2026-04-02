"""
Hope Jr 机器人配置模块 (Hope Jr Configuration Module)
==============================================

本模块定义了 Hope Jr 机器人的配置类。

配置类:
    - HopeJrHandConfig: Hope Jr 机械手配置
    - HopeJrArmConfig: Hope Jr 机械臂配置
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("hope_jr_hand")
@dataclass
class HopeJrHandConfig(RobotConfig):
    """
    Hope Jr 机械手配置类。
    
    属性:
        port: 连接机械手的端口
        side: 左/右手 ("left"/"right")
        disable_torque_on_disconnect: 断开连接时是否禁用力矩
        cameras: 相机配置字典
    """
    port: str  # 连接机械手的端口
    side: str  # 左/右手 ("left"/"right")

    disable_torque_on_disconnect: bool = True

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)


@RobotConfig.register_subclass("hope_jr_arm")
@dataclass
class HopeJrArmConfig(RobotConfig):
    port: str  # Port to connect to the hand
    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
