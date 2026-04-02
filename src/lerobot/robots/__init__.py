"""
机器人模块 (Robots Module)
========================

本模块提供了机器人控制的核心组件。

导出:
    - RobotConfig: 机器人配置基类
    - Robot: 机器人抽象基类
    - make_robot_from_config: 根据配置创建机器人实例的工厂函数
"""

from .config import RobotConfig
from .robot import Robot
from .utils import make_robot_from_config
