"""
处理器工厂模块 (Processor Factory Module)
======================================

提供创建默认处理器的工厂函数。
这些处理器用于机器人控制和遥操作场景：

- make_default_teleop_action_processor: 创建遥操作动作处理器
  用于处理从遥操作设备获取的动作

- make_default_robot_action_processor: 创建机器人动作处理器
  用于处理发送给机器人的动作

- make_default_robot_observation_processor: 创建机器人观测处理器
  用于处理从机器人传感器获取的观测数据

默认情况下，这些处理器使用 IdentityProcessorStep，
即不对数据做任何处理，直接透传。
用户可以通过添加自定义处理步骤来扩展这些处理器。
"""

# 导入数据转换工具函数
from .converters import (
    observation_to_transition,              # 观测数据转换为转移元组
    robot_action_observation_to_transition, # 机器人动作+观测转换为转移元组
    transition_to_observation,              # 转移元组转换为观测数据
    transition_to_robot_action,             # 转移元组转换为机器人动作
)
from .core import RobotAction, RobotObservation
from .pipeline import IdentityProcessorStep, RobotProcessorPipeline


def make_default_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    """
    创建默认的遥操作动作处理器。
    
    该处理器用于处理从遥操作设备（如主臂、VR控制器等）获取的动作数据。
    默认使用恒等处理（不做任何转换）。
    
    输入: (RobotAction, RobotObservation) - 遥操作动作和当前观测
    输出: RobotAction - 处理后的机器人动作
    
    Returns:
        配置好的机器人处理管道
    """
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],  # 默认使用恒等处理，不做任何转换
        to_transition=robot_action_observation_to_transition,  # 输入转换函数
        to_output=transition_to_robot_action,  # 输出转换函数
    )
    return teleop_action_processor


def make_default_robot_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    """
    创建默认的机器人动作处理器。
    
    该处理器用于处理发送给机器人执行的动作数据。
    可以在这里添加动作限制、平滑、安全检查等处理步骤。
    
    输入: (RobotAction, RobotObservation) - 机器人动作和当前观测
    输出: RobotAction - 处理后的机器人动作
    
    Returns:
        配置好的机器人处理管道
    """
    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[IdentityProcessorStep()],  # 默认使用恒等处理
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    return robot_action_processor


def make_default_robot_observation_processor() -> RobotProcessorPipeline[RobotObservation, RobotObservation]:
    """
    创建默认的机器人观测处理器。
    
    该处理器用于处理从机器人传感器（编码器、相机等）获取的观测数据。
    可以在这里添加图像预处理、数据转换等处理步骤。
    
    输入: RobotObservation - 原始机器人观测数据
    输出: RobotObservation - 处理后的观测数据
    
    Returns:
        配置好的机器人处理管道
    """
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],  # 默认使用恒等处理
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    return robot_observation_processor


def make_default_processors():
    """
    创建所有默认处理器的便捷函数。
    
    返回一个元组，包含机器人控制所需的三个核心处理器：
    1. 遥操作动作处理器 - 处理遥操作设备的输入
    2. 机器人动作处理器 - 处理发送给机器人的动作
    3. 机器人观测处理器 - 处理从传感器获取的数据
    
    Returns:
        tuple: (遥操作动作处理器, 机器人动作处理器, 机器人观测处理器)
    """
    teleop_action_processor = make_default_teleop_action_processor()
    robot_action_processor = make_default_robot_action_processor()
    robot_observation_processor = make_default_robot_observation_processor()
    return (teleop_action_processor, robot_action_processor, robot_observation_processor)
