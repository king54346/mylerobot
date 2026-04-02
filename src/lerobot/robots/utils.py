"""
机器人工具模块 (Robot Utilities Module)
======================================

本模块提供了机器人相关的工具函数。

主要函数:
    - make_robot_from_config: 根据配置创建机器人实例
    - ensure_safe_goal_position: 确保目标位置在安全范围内
"""

import logging
from pprint import pformat
from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .config import RobotConfig
from .robot import Robot


def make_robot_from_config(config: RobotConfig) -> Robot:
    """
    根据配置创建机器人实例的工厂函数。
    
    参数:
        config: 机器人配置对象
        
    返回:
        对应类型的机器人实例
        
    支持的机器人类型:
        - koch_follower: Koch 跟随者机械臂
        - omx_follower: OMX 跟随者机械臂
        - so100_follower: SO-100 跟随者
        - so101_follower: SO-101 跟随者
        - lekiwi: LeKiwi 移动机器人
        - hope_jr_hand/arm: Hope Jr 机械手/臂
        - bi_so_follower: 双臂 SO 跟随者
        - reachy2: Reachy2 机器人
        - openarm_follower: OpenArm 跟随者
        - bi_openarm_follower: 双臂 OpenArm 跟随者
    """
    # TODO(Steven): 考虑对所有类型使用 make_device_from_device_class
    if config.type == "koch_follower":
        from .koch_follower import KochFollower

        return KochFollower(config)
    elif config.type == "omx_follower":
        from .omx_follower import OmxFollower

        return OmxFollower(config)
    elif config.type == "so100_follower":
        from .so_follower import SO100Follower

        return SO100Follower(config)
    elif config.type == "so101_follower":
        from .so_follower import SO101Follower

        return SO101Follower(config)
    elif config.type == "lekiwi":
        from .lekiwi import LeKiwi

        return LeKiwi(config)
    elif config.type == "hope_jr_hand":
        from .hope_jr import HopeJrHand

        return HopeJrHand(config)
    elif config.type == "hope_jr_arm":
        from .hope_jr import HopeJrArm

        return HopeJrArm(config)
    elif config.type == "bi_so_follower":
        from .bi_so_follower import BiSOFollower

        return BiSOFollower(config)
    elif config.type == "reachy2":
        from .reachy2 import Reachy2Robot

        return Reachy2Robot(config)
    elif config.type == "openarm_follower":
        from .openarm_follower import OpenArmFollower

        return OpenArmFollower(config)
    elif config.type == "bi_openarm_follower":
        from .bi_openarm_follower import BiOpenArmFollower

        return BiOpenArmFollower(config)
    elif config.type == "mock_robot":
        from tests.mocks.mock_robot import MockRobot

        return MockRobot(config)
    else:
        try:
            return cast(Robot, make_device_from_device_class(config))
        except Exception as e:
            raise ValueError(f"Error creating robot with config {config}: {e}") from e


# TODO(pepijn): 移动到 pipeline step 中，确保不需要在机器人代码中执行此操作，
# 并且发送到机器人的动作可以干净地用于数据集
def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: float | dict[str, float]
) -> dict[str, float]:
    """
    为安全起见，限制相对动作目标的幅度。
    
    参数:
        goal_present_pos: 字典，键为关节名称，值为 (目标位置, 当前位置) 元组
        max_relative_target: 最大相对目标值，可以是单个浮点数或字典
        
    返回:
        安全的目标位置字典
    """

    if isinstance(max_relative_target, float):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    if warnings_dict:
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"{pformat(warnings_dict, indent=4)}"
        )

    return safe_goal_positions
