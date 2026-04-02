"""
遥操作器工具模块 (Teleoperator Utilities)
========================================

提供遥操作器创建和事件管理的工具函数。

主要组件：
- TeleopEvents: 遥操作事件枚举（成功、失败、重录等）
- make_teleoperator_from_config: 根据配置创建遥操作器实例的工厂函数
"""

from enum import Enum
from typing import TYPE_CHECKING, cast

from lerobot.utils.import_utils import make_device_from_device_class

from .config import TeleoperatorConfig

if TYPE_CHECKING:
    from .teleoperator import Teleoperator


class TeleopEvents(Enum):
    """
    遥操作事件枚举类。
    
    定义所有遥操作器共享的事件常量，
    用于在数据采集和辅助学习过程中标记特定事件。
    
    事件类型：
        SUCCESS: 任务成功完成
        FAILURE: 任务失败
        RERECORD_EPISODE: 重新录制当前回合
        IS_INTERVENTION: 标记人类干预（用于HIL学习）
        TERMINATE_EPISODE: 终止当前回合
    """

    SUCCESS = "success"                    # 成功
    FAILURE = "failure"                    # 失败
    RERECORD_EPISODE = "rerecord_episode"  # 重录回合
    IS_INTERVENTION = "is_intervention"    # 人类干预标记
    TERMINATE_EPISODE = "terminate_episode"  # 终止回合


def make_teleoperator_from_config(config: TeleoperatorConfig) -> "Teleoperator":
    """
    根据配置创建遥操作器实例的工厂函数。
    
    根据配置中的 type 字段动态导入并实例化相应的遥操作器类。
    
    Args:
        config: 遥操作器配置对象
        
    Returns:
        Teleoperator: 实例化的遥操作器对象
        
    Raises:
        ValueError: 如果配置类型不受支持
        
    支持的遥操作器类型：
        - keyboard: 键盘遥操作
        - keyboard_ee: 键盘末端执行器控制
        - gamepad: 游戏手柄
        - koch_leader: Koch机械臂主手
        - omx_leader: OMX机械臂主手
        - so100_leader/so101_leader: SO系列机械臂主手
        - bi_so_leader: 双臂SO主手
        - openarm_leader/bi_openarm_leader: OpenArm系列主手
        - openarm_mini: OpenArm Mini
        - homunculus_glove/homunculus_arm: Homunculus设备
        - unitree_g1: 宇树G1人形机器人
        - reachy2_teleoperator: Reachy2机器人
        - mock_teleop: 测试用模拟遥操作器
    """
    # TODO(Steven): 考虑对所有类型都使用 make_device_from_device_class
    
    # 键盘遥操作
    if config.type == "keyboard":
        from .keyboard import KeyboardTeleop

        return KeyboardTeleop(config)
    
    # Koch机械臂主手
    elif config.type == "koch_leader":
        from .koch_leader import KochLeader

        return KochLeader(config)
    
    # OMX机械臂主手
    elif config.type == "omx_leader":
        from .omx_leader import OmxLeader

        return OmxLeader(config)
    
    # SO100机械臂主手
    elif config.type == "so100_leader":
        from .so_leader import SO100Leader

        return SO100Leader(config)
    
    # SO101机械臂主手
    elif config.type == "so101_leader":
        from .so_leader import SO101Leader

        return SO101Leader(config)
    
    # 测试用模拟遥操作器
    elif config.type == "mock_teleop":
        from tests.mocks.mock_teleop import MockTeleop

        return MockTeleop(config)
    
    # 游戏手柄遥操作
    elif config.type == "gamepad":
        from .gamepad.teleop_gamepad import GamepadTeleop

        return GamepadTeleop(config)
    
    # 键盘末端执行器控制
    elif config.type == "keyboard_ee":
        from .keyboard.teleop_keyboard import KeyboardEndEffectorTeleop

        return KeyboardEndEffectorTeleop(config)
    
    # Homunculus数据手套
    elif config.type == "homunculus_glove":
        from .homunculus import HomunculusGlove

        return HomunculusGlove(config)
    
    # Homunculus手臂
    elif config.type == "homunculus_arm":
        from .homunculus import HomunculusArm

        return HomunculusArm(config)
    
    # 宇树G1人形机器人
    elif config.type == "unitree_g1":
        from .unitree_g1 import UnitreeG1Teleoperator

        return UnitreeG1Teleoperator(config)
    
    # 双臂SO系列主手
    elif config.type == "bi_so_leader":
        from .bi_so_leader import BiSOLeader

        return BiSOLeader(config)
    
    # Reachy2机器人遥操作
    elif config.type == "reachy2_teleoperator":
        from .reachy2_teleoperator import Reachy2Teleoperator

        return Reachy2Teleoperator(config)
    
    # OpenArm主手
    elif config.type == "openarm_leader":
        from .openarm_leader import OpenArmLeader

        return OpenArmLeader(config)
    
    # 双臂OpenArm主手
    elif config.type == "bi_openarm_leader":
        from .bi_openarm_leader import BiOpenArmLeader

        return BiOpenArmLeader(config)
    
    # OpenArm Mini
    elif config.type == "openarm_mini":
        from .openarm_mini import OpenArmMini

        return OpenArmMini(config)
    
    # 其他类型：尝试使用通用设备工厂
    else:
        try:
            return cast("Teleoperator", make_device_from_device_class(config))
        except Exception as e:
            raise ValueError(f"创建配置 {config} 的机器人时出错: {e}") from e
