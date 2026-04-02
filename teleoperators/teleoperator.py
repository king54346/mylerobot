"""
遥操作器基类模块 (Teleoperator Base Class)
==========================================

定义所有遥操作设备的抽象基类。
提供统一的接口用于人类操作员远程控制机器人。

主要功能：
- 统一的连接/断开接口
- 校准数据管理
- 动作读取和反馈发送
- 上下文管理器支持
"""

import abc
import builtins
from pathlib import Path
from typing import Any

import draccus

from lerobot.motors.motors_bus import MotorCalibration
from lerobot.processor import RobotAction
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from .config import TeleoperatorConfig


class Teleoperator(abc.ABC):
    """
    所有LeRobot兼容遥操作设备的抽象基类。

    此类提供了与物理遥操作设备交互的标准化接口。
    子类必须实现所有抽象方法和属性才能使用。

    类属性:
        config_class: 此遥操作器期望的配置类
        name: 用于标识此遥操作器类型的唯一名称
    
    实例属性:
        id: 遥操作器实例的唯一标识符
        calibration_dir: 校准文件存储目录
        calibration_fpath: 校准文件完整路径
        calibration: 电机校准数据字典
    
    抽象方法:
        - connect(): 建立连接
        - disconnect(): 断开连接
        - get_action(): 获取当前动作
        - send_feedback(): 发送反馈动作
        - calibrate(): 执行校准
        - configure(): 应用配置
    """

    # 在所有子类中必须设置这些类属性
    config_class: builtins.type[TeleoperatorConfig]  # 配置类
    name: str  # 遥操作器类型名称

    def __init__(self, config: TeleoperatorConfig):
        """
        初始化遥操作器。
        
        Args:
            config: 遥操作器配置对象
        """
        self.id = config.id  # 遥操作器唯一标识符
        # 设置校准文件目录，优先使用配置中指定的目录
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"  # 校准文件路径
        self.calibration: dict[str, MotorCalibration] = {}  # 电机校准数据
        # 如果校准文件存在，加载校准数据
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    def __enter__(self):
        """
        上下文管理器入口。
        自动连接到遥操作器。
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        上下文管理器出口。
        自动断开连接，确保即使发生错误也能释放资源。
        """
        self.disconnect()

    def __del__(self) -> None:
        """
        析构函数安全网。
        如果对象在未清理的情况下被垃圾回收，尝试断开连接。
        """
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        描述遥操作器产生的动作结构和类型的字典。
        
        其结构（键）应与 get_action() 返回的结构匹配。
        字典的值应该是类型，例如单个本体感知值（关节目标位置/速度）使用 `float`。
        
        注意: 此属性应在机器人连接或未连接时都能被调用。
        
        Returns:
            dict: 动作特征描述字典
        """
        pass

    @property
    @abc.abstractmethod
    def feedback_features(self) -> dict:
        """
        描述机器人期望的反馈动作结构和类型的字典。
        
        其结构（键）应与传递给 send_feedback() 的结构匹配。
        字典的值应该是类型，例如单个本体感知值（关节目标位置/速度）使用 `float`。
        
        注意: 此属性应在机器人连接或未连接时都能被调用。
        
        Returns:
            dict: 反馈特征描述字典
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        检查遥操作器是否已连接。
        
        如果返回 False，调用 get_action() 或 send_feedback() 应抛出错误。
        
        Returns:
            bool: 连接状态
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        与遥操作器建立通信。

        Args:
            calibrate: 如果为True，在连接后自动校准遥操作器（如果未校准或需要校准，
                这取决于硬件）。默认为True。
        """
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """
        检查遥操作器是否已校准。
        
        如果校准不适用于此遥操作器，应始终返回 True。
        
        Returns:
            bool: 校准状态
        """
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """
        如果适用，校准遥操作器。如果不适用，应为空操作。

        此方法应收集必要的数据（例如电机偏移量）并相应地
        更新 calibration 字典。
        """
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """
        从指定文件加载校准数据的辅助方法。

        Args:
            fpath: 校准文件的可选路径。默认为 self.calibration_fpath。
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        将校准数据保存到指定文件的辅助方法。

        Args:
            fpath: 保存校准文件的可选路径。默认为 self.calibration_fpath。
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        """
        应用任何一次性或运行时配置到遥操作器。
        
        这可能包括设置电机参数、控制模式或初始状态。
        """
        pass

    @abc.abstractmethod
    def get_action(self) -> RobotAction:
        """
        从遥操作器获取当前动作。

        Returns:
            RobotAction: 表示遥操作器当前动作的扁平字典。
                其结构应与 action_features 匹配。
        """
        pass

    @abc.abstractmethod
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """
        向遥操作器发送反馈动作命令。

        Args:
            feedback: 表示期望反馈的字典。其结构应与 feedback_features 匹配。

        注意:
            实际发送到电机的动作可能被裁剪或修改，例如受到速度安全限制。
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开与遥操作器的连接并执行必要的清理。"""
        pass
