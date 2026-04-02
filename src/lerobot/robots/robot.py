"""
机器人基类模块 (Robot Base Class)
===================================

本模块定义了 LeRobot 框架中所有机器人的抽象基类。

主要功能：
- 定义统一的机器人接口
- 割量标定管理（电机偏移等）
- 资源生命周期管理（连接/断开）
- 观测获取和动作发送

所有具体的机器人实现（如 Koch、ALOHA 等）都必须继承此类。
"""

import abc
import builtins
from pathlib import Path

import draccus

from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

from .config import RobotConfig


# TODO(aliberts): 创建类似于 gym.Env 的泛型类型，如 Generic[ObsType, ActType]
# 参考: https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py#L23
class Robot(abc.ABC):
    """
    LeRobot 兼容机器人的基本抽象类。
    
    此类为所有物理机器人提供了标准化接口。
    子类必须实现所有抽象方法和属性才能使用。
    
    属性:
        config_class (RobotConfig): 此机器人期望的配置类。
        name (str): 用于标识此机器人类型的唯一名称。
    
    主要接口方法:
        - connect(): 建立与机器人的通信
        - disconnect(): 断开连接并释放资源
        - calibrate(): 执行机器人标定
        - get_observation(): 获取当前观测值
        - send_action(): 发送动作指令
    """

    # 在所有子类中必须设置这些属性
    config_class: builtins.type[RobotConfig]
    name: str

    def __init__(self, config: RobotConfig):
        self.robot_type = self.name
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir if config.calibration_dir else HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        self.calibration: dict[str, MotorCalibration] = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()

    def __str__(self) -> str:
        return f"{self.id} {self.__class__.__name__}"

    def __enter__(self):
        """
        上下文管理器入口。
        自动连接到机器人。
        """
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        上下文管理器出口。
        自动断开连接，确保即使在发生错误时也能释放资源。
        """
        self.disconnect()

    def __del__(self) -> None:
        """
        析构函数安全网。
        在对象被垃圾回收但未进行清理时尝试断开连接。
        """
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    # TODO(aliberts): create a proper Feature class for this that links with datasets
    @property
    @abc.abstractmethod
    def observation_features(self) -> dict:
        """
        描述机器人产生的观测值的结构和类型的字典。
        其结构（键）应与 :pymeth:`get_observation` 返回值的结构匹配。
        字典的值应为：
            - 如果是简单值则为类型，例如 `float` 用于单个本体感知值（关节的位置/速度）
            - 如果是数组类型值则为表示形状的元组，例如 `(height, width, channel)` 用于图像

        注意：无论机器人是否连接，都应能调用此属性。
        """
        pass

    @property
    @abc.abstractmethod
    def action_features(self) -> dict:
        """
        描述机器人期望的动作的结构和类型的字典。
        其结构（键）应与传递给 :pymeth:`send_action` 的结构匹配。
        字典的值应为值的类型（如果是简单值），例如 `float` 用于单个本体感知值（关节的目标位置/速度）

        注意：无论机器人是否连接，都应能调用此属性。
        """
        pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        机器人当前是否已连接。如果为 `False`，调用 :pymeth:`get_observation` 或
        :pymeth:`send_action` 应抛出错误。
        """
        pass

    @abc.abstractmethod
    def connect(self, calibrate: bool = True) -> None:
        """
        建立与机器人的通信连接。

        参数:
            calibrate (bool): 如果为 True，在连接后自动标定机器人（如果尚未标定或需要标定，具体取决于硬件）。
        """
        pass

    @property
    @abc.abstractmethod
    def is_calibrated(self) -> bool:
        """机器人当前是否已标定。如果不适用则应始终返回 `True`"""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """
        如果适用则标定机器人。如果不适用，此方法应为空操作。

        此方法应收集任何必要的数据（例如电机偏移量）并相应地更新
        :pyattr:`calibration` 字典。
        """
        pass

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """
        从指定文件加载标定数据的辅助方法。

        参数:
            fpath (Path | None): 标定文件的可选路径。默认为 `self.calibration_fpath`。
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath) as f, draccus.config_type("json"):
            self.calibration = draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """
        将标定数据保存到指定文件的辅助方法。

        参数:
            fpath (Path | None): 保存标定文件的可选路径。默认为 `self.calibration_fpath`。
        """
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @abc.abstractmethod
    def configure(self) -> None:
        """
        对机器人应用任何一次性或运行时配置。
        这可能包括设置电机参数、控制模式或初始状态。
        """
        pass

    @abc.abstractmethod
    def get_observation(self) -> RobotObservation:
        """
        从机器人获取当前观测值。

        返回:
            RobotObservation: 表示机器人当前传感器状态的扁平字典。其结构
                应与 :pymeth:`observation_features` 匹配。
        """

        pass

    @abc.abstractmethod
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        向机器人发送动作命令。

        参数:
            action (RobotAction): 表示期望动作的字典。其结构应与
                :pymeth:`action_features` 匹配。

        返回:
            RobotAction: 实际发送给电机的动作，可能被裁剪或修改（例如通过速度安全限制）。
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开与机器人的连接并执行任何必要的清理工作。"""
        pass
