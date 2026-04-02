"""
机器人配置模块 (Robot Configuration Module)
==========================================

本模块定义了机器人配置的基类。

主要类:
    - RobotConfig: 所有机器人配置的抽象基类
"""

import abc
from dataclasses import dataclass
from pathlib import Path

import draccus


@dataclass(kw_only=True)
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    """
    机器人配置基类。
    
    属性:
        id: 机器人唯一标识符，用于区分同类型的不同机器人
        calibration_dir: 存储标定文件的目录
    """
    # 用于区分同类型的不同机器人
    id: str | None = None
    # 存储标定文件的目录
    calibration_dir: Path | None = None

    def __post_init__(self):
        """初始化后验证摄像头配置。"""
        if hasattr(self, "cameras") and self.cameras:
            for _, config in self.cameras.items():
                for attr in ["width", "height", "fps"]:
                    if getattr(config, attr) is None:
                        raise ValueError(
                            f"摄像头用于机器人时必须指定 '{attr}' 参数"
                        )

    @property
    def type(self) -> str:
        """返回机器人类型名称。"""
        return self.get_choice_name(self.__class__)
