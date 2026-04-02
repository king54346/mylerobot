"""
遥操作器配置模块 (Teleoperator Configuration)
==========================================

定义遥操作器的配置基类。
所有具体遥操作器实现都需继承此基类。
"""

import abc
from dataclasses import dataclass
from pathlib import Path

import draccus


@dataclass(kw_only=True)
class TeleoperatorConfig(draccus.ChoiceRegistry, abc.ABC):
    """
    遥操作器配置基类。
    
    所有遥操作器配置都需继承此类。
    使用draccus的ChoiceRegistry支持配置注册和动态实例化。
    
    Attributes:
        id: 遥操作器的唯一标识符，用于区分同类型的不同设备
        calibration_dir: 校准文件存储目录
    """
    # 用于区分同类型不同遥操作器的唯一标识符
    id: str | None = None
    # 存储校准文件的目录
    calibration_dir: Path | None = None

    @property
    def type(self) -> str:
        """获取遥操作器类型名称。"""
        return self.get_choice_name(self.__class__)
