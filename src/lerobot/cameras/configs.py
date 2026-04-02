"""
摄像头配置模块 (Camera Configuration)
======================================

定义摄像头配置相关的枚举类和基类。

枚举类型：
- ColorMode: 颜色模式 (RGB/BGR)
- Cv2Rotation: 图像旋转角度
- Cv2Backends: OpenCV视频采集后端
"""

import abc
from dataclasses import dataclass
from enum import Enum

import draccus  # type: ignore  # TODO: add type stubs for draccus


class ColorMode(str, Enum):
    """
    颜色模式枚举。
    
    定义图像的颜色通道顺序。
    
    值:
        RGB: 红绿蓝顺序（常用于深度学习模型）
        BGR: 蓝绿红顺序（OpenCV默认格式）
    """
    RGB = "rgb"  # 红绿蓝顺序
    BGR = "bgr"  # 蓝绿红顺序 (OpenCV默认)

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"`color_mode` 应为 {list(cls)} 之一，但提供的是 {value}。")


class Cv2Rotation(int, Enum):
    """
    图像旋转角度枚举。
    
    定义图像的旋转角度，用于矫正摄像头安装方向。
    
    值:
        NO_ROTATION: 无旋转 (0°)
        ROTATE_90: 顺时针旋转90°
        ROTATE_180: 旋转180°
        ROTATE_270: 顺时针旋转270° (反时针旋转90°)
    """
    NO_ROTATION = 0     # 无旋转
    ROTATE_90 = 90      # 旋转90°
    ROTATE_180 = 180    # 旋转180°
    ROTATE_270 = -90    # 旋转270° (等价于反时针旋转90°)

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"`rotation` 应为 {list(cls)} 之一，但提供的是 {value}。")


# OpenCV视频采集后端（部分值来自 https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html）
class Cv2Backends(int, Enum):
    """
    OpenCV视频采集后端枚举。
    
    定义可用的视频采集后端，不同平台支持的后端不同。
    
    值:
        ANY: 自动选择可用后端
        V4L2: Linux Video4Linux2 (Linux平台)
        DSHOW: DirectShow (Windows平台)
        PVAPI: Prosilica GigE SDK
        ANDROID: Android原生后端
        AVFOUNDATION: AVFoundation (macOS/iOS平台)
        MSMF: Microsoft Media Foundation (Windows平台)
    """
    ANY = 0             # 自动选择
    V4L2 = 200          # Linux V4L2
    DSHOW = 700         # Windows DirectShow
    PVAPI = 800         # Prosilica GigE
    ANDROID = 1000      # Android
    AVFOUNDATION = 1200 # macOS/iOS
    MSMF = 1400         # Windows Media Foundation

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"`backend` 应为 {list(cls)} 之一，但提供的是 {value}。")


@dataclass(kw_only=True)
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore  # TODO: add type stubs for draccus
    """
    摄像头配置基类。
    
    所有摄像头配置都需继承此类。
    使用draccus的ChoiceRegistry支持配置注册和动态实例化。
    
    Attributes:
        fps: 帧率（每秒帧数），可选
        width: 图像宽度（像素），可选
        height: 图像高度（像素），可选
    """
    fps: int | None = None     # 帧率
    width: int | None = None   # 图像宽度
    height: int | None = None  # 图像高度

    @property
    def type(self) -> str:
        """获取摄像头类型名称。"""
        return str(self.get_choice_name(self.__class__))
