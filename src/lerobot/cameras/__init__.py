"""
摄像头模块 (Camera Module)
=============================

该模块提供了机器人视觉系统的核心组件。
摄像头用于采集视觉观测数据，供策略网络用于决策。

支持的摄像头后端：
- OpenCV: 通用摄像头支持（USB摄像头、网络摄像头等）
- Intel RealSense: 深度摄像头（支持RGB+深度）
- ZMQ: 网络摄像头流（远程视频传输）
- Reachy2: Reachy2机器人专用摄像头

核心组件：
- Camera: 摄像头抽象基类
- CameraConfig: 摄像头配置基类
- ColorMode: 颜色模式枚举 (RGB/BGR)
- Cv2Rotation: OpenCV旋转枚举
- Cv2Backends: OpenCV后端枚举
- make_cameras_from_configs: 从配置创建摄像头的工厂函数
"""

# 摄像头抽象基类
from .camera import Camera
# 配置类和枚举
from .configs import CameraConfig, ColorMode, Cv2Backends, Cv2Rotation
# 工厂函数
from .utils import make_cameras_from_configs
