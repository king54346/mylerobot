"""
摄像头工具模块 (Camera Utilities)
==================================

提供摄像头创建和图像处理的工具函数。

主要函数：
- make_cameras_from_configs: 根据配置批量创建摄像头实例
- get_cv2_rotation: 获取OpenCV旋转常量
"""

from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    """
    根据配置字典批量创建摄像头实例。
    
    根据每个配置的 type 字段动态导入并实例化相应的摄像头类。
    
    Args:
        camera_configs: 摄像头配置字典，键为摄像头名称，值为配置对象
        
    Returns:
        dict[str, Camera]: 摄像头实例字典，键与输入相同
        
    Raises:
        ValueError: 如果配置类型不受支持
        
    支持的摄像头类型：
        - opencv: OpenCV通用摄像头（USB摄像头、网络摄像头等）
        - intelrealsense: Intel RealSense深度摄像头
        - reachy2_camera: Reachy2机器人专用摄像头
        - zmq: ZMQ网络摄像头（远程视频流）
    """
    cameras: dict[str, Camera] = {}

    for key, cfg in camera_configs.items():
        # TODO(Steven): 考虑对所有类型都使用 make_device_from_device_class
        
        # OpenCV通用摄像头
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        # Intel RealSense深度摄像头
        elif cfg.type == "intelrealsense":
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        # Reachy2机器人专用摄像头
        elif cfg.type == "reachy2_camera":
            from .reachy2_camera.reachy2_camera import Reachy2Camera

            cameras[key] = Reachy2Camera(cfg)

        # ZMQ网络摄像头
        elif cfg.type == "zmq":
            from .zmq.camera_zmq import ZMQCamera

            cameras[key] = ZMQCamera(cfg)

        # 其他类型：尝试使用通用设备工厂
        else:
            try:
                cameras[key] = cast(Camera, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"创建摄像头 {key}（配置 {cfg}）时出错: {e}") from e

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    """
    将 Cv2Rotation 枚举值转换为 OpenCV 旋转常量。
    
    Args:
        rotation: 旋转角度枚举值
        
    Returns:
        int | None: OpenCV旋转常量，或无旋转时返回None
    """
    import cv2  # type: ignore  # TODO: add type stubs for OpenCV

    if rotation == Cv2Rotation.ROTATE_90:
        return int(cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90°
    elif rotation == Cv2Rotation.ROTATE_180:
        return int(cv2.ROTATE_180)  # 旋转180°
    elif rotation == Cv2Rotation.ROTATE_270:
        return int(cv2.ROTATE_90_COUNTERCLOCKWISE)  # 反时针旋转90° (等价于顺时针270°)
    else:
        return None  # 无旋转
