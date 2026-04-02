"""
Provides the Reachy2Camera class for capturing frames from Reachy 2 cameras using Reachy 2's CameraManager.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

# Fix MSMF hardware transform compatibility for Windows before importing cv2
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy

from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _reachy2_sdk_available

if TYPE_CHECKING or _reachy2_sdk_available:
    from reachy2_sdk.media.camera import CameraView
    from reachy2_sdk.media.camera_manager import CameraManager
else:
    CameraManager = None

    class CameraView:
        LEFT = 0
        RIGHT = 1


from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from .configuration_reachy2_camera import ColorMode, Reachy2CameraConfig

logger = logging.getLogger(__name__)


class Reachy2Camera(Camera):
    """
    管理 Reachy 2 摄像头，使用 Reachy 2 的 CameraManager 提供高层接口连接、配置和读取摄像头帧。支持同步和异步读取。
    需要在配置中指定摄像头名称（如 "teleop"）和图像类型（如 "left"）。默认设置（FPS、分辨率、颜色模式）除非在配置中覆盖，否则使用默认值。
    """

    def __init__(self, config: Reachy2CameraConfig):
        """
        Initializes the Reachy2Camera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config

        self.color_mode = config.color_mode
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None

        self.cam_manager: CameraManager | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name}, {self.config.image_type})"


    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        if self.config.name == "teleop":
            return bool(
                self.cam_manager._grpc_connected and self.cam_manager.teleop if self.cam_manager else False
            )
        elif self.config.name == "depth":
            return bool(
                self.cam_manager._grpc_connected and self.cam_manager.depth if self.cam_manager else False
            )
        else:
            raise ValueError(f"Invalid camera name '{self.config.name}'. Expected 'teleop' or 'depth'.")
    # 连接方式 IP地址 + 端口
    def connect(self, warmup: bool = True) -> None:
        """
        连接 Reachy 2 摄像头，使用配置中指定的 IP 地址和端口连接到 CameraManager。
        错误:
            DeviceNotConnectedError: 如果无法连接到摄像头。
        """
        self.cam_manager = CameraManager(host=self.config.ip_address, port=self.config.port)
        if self.cam_manager is None:
            raise DeviceNotConnectedError(f"Could not connect to {self}.")
        self.cam_manager.initialize_cameras()

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detection not implemented for Reachy2 cameras.
        """
        raise NotImplementedError("Camera detection is not implemented for Reachy2 cameras.")

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
       读取摄像头帧的主要方法，直接从 Reachy 2 的底层软件获取最新的帧。根据配置处理颜色模式和旋转。
        返回:
            np.ndarray: 采集的帧作为numpy数组，格式为 (height, width, channels)，根据配置使用指定或默认的颜色模式，并应用任何配置的旋转。
        """
        start_time = time.perf_counter()

        if self.cam_manager is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        frame: NDArray[Any] = np.empty((0, 0, 3), dtype=np.uint8)
        # 两种物理摄像头
        # teleop 摄像头 —— 头部双目摄像头（用于遥操作） CameraView.LEFT CameraView.RIGHT
        # depth 摄像头 —— 类似 RealSense（用于感知）获取深度图和 RGB图
        if self.config.name == "teleop" and hasattr(self.cam_manager, "teleop"):
            if self.config.image_type == "left":
                frame = self.cam_manager.teleop.get_frame(
                    CameraView.LEFT, size=(self.config.width, self.config.height)
                )[0]
            elif self.config.image_type == "right":
                frame = self.cam_manager.teleop.get_frame(
                    CameraView.RIGHT, size=(self.config.width, self.config.height)
                )[0]
        elif self.config.name == "depth" and hasattr(self.cam_manager, "depth"):
            if self.config.image_type == "depth":
                frame = self.cam_manager.depth.get_depth_frame()[0]
            elif self.config.image_type == "rgb":
                frame = self.cam_manager.depth.get_frame(size=(self.config.width, self.config.height))[0]
        else:
            raise ValueError(f"Invalid camera name '{self.config.name}'. Expected 'teleop' or 'depth'.")

        if frame is None:
            raise RuntimeError(f"Internal error: No frame available for {self}.")

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )
        if self.color_mode == ColorMode.RGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.latest_frame = frame
        self.latest_timestamp = time.perf_counter()

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        同read
        返回：
            np.ndarray: 采集的帧作为numpy数组，格式为 (height, width, channels)，根据配置使用指定或默认的颜色模式，并应用任何配置的旋转。
        错误：
            DeviceNotConnectedError: 如果摄像头未连接。
            TimeoutError: 如果在指定的超时时间内没有帧可用。
            RuntimeError: 如果发生意外错误。
        """

        return self.read()

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent frame captured immediately (Peeking).

        This method is non-blocking and returns whatever is currently in the
        memory buffer. The frame may be stale,
        meaning it could have been captured a while ago (hanging camera scenario e.g.).

        Returns:
            tuple[NDArray, float]:
                - The frame image (numpy array).
                - The timestamp (time.perf_counter) when this frame was captured.

        Raises:
            TimeoutError: If the latest frame is older than `max_age_ms`.
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera is connected but has not captured any frames yet.
        """

        if self.latest_frame is None or self.latest_timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - self.latest_timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return self.latest_frame

    @check_if_not_connected
    def disconnect(self) -> None:
        """
        Stops the background read thread (if running).

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """

        if self.cam_manager is not None:
            self.cam_manager.disconnect()

        logger.info(f"{self} disconnected.")
