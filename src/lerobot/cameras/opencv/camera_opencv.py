"""
Provides the OpenCVCamera class for capturing frames from cameras using OpenCV.
"""

import logging
import math
import os
import platform
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

# Fix MSMF hardware transform compatibility for Windows before importing cv2
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2  # type: ignore  # TODO: add type stubs for OpenCV

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_rotation
from .configuration_opencv import ColorMode, OpenCVCameraConfig

# NOTE(Steven): The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60

logger = logging.getLogger(__name__)

# 方法  阻塞?  适用场景
# read()阻塞，等下一帧精确同步采集
# async_read(timeout_ms)等待最多 N 毫秒一般数据收集
# read_latest()不阻塞，直接返回缓存对延迟敏感的控制循环
class OpenCVCamera(Camera):
    """
        管理使用 OpenCV 进行高效帧记录的摄像头交互。
        这个类提供了一个高级接口来连接、配置和读取与 OpenCV 的 VideoCapture 兼容的摄像头的帧。它支持同步和异步帧读取。
        一个 OpenCVCamera 实例需要一个摄像头索引（例如 0）或设备路径（例如 Linux 上的 '/dev/video0'）。摄像头索引在重启或端口更改时可能不稳定，特别是在 Linux 上。使用提供的实用程序脚本查找可用的摄像头索引或路径：
        ```bash
        lerobot-find-cameras opencv
        ```
        摄像头的默认设置（FPS、分辨率、颜色模式）将被使用，除非在配置中覆盖。
        示例：
            ```python
            from lerobot.cameras.opencv import OpenCVCamera
            from lerobot.cameras.configuration_opencv import OpenCVCameraConfig

            # 使用摄像头索引 0 的基本用法
            config = OpenCVCameraConfig(index_or_path=0)
            camera = OpenCVCamera(config)
            camera.connect()

            # 同步读取 1 帧（阻塞）
            color_image = camera.read()

            # 异步读取 1 帧（等待新帧，带超时）
            async_image = camera.async_read()

            # 立即获取最新帧（不等待，返回时间戳）
            latest_image, timestamp = camera.read_latest()

            # 完成后，使用以下方法正确断开摄像头连接
            camera.disconnect()
            ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        """
        Initializes the OpenCVCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config
        self.index_or_path = config.index_or_path

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s # 连接后等待摄像头稳定，避免最初几帧曝光异常

        self.videocapture: cv2.VideoCapture | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.backend: int = config.backend
        # 旋转 90° 时，采集分辨率和输出分辨率会互换宽高，这个细节处理得比较细
        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        return isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """
        连接 到指定的 OpenCV 摄像头
        初始化 OpenCV VideoCapture 对象，设置相机属性（FPS、宽度、高度），启动后台读取线程并执行初始检查。
        参数：
            warmup (bool): 如果为 True，则在 connect() 时等待，直到后台线程
            捕获到至少一帧有效帧。默认为 True。
        异常：
            DeviceAlreadyConnectedError: 如果摄像头已连接。
            ConnectionError: 如果指定的摄像头索引/路径未找到或无法打开。
            RuntimeError: 如果摄像头打开但无法应用请求的设置。
        """

        # Use 1 thread for OpenCV operations to avoid potential conflicts or
        # blocking in multi-threaded applications, especially during data collection.
        cv2.setNumThreads(1)

        self.videocapture = cv2.VideoCapture(self.index_or_path, self.backend)

        if not self.videocapture.isOpened():
            self.videocapture.release()
            self.videocapture = None
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras opencv` to find available cameras."
            )

        self._configure_capture_settings()
        self._start_read_thread()

        if warmup and self.warmup_s > 0:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.async_read(timeout_ms=self.warmup_s * 1000)
                time.sleep(0.1)
            with self.frame_lock:
                if self.latest_frame is None:
                    raise ConnectionError(f"{self} failed to capture frames during warmup.")

        logger.info(f"{self} connected.")

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        """
        Applies the specified FOURCC, FPS, width, and height settings to the connected camera.

        This method attempts to set the camera properties via OpenCV. It checks if
        the camera successfully applied the settings and raises an error if not.
        FOURCC is set first (if specified) as it can affect the available FPS and resolution options.

        Args:
            fourcc: The desired FOURCC code (e.g., "MJPG", "YUYV"). If None, auto-detect.
            fps: The desired frames per second. If None, the setting is skipped.
            width: The desired capture width. If None, the setting is skipped.
            height: The desired capture height. If None, the setting is skipped.

        Raises:
            RuntimeError: If the camera fails to set any of the specified properties
                          to the requested value.
            DeviceNotConnectedError: If the camera is not connected.
        """

        # Set FOURCC first (if specified) as it can affect available FPS/resolution options
        if self.config.fourcc is not None:
            self._validate_fourcc()
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        default_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        default_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.width is None or self.height is None:
            self.width, self.height = default_width, default_height
            self.capture_width, self.capture_height = default_width, default_height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = default_height, default_width
                self.capture_width, self.capture_height = default_width, default_height
        else:
            self._validate_width_and_height()

        if self.fps is None:
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        else:
            self._validate_fps()

    def _validate_fps(self) -> None:
        """Validates and sets the camera's frames per second (FPS)."""

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.fps is None:
            raise ValueError(f"{self} FPS is not set")

        success = self.videocapture.set(cv2.CAP_PROP_FPS, float(self.fps))
        actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        # Use math.isclose for robust float comparison
        if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            raise RuntimeError(f"{self} failed to set fps={self.fps} ({actual_fps=}).")

    def _validate_fourcc(self) -> None:
        """Validates and sets the camera's FOURCC code."""

        fourcc_code = cv2.VideoWriter_fourcc(*self.config.fourcc)

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        success = self.videocapture.set(cv2.CAP_PROP_FOURCC, fourcc_code)
        actual_fourcc_code = self.videocapture.get(cv2.CAP_PROP_FOURCC)

        # Convert actual FOURCC code back to string for comparison
        actual_fourcc_code_int = int(actual_fourcc_code)
        actual_fourcc = "".join([chr((actual_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])

        if not success or actual_fourcc != self.config.fourcc:
            logger.warning(
                f"{self} failed to set fourcc={self.config.fourcc} (actual={actual_fourcc}, success={success}). "
                f"Continuing with default format."
            )

    def _validate_width_and_height(self) -> None:
        """Validates and sets the camera's frame capture width and height."""

        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        if self.capture_width is None or self.capture_height is None:
            raise ValueError(f"{self} capture_width or capture_height is not set")

        width_success = self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        height_success = self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))

        actual_width = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if not width_success or self.capture_width != actual_width:
            raise RuntimeError(
                f"{self} failed to set capture_width={self.capture_width} ({actual_width=}, {width_success=})."
            )

        actual_height = int(round(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not height_success or self.capture_height != actual_height:
            raise RuntimeError(
                f"{self} failed to set capture_height={self.capture_height} ({actual_height=}, {height_success=})."
            )

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        检测连接到系统的可用摄像头。
        在 Linux 上，它扫描 '/dev/video*' 路径。在其他系统（如 macOS、Windows）上，
        它检查从 0 到 `MAX_OPENCV_INDEX` 的索引。
        返回值：
            List[Dict[str, Any]]: 一个字典列表，每个字典包含 'type'、'id'（端口索引或路径）和默认配置属性（宽度、高度、fps、格式）。
        """
        found_cameras_info = []

        targets_to_scan: list[str | int]
        if platform.system() == "Linux":
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
        else:
            targets_to_scan = [int(i) for i in range(MAX_OPENCV_INDEX)]

        for target in targets_to_scan:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_format = camera.get(cv2.CAP_PROP_FORMAT)

                # Get FOURCC code and convert to string
                default_fourcc_code = camera.get(cv2.CAP_PROP_FOURCC)
                default_fourcc_code_int = int(default_fourcc_code)
                default_fourcc = "".join([chr((default_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])

                camera_info = {
                    "name": f"OpenCV Camera @ {target}",
                    "type": "OpenCV",
                    "id": target,
                    "backend_api": camera.getBackendName(),
                    "default_stream_profile": {
                        "format": default_format,
                        "fourcc": default_fourcc,
                        "width": default_width,
                        "height": default_height,
                        "fps": default_fps,
                    },
                }

                found_cameras_info.append(camera_info)
                camera.release()

        return found_cameras_info

    def _read_from_hardware(self) -> NDArray[Any]:
        if self.videocapture is None:
            raise DeviceNotConnectedError(f"{self} videocapture is not initialized")

        ret, frame = self.videocapture.read()

        if not ret:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        return frame

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        读取单帧的同步方法。这是一个阻塞调用，等待来自摄像头硬件的下一个可用帧通过OpenCV。
        返回值：
            np.ndarray：捕获的帧作为NumPy数组，格式为（高度，宽度，通道数），使用指定或默认的颜色模式，并应用任何配置的旋转。
        异常：
            DeviceNotConnectedError：如果摄像头未连接。
            RuntimeError：如果从摄像头读取帧失败，或者接收到的
                            帧尺寸在旋转前与预期不符。
            ValueError：如果请求了无效的 `color_mode`。

        """

        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        frame = self.async_read(timeout_ms=10000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """
        应用颜色转换、尺寸验证和旋转到原始帧。
        参数：
            image (np.ndarray): 原始图像帧（OpenCV默认的BGR 格式）。
            返回：
            np.ndarray: 处理后的图像帧。
            异常：
            ValueError: 如果请求的 `color_mode` 无效。
            RuntimeError: 如果原始帧的尺寸与配置的 `width` 和 `

        """

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if self.color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    # 后台线程持续从硬件读帧，存入 latest_frame
    def _read_loop(self) -> None:
        """
        内部循环由后台线程运行，用于异步读取。
        每次迭代：
        1. 读取一帧彩色图像
        2. 将结果存储在 latest_frame 中并更新时间戳（线程安全）
        3. 设置 new_frame_event 以通知监听器

        在 DeviceNotConnectedError 上停止，在其他错误上记录并继续。
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                raw_frame = self._read_from_hardware() # 从摄像头读
                processed_frame = self._postprocess_image(raw_frame) # BGR转RGB + 旋转
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = processed_frame  # 线程安全地存帧
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()  # 通知：有新帧了
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        读取最新可用帧的异步方法。
        此方法检索后台读取线程捕获的最新帧。它不会直接阻塞等待摄像头硬件，
        但可能会等待最多 timeout_ms 毫秒，以便后台线程提供帧。在高FPS下，它是“尽力而为”的。
                Args:
                    timeout_ms (float): 等待帧可用的最长时间（毫秒）。
                Returns:
                    np.ndarray: 最新捕获的帧作为NumPy数组，格式为（高度，宽度，通道数），根据配置进行处理。
                Raises:
                    DeviceNotConnectedError: 如果摄像头未连接。
                    TimeoutError: 如果在指定的超时时间内没有帧可用。
                    RuntimeError: 如果发生意外错误。

        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """
        返回最近采集的帧（窃视模式）。
        此方法是非阻塞的，返回当前内存缓冲区中的内容。返回的帧可能是过时的，
        意味着它可能是很久之前采集的（例如摄像头卡死场景）。
        使用场景:
            适合需要零延迟或解耦频率的场景，以及需要保证获取帧的情况，
            例如UI可视化、日志记录或非关键性监控。
            Args:
                max_age_ms: The maximum allowed age of the frame in milliseconds. Default is 500ms.
            Returns:
                NDArray[Any]: The frame image (numpy array).
            Raises:
                TimeoutError: If the latest frame is older than `max_age_ms`.
                NotConnectedError: If the camera is not connected.
                RuntimeError: If the camera is connected but has not captured any frames yet.

        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        """
        Disconnects from the camera and cleans up resources.

        Stops the background read thread (if running) and releases the OpenCV
        VideoCapture object.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
