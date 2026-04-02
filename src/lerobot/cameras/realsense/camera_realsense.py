"""
Provides the RealSenseCamera class for capturing frames from Intel RealSense cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

try:
    import pyrealsense2 as rs  # type: ignore  # TODO: add type stubs for pyrealsense2
except Exception as e:
    logging.info(f"Could not import realsense: {e}")

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)


class RealSenseCamera(Camera):
    """
    管理英特尔实感摄像头的帧和深度记录交互
    该类提供了一个类似于 `OpenCVCamera` 的接口，但专门针对 RealSense 设备，利用 `pyrealsense2` 库。它使用摄像头的唯一序列号进行识别，在 Linux 上比设备索引更稳定。它还支持同时捕获深度图和彩色帧。
    使用提供的实用程序脚本查找可用的摄像头索引和默认配置文件：
        ```bash
        lerobot-find-cameras realsense
        ```
        `RealSenseCamera` 实例需要一个配置对象，指定摄像头的序
                    列号或唯一设备名称。如果使用名称，请确保只连接了一个具有该名称的摄像头。
                    摄像头的默认设置（FPS、分辨率、颜色模式）来自流
                    配置文件，除非在配置中覆盖。
    例子：
        ```python
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # 使用序列号的基本用法
        config = RealSenseCameraConfig(serial_number_or_name="0123456789") # 替换为实际 SN
        camera = RealSenseCamera(config)
        camera.connect()
        # 同步读取 1 帧（阻塞）
        color_image = camera.read()
        # 异步读取 1 帧（等待新帧，带超时
        async_image = camera.async_read()
        # 立即获取最新帧（不等待，返回时间戳）
        latest_image, timestamp = camera.read_latest()
        # 使用深度捕获和自定义设置的示例
        custom_config = RealSenseCameraConfig(
            serial_number_or_name="0123456789", # 替换为实际 SN
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR, # 请求 BGR 输出
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        )
        depth_camera = RealSenseCamera(custom_config)
        depth_camera.connect()
        # 读取 1 帧深度
        depth_map = depth_camera.read_depth()
        # 使用唯一摄像头名称的示例
        name_config = RealSenseCameraConfig(serial_number_or_name="Intel RealSense D435") # 如果唯一
        name_camera = RealSenseCamera(name_config)
        # ... 连接、读取、断开连接 ...

    """

    def __init__(self, config: RealSenseCameraConfig):
        """
        Initializes the RealSenseCamera instance.

        Args:
            config: The configuration settings for the camera.
        """

        super().__init__(config)

        self.config = config

        if config.serial_number_or_name.isdigit():
            self.serial_number = config.serial_number_or_name
        else:
            self.serial_number = self._find_serial_number_from_name(config.serial_number_or_name)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.rs_pipeline: rs.pipeline | None = None  # 读取通道
        self.rs_profile: rs.pipeline_profile | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started and streams are active."""
        return self.rs_pipeline is not None and self.rs_profile is not None

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """
        连接到指定的 RealSense 摄像头。
        初始化 RealSense 管道，配置所需的流（颜色和可选的深度），启动管道，并验证实际流设置。
        参数:
            warmup (bool): 如果为 True，则在 connect() 时等待，直到后台线程捕获到至少一帧有效帧。默认为 True。
        异常:
            DeviceAlreadyConnectedError: 如果摄像头已经连接。
            ValueError: 如果配置无效（例如，缺少序列号/名称， 名称不唯一）。
            ConnectionError: 如果找到摄像头但无法启动管道，或者根本没有检测到 RealSense 设备。
            RuntimeError: 如果管道启动但无法应用请求的设置。
        """

        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        self._configure_rs_pipeline_config(rs_config)

        try:
            self.rs_profile = self.rs_pipeline.start(rs_config)
        except RuntimeError as e:
            self.rs_profile = None
            self.rs_pipeline = None
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras realsense` to find available cameras."
            ) from e

        self._configure_capture_settings()
        self._start_read_thread()

        # NOTE(Steven/Caroline): Enforcing at least one second of warmup as RS cameras need a bit of time before the first read. If we don't wait, the first read from the warmup will raise.
        self.warmup_s = max(self.warmup_s, 1)

        start_time = time.time()
        while time.time() - start_time < self.warmup_s:
            self.async_read(timeout_ms=self.warmup_s * 1000)
            time.sleep(0.1)
        with self.frame_lock:
            if self.latest_color_frame is None or self.use_depth and self.latest_depth_frame is None:
                raise ConnectionError(f"{self} failed to capture frames during warmup.")

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Intel RealSense cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (serial number), 'name',
            firmware version, USB type, and other available specs, and the default profile properties (width, height, fps, format).

        Raises:
            OSError: If pyrealsense2 is not installed.
            ImportError: If pyrealsense2 is not installed.
        """
        found_cameras_info = []
        context = rs.context()
        devices = context.query_devices()

        for device in devices:
            camera_info = {
                "name": device.get_info(rs.camera_info.name),
                "type": "RealSense",
                "id": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "product_id": device.get_info(rs.camera_info.product_id),
                "product_line": device.get_info(rs.camera_info.product_line),
            }

            # Get stream profiles for each sensor
            sensors = device.query_sensors()
            for sensor in sensors:
                profiles = sensor.get_stream_profiles()

                for profile in profiles:
                    if profile.is_video_stream_profile() and profile.is_default():
                        vprofile = profile.as_video_stream_profile()
                        stream_info = {
                            "stream_type": vprofile.stream_name(),
                            "format": vprofile.format().name,
                            "width": vprofile.width(),
                            "height": vprofile.height(),
                            "fps": vprofile.fps(),
                        }
                        camera_info["default_stream_profile"] = stream_info

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """Finds the serial number for a given unique camera name."""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No RealSense camera found with name '{name}'. Available camera names: {available_names}"
            )

        if len(found_devices) > 1:
            serial_numbers = [dev["serial_number"] for dev in found_devices]
            raise ValueError(
                f"Multiple RealSense cameras found with name '{name}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )

        serial_number = str(found_devices[0]["serial_number"])
        return serial_number

    def _configure_rs_pipeline_config(self, rs_config: Any) -> None:
        """Creates and configures the RealSense pipeline configuration object."""
        rs.config.enable_device(rs_config, self.serial_number)

        if self.width and self.height and self.fps:
            rs_config.enable_stream(
                rs.stream.color, self.capture_width, self.capture_height, rs.format.rgb8, self.fps
            )
            if self.use_depth:
                rs_config.enable_stream(
                    rs.stream.depth, self.capture_width, self.capture_height, rs.format.z16, self.fps
                )
        else:
            rs_config.enable_stream(rs.stream.color)
            if self.use_depth:
                rs_config.enable_stream(rs.stream.depth)

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        """Sets fps, width, and height from device stream if not already configured.

        Uses the color stream profile to update unset attributes. Handles rotation by
        swapping width/height when needed. Original capture dimensions are always stored.

        Raises:
            DeviceNotConnectedError: If device is not connected.
        """

        if self.rs_profile is None:
            raise RuntimeError(f"{self}: rs_profile must be initialized before use.")

        stream = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()

        if self.fps is None:
            self.fps = stream.fps()

        if self.width is None or self.height is None:
            actual_width = int(round(stream.width()))
            actual_height = int(round(stream.height()))
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """
        读取 单 帧（深度）同步从摄像头。
        这是一个阻塞调用。它等待来自摄像头硬件的连贯帧集（深度）
        通过 RealSense 管道。
        返回：
            np.ndarray: 深度图作为 NumPy 数组（高度、宽度） 类型为 `np.uint16`（以毫米为单位的原始深度值 ）和旋转。
        异常：
            DeviceNotConnectedError: 如果摄像头未连接。
            RuntimeError: 如果从管道读取帧失败或帧无效。

        """
        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        # 复用 async_read() 触发一次帧同步
        _ = self.async_read(timeout_ms=10000)
        # 然后单独取深度帧
        with self.frame_lock:
            depth_map = self.latest_depth_frame

        if depth_map is None:
            raise RuntimeError("No depth frame available. Ensure camera is streaming.")

        return depth_map # shape: (height, width)，没有通道维度

    # 获取 composite_frame 的颜色帧和深度帧，进行后处理，并存储在 latest_color_frame 和 latest_depth_frame 中。它还更新 latest_timestamp 并设置 new_frame_event 以通知等待的读取器。
    def _read_from_hardware(self):
        if self.rs_pipeline is None:
            raise RuntimeError(f"{self}: rs_pipeline must be initialized before use.")

        ret, frame = self.rs_pipeline.try_wait_for_frames(timeout_ms=10000)

        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        return frame

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 0) -> NDArray[Any]:
        """
        Reads a single frame (color) synchronously from the camera.

        This is a blocking call. It waits for a coherent set of frames (color)
        from the camera hardware via the RealSense pipeline.

        Returns:
            np.ndarray: The captured color frame as a NumPy array
              (height, width, channels), processed according to `color_mode` and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
            ValueError: If an invalid `color_mode` is requested.
        """

        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()

        frame = self.async_read(timeout_ms=10000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _postprocess_image(self, image: NDArray[Any], depth_frame: bool = False) -> NDArray[Any]:
        """
        Applies color conversion, dimension validation, and rotation to a raw color frame.

        Args:
            image (np.ndarray): The raw image frame (expected RGB format from RealSense).

        Returns:
            np.ndarray: The processed image frame according to `self.color_mode` and `self.rotation`.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured
                          `width` and `height`.
        """

        if self.color_mode and self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, c = image.shape

            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        processed_image = image
        if self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self) -> None:
        """
        内部循环由后台线程运行，用于异步读取。
        在每次迭代中：
        1. 以500ms的超时读取一个颜色帧
        2. 以线程安全的方式将结果存储在latest_frame中并更新时间
        3. 设置new_frame_event以通知监听器
        在DeviceNotConnectedError上停止，在其他错误上记录并继续。
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frame = self._read_from_hardware()
                color_frame_raw = frame.get_color_frame()
                color_frame = np.asanyarray(color_frame_raw.get_data())
                processed_color_frame = self._postprocess_image(color_frame)

                if self.use_depth:
                    depth_frame_raw = frame.get_depth_frame()
                    depth_frame = np.asanyarray(depth_frame_raw.get_data())
                    processed_depth_frame = self._postprocess_image(depth_frame, depth_frame=True)

                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_color_frame = processed_color_frame
                    if self.use_depth:
                        self.latest_depth_frame = processed_depth_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
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

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    # NOTE(Steven): Missing implementation for depth for now
    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame data (color) asynchronously.

        This method retrieves the most recent color frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.
        It is “best effort” under high FPS.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray:
            The latest captured frame data (color image), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another error occurs.
        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    # NOTE(Steven): Missing implementation for depth for now
    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent (color) frame captured immediately (Peeking).

        This method is non-blocking and returns whatever is currently in the
        memory buffer. The frame may be stale,
        meaning it could have been captured a while ago (hanging camera scenario e.g.).

        Returns:
            NDArray[Any]: The frame image (numpy array).

        Raises:
            TimeoutError: If the latest frame is older than `max_age_ms`.
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera is connected but has not captured any frames yet.
        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_color_frame
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
        Disconnects from the camera, stops the pipeline, and cleans up resources.

        Stops the background read thread (if running) and stops the RealSense pipeline.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected (pipeline not running).
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.rs_pipeline is not None:
            self.rs_pipeline.stop()
            self.rs_pipeline = None
            self.rs_profile = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
