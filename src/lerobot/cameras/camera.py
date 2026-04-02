"""
摄像头基类模块 (Camera Base Class)
====================================

本模块定义了所有摄像头实现的抽象基类。

主要功能：
- 定义统一的摄像头操作接口
- 支持同步/异步帧采集
- 资源生命周期管理（连接/断开）
- 上下文管理器支持

支持的摄像头后端：
- OpenCV: 通用摄像头支持 (USB摄像头 → cv2.VideoCapture → 你的程序)
- Intel RealSense: 深度摄像头
(RealSense摄像头 → pyrealsense2 SDK → 你的程序
         ↓
   同时输出：RGB图像 + 深度图(每个像素的距离))
- ZMQ: 网络摄像头流
另一台电脑/树莓派          你的主机
[摄像头] → [发布图像] ——网络——→ [订阅接收] → 你的程序
              ZMQ Publisher        ZMQ Subscriber
- Reachy2: Reachy2 机器人专用摄像头
Reachy2机器人内置摄像头 → 专用API reachy2_sdk → 你的程序
"""

import abc
import warnings
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from .configs import CameraConfig


class Camera(abc.ABC):
    """
    摄像头实现的抽象基类。

    为不同后端的摄像头操作定义标准接口。
    子类必须实现所有抽象方法。

    管理基本摄像头属性（帧率、分辨率）和核心操作：
    - 连接/断开连接
    - 帧采集（同步/异步/最新帧）

    属性:
        fps (int | None): 配置的帧率（每秒帧数）
        width (int | None): 帧宽度（像素）
        height (int | None): 帧高度（像素）
    
    主要接口方法:
        - connect(): 建立摄像头连接
        - disconnect(): 断开连接并释放资源
        - read(): 同步读取一帧
        - async_read(): 异步读取最新帧
        - read_latest(): 立即返回最新帧（可能过时）
    """

    def __init__(self, config: CameraConfig):
        """
        使用给定配置初始化摄像头。

        Args:
            config: 包含帧率和分辨率的摄像头配置。
        """
        self.fps: int | None = config.fps      # 帧率
        self.width: int | None = config.width  # 图像宽度
        self.height: int | None = config.height  # 图像高度

    def __enter__(self):
        """
        上下文管理器入口。
        自动连接到摄像头。
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
    def is_connected(self) -> bool:
        """
        检查摄像头是否已连接。

        Returns:
            bool: 如果摄像头已连接并准备好采集帧则返回True，否则返回False。
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        检测连接到系统的可用摄像头。
        
        Returns:
            List[Dict[str, Any]]: 字典列表，每个字典包含检测到的摄像头信息。
        """
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """
        建立与摄像头的连接。

        Args:
            warmup: 如果为True（默认），在返回前采集一帧预热。
                   对于需要时间调整采集设置的摄像头很有用。
                   如果为False，跳过预热帧。
        """
        pass

    @abc.abstractmethod
    def read(self) -> NDArray[Any]:
        """
        同步采集并返回一帧图像。

        这是一个阻塞调用，会等待硬件及其SDK。

        Returns:
            np.ndarray: 采集的帧作为numpy数组。
        """
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = ...) -> NDArray[Any]:
        """
        返回最新的新帧。

        此方法获取后台线程采集的最新帧。
        如果缓冲区中已有新帧（自上次调用后采集），立即返回。

        只有当缓冲区为空或最新帧已被上一次 async_read 调用消费时，
        才会阻塞等待最多 timeout_ms 毫秒。

        本质上，此方法返回最新的未消费帧，必要时等待新帧在指定超时内到达。

        使用场景:
            - 适合控制循环，确保每个处理的帧都是新鲜的，
              有效地将循环与摄像头的FPS同步。
            - 超时通常由以下原因引起：摄像头FPS很低、处理负载很重、
              或摄像头已断开连接。

        Args:
            timeout_ms: 等待新帧的最大时间（毫秒）。默认为200ms（0.2秒）。

        Returns:
            np.ndarray: 采集的帧作为numpy数组。

        Raises:
            TimeoutError: 如果在 timeout_ms 内没有新帧到达。
        """
        pass

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """
        立即返回最近采集的帧（窃视模式）。

        此方法是非阻塞的，返回当前内存缓冲区中的内容。
        返回的帧可能是过时的，意味着它可能是很久之前采集的
        （例如摄像头卡死场景）。

        使用场景:
            适合需要零延迟或解耦频率的场景，以及需要保证获取帧的情况，
            例如UI可视化、日志记录或非关键性监控。

        Args:
            max_age_ms: 帧的最大允许年龄（毫秒）。默认500ms。

        Returns:
            NDArray[Any]: 帧图像（numpy数组）。

        Raises:
            TimeoutError: 如果最新帧超过 max_age_ms。
            NotConnectedError: 如果摄像头未连接。
            RuntimeError: 如果摄像头已连接但尚未采集任何帧。
        """
        warnings.warn(
            f"{self.__class__.__name__}.read_latest() 未实现。"
            "请重写 read_latest()；它将在未来版本中成为必需。",
            FutureWarning,
            stacklevel=2,
        )
        return self.async_read()

    @abc.abstractmethod
    def disconnect(self) -> None:
        """断开与摄像头的连接并释放资源。"""
        pass
