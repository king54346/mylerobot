"""EarthRover Mini Plus 机器人，使用 Frodobots SDK。"""

import base64
import logging
from functools import cached_property

import cv2
import numpy as np
import requests

from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from .config_earthrover_mini_plus import EarthRoverMiniPlusConfig

logger = logging.getLogger(__name__)

# 动作特征键
ACTION_LINEAR_VEL = "linear.vel"
ACTION_ANGULAR_VEL = "angular.vel"

# 观测特征键
OBS_FRONT = "front"
OBS_REAR = "rear"
OBS_LINEAR_VEL = "linear.vel"
OBS_BATTERY_LEVEL = "battery.level"
OBS_ORIENTATION_DEG = "orientation.deg"
OBS_GPS_LATITUDE = "gps.latitude"
OBS_GPS_LONGITUDE = "gps.longitude"
OBS_GPS_SIGNAL = "gps.signal"
OBS_SIGNAL_LEVEL = "signal.level"
OBS_VIBRATION = "vibration"
OBS_LAMP_STATE = "lamp.state"


class EarthRoverMiniPlus(Robot):
    """
    通过 Frodobots SDK HTTP API 控制的 EarthRover Mini Plus 机器人。

    该机器人使用云端控制而非直接硬件连接，通过 Frodobots SDK 进行控制。
    相机通过 Agora 云以 WebRTC 方式流式传输，
    控制命令通过 HTTP POST 请求发送。

    机器人支持:
    - 双相机（前置和后置）通过 SDK HTTP 端点访问
    - 线速度和角速度控制
    - 电池和方向遥测

    属性:
        config: 机器人配置
        sdk_base_url: Frodobots SDK 服务器的 URL（默认: http://localhost:8000）
    """

    config_class = EarthRoverMiniPlusConfig
    name = "earthrover_mini_plus"

    def __init__(self, config: EarthRoverMiniPlusConfig):
        """
        初始化 EarthRover Mini Plus 机器人。

        参数:
            config: 机器人配置，包含 SDK URL
        """
        super().__init__(config)
        self.config = config
        self.sdk_base_url = "http://localhost:8000"

        # 空的相机字典，用于与录制脚本兼容
        # 相机通过 SDK 直接访问，而不是通过 Camera 对象
        self.cameras = {}
        self._is_connected = False

        # 相机帧缓存（请求失败时的回退）
        self._last_front_frame = None
        self._last_rear_frame = None

        # 机器人遥测数据缓存（请求失败时的回退）
        self._last_robot_data = None

        logger.info(f"已初始化 {self.name}，SDK 地址: {self.sdk_base_url}")

    @property
    def is_connected(self) -> bool:
        """检查机器人是否已连接到 SDK。"""
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        通过 Frodobots SDK 连接到机器人。

        参数:
            calibrate: SDK 机器人不使用此参数（保留用于 API 兼容性）

        异常:
            DeviceAlreadyConnectedError: 如果机器人已连接
            DeviceNotConnectedError: 如果无法连接到 SDK 服务器
        """

        # 验证 SDK 是否正在运行并可访问
        try:
            response = requests.get(f"{self.sdk_base_url}/data", timeout=10.0)
            if response.status_code != 200:
                raise DeviceNotConnectedError(
                    f"Cannot connect to SDK at {self.sdk_base_url}. "
                    "Make sure it's running: hypercorn main:app --reload"
                )
        except requests.RequestException as e:
            raise DeviceNotConnectedError(f"Cannot connect to SDK at {self.sdk_base_url}: {e}") from e

        self._is_connected = True
        logger.info(f"{self.name} 已连接到 SDK")

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        """基于 SDK 的机器人不需要标定。"""
        logger.info("基于 SDK 的机器人不需要标定")

    @property
    def is_calibrated(self) -> bool:
        """
        SDK 机器人不需要标定。

        返回:
            bool: 对于基于 SDK 的机器人始终返回 True
        """
        return True

    def configure(self) -> None:
        """配置机器人（基于 SDK 的机器人无操作）。"""
        pass

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """
        定义用于数据集记录的观测空间。

        返回:
            dict: 观测特征及其类型/形状:
                - front: (480, 640, 3) - 前置相机 RGB 图像
                - rear: (480, 640, 3) - 后置相机 RGB 图像
                - linear.vel: float - 当前速度 (0-1, SDK 仅报告正速度)
                - battery.level: float - 电池电量 (0-1, 从 0-100 归一化)
                - orientation.deg: float - 机器人方向 (0-1, 从原始值归一化)
                - gps.latitude: float - GPS 纬度坐标
                - gps.longitude: float - GPS 经度坐标
                - gps.signal: float - GPS 信号强度 (0-1, 从百分比归一化)
                - signal.level: float - 网络信号等级 (0-1, 从 0-5 归一化)
                - vibration: float - 振动传感器读数
                - lamp.state: float - 灯状态 (0=关, 1=开)
        """
        return {
            # 相机 (高度, 宽度, 通道)
            OBS_FRONT: (480, 640, 3),
            OBS_REAR: (480, 640, 3),
            # 运动状态
            OBS_LINEAR_VEL: float,
            # 机器人状态
            OBS_BATTERY_LEVEL: float,
            OBS_ORIENTATION_DEG: float,
            # GPS
            OBS_GPS_LATITUDE: float,
            OBS_GPS_LONGITUDE: float,
            OBS_GPS_SIGNAL: float,
            # 传感器
            OBS_SIGNAL_LEVEL: float,
            OBS_VIBRATION: float,
            OBS_LAMP_STATE: float,
        }

    @cached_property
    def action_features(self) -> dict[str, type]:
        """
        定义动作空间。

        返回:
            dict: 动作特征及其类型:
                - linear.vel: float - 目标线速度
                - angular.vel: float - 目标角速度
        """
        return {
            ACTION_LINEAR_VEL: float,
            ACTION_ANGULAR_VEL: float,
        }

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        从 SDK 获取机器人当前观测值。

        返回:
            RobotObservation: 观测值包含:
                - front: 前置相机图像 (480, 640, 3) RGB 格式
                - rear: 后置相机图像 (480, 640, 3) RGB 格式
                - linear.vel: 当前速度 (0-1, SDK 仅报告正速度)
                - battery.level: 电池电量 (0-1, 从 0-100 归一化)
                - orientation.deg: 机器人方向 (0-1, 从原始值归一化)
                - gps.latitude: GPS 纬度坐标
                - gps.longitude: GPS 经度坐标
                - gps.signal: GPS 信号强度 (0-1, 从百分比归一化)
                - signal.level: 网络信号等级 (0-1, 从 0-5 归一化)
                - vibration: 振动传感器读数
                - lamp.state: 灯状态 (0=关, 1=开)

        异常:
            DeviceNotConnectedError: 如果机器人未连接

        注意:
            相机帧从 SDK 端点 /v2/front 和 /v2/rear 获取。
            帧从 base64 解码并从 BGR 转换为 RGB 格式。
            机器人遥测数据从 /data 端点获取。
            所有 SDK 值都归一化到适当的范围以便数据集记录。
        """

        observation = {}

        # 从 SDK 获取相机图像
        frames = self._get_camera_frames()
        observation[OBS_FRONT] = frames["front"]
        observation[OBS_REAR] = frames["rear"]

        # 从 SDK 获取机器人状态
        robot_data = self._get_robot_data()

        # 运动状态
        observation[OBS_LINEAR_VEL] = robot_data["speed"] / 100.0  # 将 0-100 归一化到 0-1

        # 机器人状态
        observation[OBS_BATTERY_LEVEL] = robot_data["battery"] / 100.0  # 将 0-100 归一化到 0-1
        observation[OBS_ORIENTATION_DEG] = robot_data["orientation"] / 360.0  # 归一化到 0-1

        # GPS 数据
        observation[OBS_GPS_LATITUDE] = robot_data["latitude"]
        observation[OBS_GPS_LONGITUDE] = robot_data["longitude"]
        observation[OBS_GPS_SIGNAL] = robot_data["gps_signal"] / 100.0  # 将百分比归一化到 0-1

        # 传感器
        observation[OBS_SIGNAL_LEVEL] = robot_data["signal_level"] / 5.0  # 将 0-5 归一化到 0-1
        observation[OBS_VIBRATION] = robot_data["vibration"]
        observation[OBS_LAMP_STATE] = float(robot_data["lamp"])  # 0 或 1

        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        通过 SDK 向机器人发送动作。

        参数:
            action: 动作字典，包含以下键:
                - linear.vel: 目标线速度 (-1 到 1)
                - angular.vel: 目标角速度 (-1 到 1)

        返回:
            RobotAction: 已发送的动作（与 action_features 键匹配）

        异常:
            DeviceNotConnectedError: 如果机器人未连接

        注意:
            动作通过 POST /control 端点发送到 SDK。
            SDK 期望命令在 [-1, 1] 范围内。
        """

        # 提取动作值并转换为浮点数
        linear = float(action.get(ACTION_LINEAR_VEL, 0.0))
        angular = float(action.get(ACTION_ANGULAR_VEL, 0.0))

        # 向 SDK 发送命令
        try:
            self._send_command_to_sdk(linear, angular)
        except Exception as e:
            logger.error(f"发送动作时出错: {e}")

        # 返回与 action_features 格式匹配的动作
        return {
            ACTION_LINEAR_VEL: linear,
            ACTION_ANGULAR_VEL: angular,
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        """
        断开与机器人的连接。

        停止机器人并关闭与 SDK 的连接。

        异常:
            DeviceNotConnectedError: 如果机器人未连接
        """

        # 断开连接前停止机器人
        try:
            self._send_command_to_sdk(0.0, 0.0)
        except Exception as e:
            logger.warning(f"断开连接时停止机器人失败: {e}")

        self._is_connected = False
        logger.info(f"{self.name} 已断开连接")

    # SDK 通信的私有辅助方法

    def _get_camera_frames(self) -> dict[str, np.ndarray]:
        """
        使用带缓存回退的 v2 端点从 SDK 获取相机帧。

        返回:
            dict: 包含 'front' 和 'rear' 键的字典:
                - 当前帧（如果请求成功）
                - 缓存帧（如果请求失败但缓存存在）
                - 零数组（如果请求失败且没有缓存）

        注意:
            使用 /v2/front 和 /v2/rear 端点，比 /screenshot 快 15 倍。
            图像为 base64 编码，调整到 640x480，并从 BGR 转换为 RGB。
            如果请求失败，返回最后一次成功获取的帧（缓存）。
        """
        frames = {}

        # 获取前置相机
        try:
            response = requests.get(f"{self.sdk_base_url}/v2/front", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if "front_frame" in data and data["front_frame"]:
                    front_img = self._decode_base64_image(data["front_frame"])
                    if front_img is not None:
                        # 调整大小并从 BGR 转换为 RGB
                        front_img = cv2.resize(front_img, (640, 480))
                        front_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
                        frames["front"] = front_rgb
                        # 缓存成功的帧
                        self._last_front_frame = front_rgb
        except Exception as e:
            logger.warning(f"获取前置相机时出错: {e}")

        # 回退: 使用缓存或零数组
        if "front" not in frames:
            if self._last_front_frame is not None:
                frames["front"] = self._last_front_frame
            else:
                frames["front"] = np.zeros((480, 640, 3), dtype=np.uint8)

        # 获取后置相机
        try:
            response = requests.get(f"{self.sdk_base_url}/v2/rear", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                if "rear_frame" in data and data["rear_frame"]:
                    rear_img = self._decode_base64_image(data["rear_frame"])
                    if rear_img is not None:
                        # 调整大小并从 BGR 转换为 RGB
                        rear_img = cv2.resize(rear_img, (640, 480))
                        rear_rgb = cv2.cvtColor(rear_img, cv2.COLOR_BGR2RGB)
                        frames["rear"] = rear_rgb
                        # 缓存成功的帧
                        self._last_rear_frame = rear_rgb
        except Exception as e:
            logger.warning(f"获取后置相机时出错: {e}")

        # 回退: 使用缓存或零数组
        if "rear" not in frames:
            if self._last_rear_frame is not None:
                frames["rear"] = self._last_rear_frame
            else:
                frames["rear"] = np.zeros((480, 640, 3), dtype=np.uint8)

        return frames

    def _decode_base64_image(self, base64_string: str) -> np.ndarray | None:
        """
        将 base64 字符串解码为图像。

        参数:
            base64_string: Base64 编码的图像字符串

        返回:
            np.ndarray: BGR 格式的解码图像（OpenCV 默认），或解码失败时返回 None
        """
        try:
            img_bytes = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img  # 返回 BGR 格式（OpenCV 默认）
        except Exception as e:
            logger.error(f"解码图像时出错: {e}")
            return None

    def _get_robot_data(self) -> dict:
        """
        从 SDK 获取机器人遥测数据。

        返回:
            dict: 机器人遥测数据，包括电池、速度、方向、GPS 等:
                - 当前数据（如果请求成功）
                - 缓存数据（如果请求失败但缓存存在）
                - 默认值（如果请求失败且没有缓存）

        注意:
            使用 /data 端点提供全面的机器人状态。
            如果请求失败，返回最后一次成功获取的数据（缓存）。
        """
        try:
            response = requests.get(f"{self.sdk_base_url}/data", timeout=2.0)
            if response.status_code == 200:
                data = response.json()
                # 缓存成功的数据
                self._last_robot_data = data
                return data
        except Exception as e:
            logger.warning(f"获取机器人数据时出错: {e}")

        # 回退: 使用缓存或默认值
        if self._last_robot_data is not None:
            return self._last_robot_data
        else:
            # 返回带默认值的字典（仅在任何缓存存在之前的第一次失败时使用）
            return {
                "speed": 0,
                "battery": 0,
                "orientation": 0,
                "latitude": 0.0,
                "longitude": 0.0,
                "gps_signal": 0,
                "signal_level": 0,
                "vibration": 0.0,
                "lamp": 0,
            }

    def _send_command_to_sdk(self, linear: float, angular: float, lamp: int = 0) -> bool:
        """
        向 SDK 发送控制命令。

        参数:
            linear: 线速度命令 (-1 到 1)
            angular: 角速度命令 (-1 到 1)
            lamp: 灯控制 (0=关, 1=开)

        返回:
            bool: 命令发送成功返回 True，否则返回 False

        注意:
            使用 POST /control 端点。命令以 JSON 负载发送。
        """
        try:
            payload = {
                "command": {
                    "linear": linear,
                    "angular": angular,
                    "lamp": lamp,
                }
            }

            response = requests.post(
                f"{self.sdk_base_url}/control",
                json=payload,
                timeout=1.0,
            )

            return response.status_code == 200
        except Exception as e:
            logger.error(f"发送命令时出错: {e}")
            return False
