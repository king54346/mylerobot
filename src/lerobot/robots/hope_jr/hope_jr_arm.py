"""
Hope Jr 机械臂模块 (Hope Jr Arm Robot Module)
=========================================

本模块实现了 Hope Jr 机械臂的控制。

电机配置:
    - shoulder_pitch: SM8512BL (ID: 1)
    - shoulder_yaw: STS3250 (ID: 2)
    - shoulder_roll: STS3250 (ID: 3)
    - elbow_flex: STS3250 (ID: 4)
    - wrist_roll: STS3250 (ID: 5)
    - wrist_yaw: STS3250 (ID: 6)
    - wrist_pitch: STS3250 (ID: 7)
"""

import logging
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.calibration_gui import RangeFinderGUI
from lerobot.motors.feetech import (
    FeetechMotorsBus,
)
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_hope_jr import HopeJrArmConfig

logger = logging.getLogger(__name__)


class HopeJrArm(Robot):
    config_class = HopeJrArmConfig
    name = "hope_jr_arm"

    def __init__(self, config: HopeJrArmConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pitch": Motor(1, "sm8512bl", MotorNormMode.RANGE_M100_100),
                "shoulder_yaw": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
                "shoulder_roll": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_yaw": Motor(6, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_pitch": Motor(7, "sts3250", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        # HACK: 特殊处理 shoulder_pitch 电机
        self.shoulder_pitch = "shoulder_pitch"
        self.other_motors = [m for m in self.bus.motors if m != "shoulder_pitch"]

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        建立与机器人的连接。

        假设连接时机械臂处于静止位置，
        可以安全地禁用力矩来进行标定。
        """

        self.bus.connect(handshake=False)
        if not self.is_calibrated and calibrate:
            self.calibrate()

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """标定机械臂电机。"""
        groups = {
            "all": list(self.bus.motors.keys()),
            "shoulder": ["shoulder_pitch", "shoulder_yaw", "shoulder_roll"],
            "elbow": ["elbow_flex"],
            "wrist": ["wrist_roll", "wrist_yaw", "wrist_pitch"],
        }

        self.calibration = RangeFinderGUI(self.bus, groups).run()
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        """配置电机参数。"""
        with self.bus.torque_disabled():
            self.bus.configure_motors(maximum_acceleration=30, acceleration=30)

    def setup_motors(self) -> None:
        """设置电机 ID，需要逐个连接电机。"""
        # TODO: 添加更详细的文档说明
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """获取机械臂当前观测值。"""
        # 读取机械臂位置
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position", self.other_motors)
        obs_dict[self.shoulder_pitch] = self.bus.read("Present_Position", self.shoulder_pitch)
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} 读取状态: {dt_ms:.1f}ms")

        # 从相机捕获图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} 读取 {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """向机械臂发送动作命令。"""
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # 当目标位置与当前位置相差过大时限制目标位置。
        # /!\ 由于需要从跟随者读取，fps 会变慢。
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        """断开与机械臂的连接。"""
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} 已断开连接。")
