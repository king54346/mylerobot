"""
Koch 跟随者机械臂模块 (Koch Follower Robot Module)
=============================================

本模块实现了 Koch 跟随者机械臂的控制。

支持的硬件:
    - Koch v1.0: https://github.com/AlexanderKoch-Koch/low_cost_robot
      由 Tau Robotics 的 Alexander Koch 开发
    - Koch v1.1: https://github.com/jess-moss/koch-v1-1
      由 Jess Moss 开发

电机配置:
    - shoulder_pan: XL430-W250 (ID: 1)
    - shoulder_lift: XL430-W250 (ID: 2)
    - elbow_flex: XL330-M288 (ID: 3)
    - wrist_flex: XL330-M288 (ID: 4)
    - wrist_roll: XL330-M288 (ID: 5)
    - gripper: XL330-M288 (ID: 6)
"""

import logging
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_koch_follower import KochFollowerConfig

logger = logging.getLogger(__name__)


class KochFollower(Robot):
    """
    Koch 跟随者机器人。

    支持的硬件:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot)，包含和不包含腕部到肘部的
        扩展，由 [Tau Robotics](https://tau-robotics.com) 的 Alexander Koch 开发
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) 由 Jess Moss 开发
    """

    config_class = KochFollowerConfig
    name = "koch_follower"

    def __init__(self, config: KochFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "xl430-w250", norm_mode_body),
                "shoulder_lift": Motor(2, "xl430-w250", norm_mode_body),
                "elbow_flex": Motor(3, "xl330-m288", norm_mode_body),
                "wrist_flex": Motor(4, "xl330-m288", norm_mode_body),
                "wrist_roll": Motor(5, "xl330-m288", norm_mode_body),
                "gripper": Motor(6, "xl330-m288", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

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

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """标定机械臂电机。"""
        self.bus.disable_torque()
        if self.calibration:
            # 标定文件存在，询问用户是使用该文件还是运行新的标定
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\nRunning calibration of {self}")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        full_turn_motors = ["shoulder_pan", "wrist_roll"]
        unknown_range_motors = [motor for motor in self.bus.motors if motor not in full_turn_motors]
        print(
            f"Move all joints except {full_turn_motors} sequentially through their entire "
            "ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for motor in full_turn_motors:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """配置电机参数。"""
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            # 除了夹爪外，所有电机使用'扩展位置模式'，因为在关节模式下伺服电机
            # 无法旋转超过 360 度（从 0 到 4095）。而且在组装机械臂时可能会出现
            # 错误，导致伺服电机在关键点的位置为 0 或 4095
            for motor in self.bus.motors:
                if motor != "gripper":
                    self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

            # 夹爪使用'基于电流的位置控制'，以便通过电流限制来限制力。对于
            # 跟随者夹爪，这意味着它可以在不施加过大力的情况下抓取物体，即使其
            # 目标位置是完全抓取（两个夹爪手指被命令合拢并接触）。
            # 对于导引端夹爪，这意味着我们可以将其用作物理触发器，因为我们可以
            # 用手指施力使其移动，当我们释放力时，它会回到原来的目标位置。
            self.bus.write("Operating_Mode", "gripper", OperatingMode.CURRENT_POSITION.value)

            # 设置更好的 PID 值以减小记录状态和动作之间的差距
            # TODO(rcadene): 实现自动程序为每个电机设置最优 PID 值
            self.bus.write("Position_P_Gain", "elbow_flex", 1500)
            self.bus.write("Position_I_Gain", "elbow_flex", 0)
            self.bus.write("Position_D_Gain", "elbow_flex", 600)

    def setup_motors(self) -> None:
        """设置电机 ID，需要逐个连接电机。"""
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """获取机械臂当前观测值。"""
        # 读取机械臂位置
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
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
        """
        命令机械臂移动到目标关节配置。
    
        相对动作幅度可能会根据配置参数 `max_relative_target` 被裁剪。
        在这种情况下，发送的动作与原始动作不同。
        因此，此函数始终返回实际发送的动作。
    
        参数:
            action (RobotAction): 电机的目标位置。
    
        返回:
            RobotAction: 发送到电机的动作，可能已被裁剪。
        """
    
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
    
        # 当目标位置与当前位置相差过大时限制目标位置。
        # /!\ 由于需要从跟随者读取，fps 会变慢。
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
    
        # 向机械臂发送目标位置
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        """断开与机械臂的连接。"""
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} 已断开连接。")
