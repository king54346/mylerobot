import logging
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import (
    DriveMode,
    DynamixelMotorsBus,
    OperatingMode,
)
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_omx_follower import OmxFollowerConfig

logger = logging.getLogger(__name__)


class OmxFollower(Robot):
    """
    OMX 跟随者机器人。

    支持 [OMX](https://github.com/ROBOTIS-GIT/open_manipulator)，
    由 [ROBOTIS](https://ai.robotis.com/) 的 Woojin Wie 和 Junha Cha 开发。
    """

    config_class = OmxFollowerConfig
    name = "omx_follower"

    def __init__(self, config: OmxFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(11, "xl430-w250", norm_mode_body),
                "shoulder_lift": Motor(12, "xl430-w250", norm_mode_body),
                "elbow_flex": Motor(13, "xl430-w250", norm_mode_body),
                "wrist_flex": Motor(14, "xl330-m288", norm_mode_body),
                "wrist_roll": Motor(15, "xl330-m288", norm_mode_body),
                "gripper": Motor(16, "xl330-m288", MotorNormMode.RANGE_0_100),
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
        连接到机器人。

        对于预标定的 OMX 机器人:
        - 如果包中的默认标定与电机不匹配，则从电机读取并保存
        - 这允许使用预标定的机器人而无需手动标定
        - 如果不存在标定文件，则使用出厂默认值（homing_offset=0, range_min=0, range_max=4095）
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
        """使用出厂默认值标定机器人。"""
        self.bus.disable_torque()
        logger.info(f"\n为 {self} 使用出厂默认标定值")
        logger.info(f"\n将 {self} 的默认配置写入电机")
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.EXTENDED_POSITION.value)

        for motor in self.bus.motors:
            self.bus.write("Drive_Mode", motor, DriveMode.NON_INVERTED.value)

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=0,
                range_min=0,
                range_max=4095,
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
