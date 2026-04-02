import logging
from functools import cached_property

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_openarm_follower import BiOpenArmFollowerConfig

logger = logging.getLogger(__name__)


class BiOpenArmFollower(Robot):
    """
    双臂 OpenArm 跟随者机器人。
    """

    config_class = BiOpenArmFollowerConfig
    name = "bi_openarm_follower"

    def __init__(self, config: BiOpenArmFollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = OpenArmFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.left_arm_config.max_relative_target,
            cameras=config.left_arm_config.cameras,
            side=config.left_arm_config.side,
            can_interface=config.left_arm_config.can_interface,
            use_can_fd=config.left_arm_config.use_can_fd,
            can_bitrate=config.left_arm_config.can_bitrate,
            can_data_bitrate=config.left_arm_config.can_data_bitrate,
            motor_config=config.left_arm_config.motor_config,
            position_kd=config.left_arm_config.position_kd,
            position_kp=config.left_arm_config.position_kp,
            joint_limits=config.left_arm_config.joint_limits,
        )

        right_arm_config = OpenArmFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.right_arm_config.max_relative_target,
            cameras=config.right_arm_config.cameras,
            side=config.right_arm_config.side,
            can_interface=config.right_arm_config.can_interface,
            use_can_fd=config.right_arm_config.use_can_fd,
            can_bitrate=config.right_arm_config.can_bitrate,
            can_data_bitrate=config.right_arm_config.can_data_bitrate,
            motor_config=config.right_arm_config.motor_config,
            position_kd=config.right_arm_config.position_kd,
            position_kp=config.right_arm_config.position_kp,
            joint_limits=config.right_arm_config.joint_limits,
        )

        self.left_arm = OpenArmFollower(left_arm_config)
        self.right_arm = OpenArmFollower(right_arm_config)

        # 仅用于与代码库中其他期望 `robot.cameras` 属性的部分兼容
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_arm_motors_ft = self.left_arm._motors_ft
        right_arm_motors_ft = self.right_arm._motors_ft

        return {
            **{f"left_{k}": v for k, v in left_arm_motors_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        left_arm_cameras_ft = self.left_arm._cameras_ft
        right_arm_cameras_ft = self.right_arm._cameras_ft

        return {
            **{f"left_{k}": v for k, v in left_arm_cameras_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_cameras_ft.items()},
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        """设置电机 ID。"""
        raise NotImplementedError(
            "CAN 电机的 ID 配置通常通过制造商工具完成。"
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """获取双臂当前观测值。"""
        obs_dict = {}

        # 添加 "left_" 前缀
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # 添加 "right_" 前缀
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        return obs_dict

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        """向双臂发送动作命令。"""
        # 移除 "left_" 前缀
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # 移除 "right_" 前缀
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_action_left = self.left_arm.send_action(left_action, custom_kp, custom_kd)
        sent_action_right = self.right_arm.send_action(right_action, custom_kp, custom_kd)

        # 重新添加前缀
        prefixed_sent_action_left = {f"left_{key}": value for key, value in sent_action_left.items()}
        prefixed_sent_action_right = {f"right_{key}": value for key, value in sent_action_right.items()}

        return {**prefixed_sent_action_left, **prefixed_sent_action_right}

    @check_if_not_connected
    def disconnect(self):
        """断开与双臂的连接。"""
        self.left_arm.disconnect()
        self.right_arm.disconnect()
