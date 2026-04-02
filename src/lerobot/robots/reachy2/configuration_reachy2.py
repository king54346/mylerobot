"""
Reachy2 机器人配置模块 (Reachy2 Configuration Module)
================================================

本模块定义了 Reachy2 机器人的配置类。

Reachy2 是 Pollen Robotics 开发的人形机器人。

配置选项:
    - 机器人部件 (左臂/右臂/颈部/天线/移动底盘)
    - 相机配置 (遥操相机/躯干相机)
    - 安全限制 (max_relative_target)
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("reachy2")
@dataclass
class Reachy2RobotConfig(RobotConfig):
    """
    Reachy2 机器人配置类。
    
    属性:
        max_relative_target: 最大相对目标位置，用于安全限制
        ip_address: Reachy2 机器人的 IP 地址
        port: Reachy2 机器人的端口
        disable_torque_on_disconnect: 断开连接时是否平滑关闭
        use_external_commands: 是否使用外部命令系统
        with_*: 启用/禁用各机器人部件
    """
    # 最大相对目标位置，用于安全限制
    max_relative_target: float | None = None

    # IP address of the Reachy 2 robot
    ip_address: str | None = "localhost"
    # Port of the Reachy 2 robot
    port: int = 50065

    # If True, turn_off_smoothly() will be sent to the robot before disconnecting.
    disable_torque_on_disconnect: bool = False

    # Tag for external commands control
    # Set to True if you use an external commands system to control the robot,
    # such as the official teleoperation application: https://github.com/pollen-robotics/Reachy2Teleoperation
    # If True, robot.send_action() will not send commands to the robot.
    use_external_commands: bool = False

    # Robot parts
    # Set to False to not add the corresponding joints part to the robot list of joints.
    # By default, all parts are set to True.
    with_mobile_base: bool = True
    with_l_arm: bool = True
    with_r_arm: bool = True
    with_neck: bool = True
    with_antennas: bool = True

    # Robot cameras
    # Set to True if you want to use the corresponding cameras in the observations.
    # By default, no camera is used.
    with_left_teleop_camera: bool = False
    with_right_teleop_camera: bool = False
    with_torso_camera: bool = False

    # Camera parameters
    camera_width: int = 640
    camera_height: int = 480

    # For cameras other than the 3 default Reachy 2 cameras.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Add cameras with same ip_address as the robot
        if self.with_left_teleop_camera:
            self.cameras["teleop_left"] = Reachy2CameraConfig(
                name="teleop",
                image_type="left",
                ip_address=self.ip_address,
                port=self.port,
                width=self.camera_width,
                height=self.camera_height,
                fps=30,  # Not configurable for Reachy 2 cameras
                color_mode=ColorMode.RGB,
            )
        if self.with_right_teleop_camera:
            self.cameras["teleop_right"] = Reachy2CameraConfig(
                name="teleop",
                image_type="right",
                ip_address=self.ip_address,
                port=self.port,
                width=self.camera_width,
                height=self.camera_height,
                fps=30,  # Not configurable for Reachy 2 cameras
                color_mode=ColorMode.RGB,
            )
        if self.with_torso_camera:
            self.cameras["torso_rgb"] = Reachy2CameraConfig(
                name="depth",
                image_type="rgb",
                ip_address=self.ip_address,
                port=self.port,
                width=self.camera_width,
                height=self.camera_height,
                fps=30,  # Not configurable for Reachy 2 cameras
                color_mode=ColorMode.RGB,
            )

        super().__post_init__()

        if not (
            self.with_mobile_base
            or self.with_l_arm
            or self.with_r_arm
            or self.with_neck
            or self.with_antennas
        ):
            raise ValueError(
                "No Reachy2Robot part used.\n"
                "At least one part of the robot must be set to True "
                "(with_mobile_base, with_l_arm, with_r_arm, with_neck, with_antennas)"
            )
