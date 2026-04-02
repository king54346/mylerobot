from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("omx_leader")
@dataclass
class OmxLeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm
    port: str

    # Sets the arm in torque mode with the gripper motor set to this value. This makes it possible to squeeze
    # the gripper and have it spring back to an open position on its own.
    gripper_open_pos: float = 60.0
