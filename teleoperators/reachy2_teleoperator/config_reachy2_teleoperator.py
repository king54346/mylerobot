from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("reachy2_teleoperator")
@dataclass
class Reachy2TeleoperatorConfig(TeleoperatorConfig):
    # IP address of the Reachy 2 robot used as teleoperator
    ip_address: str | None = "localhost"

    # Whether to use the present position of the joints as actions
    # if False, the goal position of the joints will be used
    use_present_position: bool = False

    # Which parts of the robot to use
    with_mobile_base: bool = True
    with_l_arm: bool = True
    with_r_arm: bool = True
    with_neck: bool = True
    with_antennas: bool = True

    def __post_init__(self):
        if not (
            self.with_mobile_base
            or self.with_l_arm
            or self.with_r_arm
            or self.with_neck
            or self.with_antennas
        ):
            raise ValueError(
                "No Reachy2Teleoperator part used.\n"
                "At least one part of the robot must be set to True "
                "(with_mobile_base, with_l_arm, with_r_arm, with_neck, with_antennas)"
            )
