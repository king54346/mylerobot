from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class ExoskeletonArmPortConfig:
    """Serial port configuration for individual exoskeleton arm."""

    port: str = ""
    baud_rate: int = 115200


@TeleoperatorConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1TeleoperatorConfig(TeleoperatorConfig):
    left_arm_config: ExoskeletonArmPortConfig = field(default_factory=ExoskeletonArmPortConfig)
    right_arm_config: ExoskeletonArmPortConfig = field(default_factory=ExoskeletonArmPortConfig)

    # Frozen joints (comma-separated joint names that won't be moved by IK)
    frozen_joints: str = ""
