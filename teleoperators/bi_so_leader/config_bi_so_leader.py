from dataclasses import dataclass

from lerobot.teleoperators.so_leader import SOLeaderConfig

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_so_leader")
@dataclass
class BiSOLeaderConfig(TeleoperatorConfig):
    """Configuration class for Bi SO Leader teleoperators."""

    left_arm_config: SOLeaderConfig
    right_arm_config: SOLeaderConfig
