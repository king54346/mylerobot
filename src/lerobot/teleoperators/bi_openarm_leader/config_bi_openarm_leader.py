from dataclasses import dataclass

from lerobot.teleoperators.openarm_leader import OpenArmLeaderConfigBase

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_openarm_leader")
@dataclass
class BiOpenArmLeaderConfig(TeleoperatorConfig):
    """Configuration class for Bi OpenArm Follower robots."""

    left_arm_config: OpenArmLeaderConfigBase
    right_arm_config: OpenArmLeaderConfigBase
