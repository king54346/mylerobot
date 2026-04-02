from dataclasses import dataclass

from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_openarm_follower")
@dataclass
class BiOpenArmFollowerConfig(RobotConfig):
    """Configuration class for Bi OpenArm Follower robots."""

    left_arm_config: OpenArmFollowerConfigBase
    right_arm_config: OpenArmFollowerConfigBase
