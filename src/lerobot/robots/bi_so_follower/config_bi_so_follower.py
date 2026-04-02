from dataclasses import dataclass

from lerobot.robots.so_follower import SOFollowerConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_so_follower")
@dataclass
class BiSOFollowerConfig(RobotConfig):
    """Configuration class for Bi SO Follower robots."""

    left_arm_config: SOFollowerConfig
    right_arm_config: SOFollowerConfig
