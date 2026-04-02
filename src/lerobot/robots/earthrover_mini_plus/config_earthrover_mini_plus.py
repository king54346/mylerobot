"""Configuration for EarthRover Mini Plus robot."""

from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("earthrover_mini_plus")
@dataclass
class EarthRoverMiniPlusConfig(RobotConfig):
    """Configuration for EarthRover Mini Plus robot using Frodobots SDK.

    This robot uses cloud-based control via the Frodobots SDK HTTP API.
    Camera frames are accessed directly through SDK HTTP endpoints.

    Attributes:
        sdk_url: URL of the Frodobots SDK server (default: http://localhost:8000)
    """

    sdk_url: str = "http://localhost:8000"
