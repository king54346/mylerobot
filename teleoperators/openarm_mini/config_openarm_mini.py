from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("openarm_mini")
@dataclass
class OpenArmMiniConfig(TeleoperatorConfig):
    """Configuration for OpenArm Mini teleoperator with Feetech motors (dual arms)."""

    port_right: str = "/dev/ttyUSB0"
    port_left: str = "/dev/ttyUSB1"

    use_degrees: bool = True
