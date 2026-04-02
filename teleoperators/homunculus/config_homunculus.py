from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("homunculus_glove")
@dataclass
class HomunculusGloveConfig(TeleoperatorConfig):
    port: str  # Port to connect to the glove
    side: str  # "left" / "right"
    baud_rate: int = 115_200

    def __post_init__(self):
        if self.side not in ["right", "left"]:
            raise ValueError(self.side)


@TeleoperatorConfig.register_subclass("homunculus_arm")
@dataclass
class HomunculusArmConfig(TeleoperatorConfig):
    port: str  # Port to connect to the arm
    baud_rate: int = 115_200
