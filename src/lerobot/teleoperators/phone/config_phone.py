from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config import TeleoperatorConfig


class PhoneOS(Enum):
    ANDROID = "android"
    IOS = "ios"


@TeleoperatorConfig.register_subclass("phone")
@dataclass
class PhoneConfig(TeleoperatorConfig):
    phone_os: PhoneOS = PhoneOS.IOS
    camera_offset = np.array(
        [0.0, -0.02, 0.04]
    )  # iPhone 14 Pro camera is 2cm off center and 4cm above center
