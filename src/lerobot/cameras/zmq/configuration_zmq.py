from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode

__all__ = ["ZMQCameraConfig", "ColorMode"]


@CameraConfig.register_subclass("zmq")
@dataclass
class ZMQCameraConfig(CameraConfig):
    server_address: str
    port: int = 5555
    camera_name: str = "zmq_camera"
    color_mode: ColorMode = ColorMode.RGB
    timeout_ms: int = 5000
    warmup_s: int = 1

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)

        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be positive, but {self.timeout_ms} is provided.")

        if not self.server_address:
            raise ValueError("`server_address` cannot be empty.")

        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"`port` must be between 1 and 65535, but {self.port} is provided.")
