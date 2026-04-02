"""
观测处理器模块 (Observation Processor Module)
============================================

将各种Gym环境的观测数据转换为LeRobot标准格式。

主要组件：
- VanillaObservationProcessorStep: 处理标准Gymnasium观测

转换包括：
- 图像: (H,W,C) uint8 -> (B,C,H,W) float32 [0,1]
- 状态: numpy -> torch tensor
- 键重命名: pixels -> observation.image 等
"""

from dataclasses import dataclass

import einops
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="observation_processor")
class VanillaObservationProcessorStep(ObservationProcessorStep):
    """
    将标准Gymnasium观测处理为LeRobot格式。
    
    此步骤处理来自典型观测字典的图像和状态数据，
    准备它们以供 LeRobot 策略使用。
    
    **图像处理:**
    - 将通道最后(H, W, C)、`uint8`图像转换为
      通道最前(C, H, W)、`float32`张量
    - 将像素值从[0, 255]归一化到[0, 1]
    - 如果没有批次维度则添加
    - 识别键"pixels"下的单张图像并映射到"observation.image"
    - 识别键"pixels"下的图像字典并映射到"observation.images.{camera_name}"
    
    **状态处理:**
    - 将"environment_state"键映射到"observation.environment_state"
    - 将"agent_pos"键映射到"observation.state"
    - 将NumPy数组转换为PyTorch张量
    - 如果没有批次维度则添加
    """

    def _process_single_image(self, img: np.ndarray) -> Tensor:
        """
        将单个NumPy图像数组处理为通道最前的归一化张量。
        
        Args:
            img: 表示图像的NumPy数组，期望为通道最后(H, W, C)格式，
                 `uint8` dtype
        
        Returns:
            通道最前(B, C, H, W)格式的`float32` PyTorch张量，
            像素值归一化到[0, 1]范围
        
        Raises:
            ValueError: 如果输入图像不是通道最后格式或不是`uint8` dtype
        """
        # 转换为张量
        img_tensor = torch.from_numpy(img)

        # 如果需要添加批次维度
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 验证图像格式
        _, h, w, c = img_tensor.shape
        if not (c < h and c < w):
            raise ValueError(f"Expected channel-last images, but got shape {img_tensor.shape}")

        if img_tensor.dtype != torch.uint8:
            raise ValueError(f"Expected torch.uint8 images, but got {img_tensor.dtype}")

        # 转换为通道最前格式
        img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()

        # 转换为float32并归一化到[0, 1]
        img_tensor = img_tensor.type(torch.float32) / 255.0

        return img_tensor

    def _process_observation(self, observation):
        """
        处理图像和状态观测数据。
        """

        processed_obs = observation.copy()

        if "pixels" in processed_obs:
            pixels = processed_obs.pop("pixels")

            if isinstance(pixels, dict):
                imgs = {f"{OBS_IMAGES}.{key}": img for key, img in pixels.items()}
            else:
                imgs = {OBS_IMAGE: pixels}

            for imgkey, img in imgs.items():
                processed_obs[imgkey] = self._process_single_image(img)

        if "environment_state" in processed_obs:
            env_state_np = processed_obs.pop("environment_state")
            env_state = torch.from_numpy(env_state_np).float()
            if env_state.dim() == 1:
                env_state = env_state.unsqueeze(0)
            processed_obs[OBS_ENV_STATE] = env_state

        if "agent_pos" in processed_obs:
            agent_pos_np = processed_obs.pop("agent_pos")
            agent_pos = torch.from_numpy(agent_pos_np).float()
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            processed_obs[OBS_STATE] = agent_pos

        return processed_obs

    def observation(self, observation):
        return self._process_observation(observation)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        将特征键从Gym标准转换为LeRobot标准。
        
        此方法通过根据LeRobot约定重命名键来标准化特征字典，
        确保可以正确构建策略。它处理各种原始键格式，
        包括带有"observation."前缀的键。
        
        **重命名规则:**
        - `pixels` 或 `observation.pixels` -> `observation.image`
        - `pixels.{cam}` 或 `observation.pixels.{cam}` -> `observation.images.{cam}`
        - `environment_state` 或 `observation.environment_state` -> `observation.environment_state`
        - `agent_pos` 或 `observation.agent_pos` -> `observation.state`
        
        Args:
            features: 具有Gym风格键的策略特征字典
        
        Returns:
            具有标准化LeRobot键的策略特征字典
        """
        # 构建按相同 FeatureType 桶键控的新特征映射
        # 我们假设调用者已经将特征放在正确的 FeatureType 中
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {ft: {} for ft in features}

        exact_pairs = {
            "pixels": OBS_IMAGE,
            "environment_state": OBS_ENV_STATE,
            "agent_pos": OBS_STATE,
        }

        prefix_pairs = {
            "pixels.": f"{OBS_IMAGES}.",
        }

        # 遍历所有传入的特征桶并标准化/移动每个条目
        for src_ft, bucket in features.items():
            for key, feat in list(bucket.items()):
                handled = False

                # 前缀规则（例如 pixels.cam1 -> OBS_IMAGES.cam1）
                for old_prefix, new_prefix in prefix_pairs.items():
                    prefixed_old = f"{OBS_STR}.{old_prefix}"
                    if key.startswith(prefixed_old):
                        suffix = key[len(prefixed_old) :]
                        new_key = f"{new_prefix}{suffix}"
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                    if key.startswith(old_prefix):
                        suffix = key[len(old_prefix) :]
                        new_key = f"{new_prefix}{suffix}"
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                if handled:
                    continue

                # 精确名称规则（pixels, environment_state, agent_pos）
                for old, new in exact_pairs.items():
                    if key == old or key == f"{OBS_STR}.{old}":
                        new_key = new
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                if handled:
                    continue

                # 默认：在相同的源 FeatureType 桶中保留键
                new_features[src_ft][key] = feat

        return new_features
