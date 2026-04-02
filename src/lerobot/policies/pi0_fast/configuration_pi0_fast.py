#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi0_fast")
@dataclass
class PI0FastConfig(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    chunk_size: int = 50  # 要预测的动作步数，在 openpi 中称为 "action_horizon"
    n_action_steps: int = 50  # 要执行的动作步数

    # 较短的状态和动作向量将被填充到这些维度
    max_state_dim: int = 32
    max_action_dim: int = 32
    max_action_tokens: int = 256

    # 实时分块 (RTC) 配置
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # 参见 openpi `preprocessing_pytorch.py`

    # 添加空图像。当没有图像特征时用于添加空相机。
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # 参见 openpi `__post_init__`
    text_tokenizer_name: str = "google/paligemma-3b-pt-224"
    action_tokenizer_name: str = "physical-intelligence/fast"
    temperature: float = 0.0
    max_decoding_steps: int = 256
    fast_skip_tokens: int = 128

    # 是否验证解码的动作 token 以 "Action: " 前缀开头
    validate_action_token_prefix: bool = True

    # 是否使用 KV 缓存加速自回归解码
    use_kv_cache: bool = True

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,  # Pi0Fast 使用均值标准差归一化状态
            "ACTION": NormalizationMode.MEAN_STD,  # Pi0Fast 使用均值标准差归一化动作
        }
    )

    # 训练设置
    gradient_checkpointing: bool = False  # 启用梯度检查点以优化内存
    compile_model: bool = False  # 是否使用 torch.compile 进行模型优化
    compile_mode: str = "max-autotune"  # Torch 编译模式
    device: str | None = None  # 用于模型的设备 (None = 自动检测)

    # 优化器设置: 参见 openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # 参见 openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # 调度器设置: 参见 openpi `CosineDecaySchedule`
    # 注意: 如果 --steps < scheduler_decay_steps，这些值会自动缩放
    # 例如，--steps=3000 会将预热缩放到 100，衰减缩放到 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # 验证配置
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
