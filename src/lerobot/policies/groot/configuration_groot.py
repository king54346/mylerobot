#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("groot")
@dataclass
class GrootConfig(PreTrainedConfig):
    """Groot 策略包装器的配置。"""

    # 基本策略设置
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    # 维度设置（必须匹配预训练 GR00T 模型的期望）
    # 最大状态维度。较短的状态将被零填充。
    max_state_dim: int = 64

    # 最大动作维度。较短的动作将被零填充。
    max_action_dim: int = 32

    # 归一化（从恒等开始，根据需要调整）
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # 图像预处理（调整以匹配 Groot 的期望输入）
    image_size: tuple[int, int] = (224, 224)

    # Groot 特定模型参数（来自 groot_finetune_script.py）

    # 基础 Groot 模型的路径或 HuggingFace 模型 ID
    base_model_path: str = "nvidia/GR00T-N1.5-3B"

    # 托管 vocab.json 和 merges.txt 的 HF 仓库 ID（或本地路径）用于 Eagle 分词器。
    tokenizer_assets_repo: str = "lerobot/eagle2hg-processor-groot-n1p5"

    # 用于训练的具身标签（例如 'new_embodiment', 'gr1'）
    embodiment_tag: str = "new_embodiment"

    # 微调控制参数

    # 是否微调 llm 骨干网络
    tune_llm: bool = False

    # 是否微调视觉塔
    tune_visual: bool = False

    # 是否微调投影器
    tune_projector: bool = True

    # 是否微调扩散模型
    tune_diffusion_model: bool = True

    # LoRA 参数（来自 groot_finetune_script.py）
    # LORA 模型的秩。如果为 0，则不使用 LORA。
    lora_rank: int = 0

    # LORA 模型的 Alpha 值
    lora_alpha: int = 16

    # LORA 模型的 Dropout 率
    lora_dropout: float = 0.1

    # 是否对 LORA 使用完整模型
    lora_full_model: bool = False

    # 训练参数（匹配 groot_finetune_script.py）
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    use_bf16: bool = True

    # 数据集参数
    # 用于训练的视频后端（'decord' 或 'torchvision_av'）
    video_backend: str = "decord"

    # 是否在混合数据集中平衡数据集权重
    balance_dataset_weights: bool = True

    # 是否按轨迹长度加权采样轨迹
    balance_trajectory_weights: bool = True

    # 可选数据集路径，用于委托训练给 Isaac-GR00T 运行器
    dataset_paths: list[str] | None = None
    output_dir: str = "./tmp/gr00t"
    save_steps: int = 1000
    max_steps: int = 10000
    batch_size: int = 32
    dataloader_num_workers: int = 8
    report_to: str = "wandb"
    resume: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )

        # groot_repo_path 现在是可选的，因为我们已移植了组件
        # 不需要验证

    def validate_features(self) -> None:
        """为 Groot 验证并设置输入/输出特征。"""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Groot policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature
        else:
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature
        else:
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        """返回优化器配置。"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """返回调度器配置。"""
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=int(10000 * self.warmup_ratio),  # 默认 5% 预热
            num_decay_steps=10000,  # 根据训练步数调整
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * 0.1,
        )

    @property
    def observation_delta_indices(self) -> None:
        """返回增量观测的索引（Groot 为 None）。"""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """返回增量动作的索引。"""
        return list(range(min(self.chunk_size, 16)))

    @property
    def reward_delta_indices(self) -> None:
        """返回增量奖励的索引（Groot 为 None）。"""
        return None
