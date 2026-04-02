#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Action Chunking Transformers (ACT) 策略配置类。

    ACT 策略配置模块
    ======================

    默认配置用于双臂 Aloha 任务训练，如“插入”或“转移”。

    您最可能需要修改的参数是与环境/传感器相关的参数：
    `input_features` 和 `output_features`。

    输入输出说明:
        - 事件之一：
            - 至少需要一个以 "observation.image" 开头的键作为输入
              和/或
            - 需要 "observation.environment_state" 键作为输入
        - 如果有多个以 "observation.images." 开头的键，它们被视为多个摄像头视角
          目前仅支持所有图像具有相同形状
        - 可选在没有 "observation.state" 键（本体感受机器人状态）的情况下工作
        - 需要 "action" 作为输出键

    Args:
        n_obs_steps: 传递给策略的观测环境步数（取当前步和之前的额外步数）
        chunk_size: 动作预测“块”的大小，以环境步数为单位
        n_action_steps: 每次策略调用在环境中执行的动作步数。
            不应超过 chunk_size。例如，如果 chunk_size 为 100，
            可以设为 50。这意味着模型预测 100 步动作，在环境中执行 50 步，
            丢弃另外 50 步
        input_features: 定义策略输入数据 PolicyFeature 的字典。键表示输入数据名称，
            值是包含 FeatureType 和 shape 属性的 PolicyFeature
        output_features: 定义策略输出数据 PolicyFeature 的字典
        normalization_mapping: 从 FeatureType 字符串值（如 "STATE", "VISUAL"）到
            对应 NormalizationMode（如 NormalizationMode.MIN_MAX）的映射字典
        vision_backbone: 用于编码图像的 torchvision resnet 主干网络名称
        pretrained_backbone_weights: 用于初始化主干网络的 torchvision 预训练权重。
            `None` 表示不使用预训练权重
        replace_final_stride_with_dilation: 是否将 ResNet 最后的 2x2 步幅替换为空洞卷积
        pre_norm: 是否在 transformer 块中使用“pre-norm”
        dim_model: transformer 块的主要隐藏维度
        n_heads: transformer 块多头注意力中使用的头数
        dim_feedforward: transformer 前馈层中隐藏维度扩展的维度
        feedforward_activation: transformer 块前馈层中使用的激活函数
        n_encoder_layers: transformer 编码器使用的层数
        n_decoder_layers: transformer 解码器使用的层数
        use_vae: 是否在训练期间使用变分目标。这会引入另一个作为 VAE 编码器的 transformer
            （不要与 transformer 编码器混淆 - 参见策略类中的文档）
        latent_dim: VAE 的潜在维度
        n_vae_encoder_layers: VAE 编码器使用的 transformer 层数
        temporal_ensemble_coeff: 用于时间集成的指数加权方案系数。
            默认为 None 表示不使用时间集成。使用此功能时 `n_action_steps` 必须为 1，
            因为每步都需要进行推理以形成集成。更多信息请参见 `ACTTemporalEnsembler`
        dropout: transformer 层中使用的 dropout（详见代码）
        kl_weight: 如果启用变分目标，损失中 KL 散度分量的权重。
            损失计算为: `reconstruction_loss + kl_weight * kld_loss`
    """

    # 输入/输出结构
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # 架构
    # 视觉主干网络
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer 层
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # 注意: 虽然原始 ACT 实现中 `n_decoder_layers` 为 7，但代码中存在 bug
    # 导致只有第一层被使用。这里通过设为 1 来匹配原始实现。
    # 参见 https://github.com/tonyzhaozh/act/issues/25#issue-2258740521
    n_decoder_layers: int = 1
    # VAE 变分自编码器
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # 推理
    # 注意: ACT 启用时间集成时使用的值为 0.01
    temporal_ensemble_coeff: float | None = None

    # 训练和损失计算
    dropout: float = 0.1
    kl_weight: float = 10.0

    # 训练预设
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """输入验证（非穷尽）。"""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
