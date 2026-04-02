#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """DiffusionPolicy 配置类。

    Diffusion 策略配置模块
    ======================

    默认配置用于使用 PushT 训练，提供本体感受和单摄像头观测。

    您最可能需要修改的参数是与环境/传感器相关的参数：
    `input_features` 和 `output_features`。

    输入输出说明:
        - 需要 "observation.state" 作为输入键
        - 事件之一:
            - 至少需要一个以 "observation.image" 开头的键作为输入
              和/或
            - 需要 "observation.environment_state" 键作为输入
        - 如果有多个以 "observation.image" 开头的键，它们被视为多个摄像头视角
          目前仅支持所有图像具有相同形状
        - 需要 "action" 作为输出键

    Args:
        n_obs_steps: 传递给策略的观测环境步数（取当前步和之前的额外步数）
        horizon: 扩散模型动作预测大小，详见 `DiffusionPolicy.select_action`
        n_action_steps: 每次策略调用在环境中执行的动作步数。
            详见 `DiffusionPolicy.select_action`
        input_features: 定义策略输入数据 PolicyFeature 的字典
        output_features: 定义策略输出数据 PolicyFeature 的字典
        normalization_mapping: 从 FeatureType 字符串值到 NormalizationMode 的映射字典
        vision_backbone: 用于编码图像的 torchvision resnet 主干网络名称
        resize_shape: 作为视觉主干预处理步骤的 (H, W) 图像调整大小。
            如果为 None，则不进行调整，使用原始图像分辨率
        crop_ratio: (0, 1] 范围内的比例，用于从 resize_shape 导出裁剪大小
            (crop_h = int(resize_shape[0] * crop_ratio)，宽度类似)。
            设为 1.0 禁用裁剪。仅在 resize_shape 不为 None 时生效
        crop_shape: (H, W) 图像裁剪大小。当设置 resize_shape 且 crop_ratio < 1.0 时，
            自动计算。也可直接设置用于仅使用裁剪的旧配置。
            如果为 None 且无派生适用，则不进行裁剪
        crop_is_random: 训练时裁剪是否为随机（eval 模式下始终为中心裁剪）
        pretrained_backbone_weights: 用于初始化主干的 torchvision 预训练权重。
            `None` 表示不使用预训练权重
        use_group_norm: 是否在主干中将批归一化替换为组归一化。
            组大小设为约 16 (feature_dim // 16)
        spatial_softmax_num_keypoints: SpatialSoftmax 的关键点数量
        use_separate_rgb_encoder_per_camera: 是否为每个摄像头视角使用单独的 RGB 编码器
        down_dims: 扩散建模 Unet 中每个时间下采样阶段的特征维度。
            可提供可变数量的维度，从而也控制下采样程度
        kernel_size: 扩散建模 Unet 的卷积核大小
        n_groups: Unet 卷积块中组归一化使用的组数
        diffusion_step_embed_dim: Unet 通过小型非线性网络以扩散时间步为条件。
            这是该网络的输出维度，即嵌入维度
        use_film_scale_modulation: FiLM (https://huggingface.co/papers/1709.07871) 用于 Unet 条件化。
            默认使用偏置调制，此参数指示是否也使用尺度调制
        noise_scheduler_type: 噪声调度器名称。支持的选项: ["DDPM", "DDIM"]
        num_train_timesteps: 前向扩散调度的扩散步数
        beta_schedule: 按 HuggingFace diffusers DDPMScheduler 的扩散 beta 调度名称
        beta_start: 第一个前向扩散步的 beta 值
        beta_end: 最后一个前向扩散步的 beta 值
        prediction_type: 扩散建模 Unet 的预测类型。从 "epsilon" 或 "sample" 中选择。
            从潜在变量建模角度这些有等效结果，但 "epsilon" 在许多深度神经网络设置中
            表现更好
        clip_sample: 推理时每个去噪步骤是否将样本裁剪到 [-`clip_sample_range`, +`clip_sample_range`]。
            警告: 需要确保动作空间已归一化到此范围内
        clip_sample_range: 上述裁剪范围的幅度
        num_inference_steps: 推理时使用的反向扩散步数（步骤均匀分布）。
            如果未提供，默认与 `num_train_timesteps` 相同
        do_mask_loss_for_padding: 是否在有复制填充动作时对损失进行掩码。
            详见 `LeRobotDataset` 和 `load_previous_and_future_frames`。
            注意，默认为 False，与原始 Diffusion Policy 实现一致
    """

    # 输入/输出结构
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 原始实现不为最后 7 步采样帧，
    # 这避免了过多的填充并导致更好的训练结果
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # 架构/建模
    # 视觉主干网络
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet 网络
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # 噪声调度器
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # 推理
    num_inference_steps: int | None = None

    # 优化
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # 损失计算
    do_mask_loss_for_padding: bool = False

    # 训练预设
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """输入验证（非穷尽）。"""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                # 当 crop_ratio == 1.0 时，显式禁用 resize+ratio 路径的裁剪
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        # 检查 horizon 大小和 U-Net 下采样是否兼容
        # U-Net 每个阶段下采样 2 倍
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        # 检查所有输入图像是否具有相同形状
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
