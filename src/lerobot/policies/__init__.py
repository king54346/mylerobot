"""
策略模型模块 (Policy Models Module)
================================

提供各种机器人学习策略的实现。

支持的策略类型：

**模仿学习 (Imitation Learning):**
- ACT (Action Chunking with Transformers): 基于Transformer的动作分块策略
- Diffusion: 基于扩散模型的策略
- VQ-BeT: 矢量量化行为Transformer
- TDMPC: 时序差分模型预测控制

**视觉-语言-动作模型 (VLA):**
- PI0/PI05/PI0Fast: Physical Intelligence 的 PI0 系列模型
- SmolVLA: 轻量级VLA模型
- XVLA: 可扩展VLA模型
- GROOT: NVIDIA 的 Groot 模型

**强化学习 (Reinforcement Learning):**
- WallX: Wall-following 控制策略
"""

# ACT - Action Chunking with Transformers
from .act.configuration_act import ACTConfig as ACTConfig

# Diffusion Policy - 扩散策略
from .diffusion.configuration_diffusion import DiffusionConfig as DiffusionConfig

# GROOT - NVIDIA Groot
from .groot.configuration_groot import GrootConfig as GrootConfig

# PI0 系列 - Physical Intelligence
from .pi0.configuration_pi0 import PI0Config as PI0Config
from .pi0_fast.configuration_pi0_fast import PI0FastConfig as PI0FastConfig
from .pi05.configuration_pi05 import PI05Config as PI05Config

# SmolVLA - 轻量级视觉语言动作模型
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
from .smolvla.processor_smolvla import SmolVLANewLineProcessor

# TDMPC - 时序差分模型预测控制
from .tdmpc.configuration_tdmpc import TDMPCConfig as TDMPCConfig

# VQ-BeT - 矢量量化行为Transformer
from .vqbet.configuration_vqbet import VQBeTConfig as VQBeTConfig

# WallX - Wall-following策略
from .wall_x.configuration_wall_x import WallXConfig as WallXConfig

# XVLA - 可扩展VLA
from .xvla.configuration_xvla import XVLAConfig as XVLAConfig

__all__ = [
    "ACTConfig",
    "DiffusionConfig",
    "PI0Config",
    "PI05Config",
    "PI0FastConfig",
    "SmolVLAConfig",
    "SARMConfig",
    "TDMPCConfig",
    "VQBeTConfig",
    "GrootConfig",
    "XVLAConfig",
    "WallXConfig",
]
