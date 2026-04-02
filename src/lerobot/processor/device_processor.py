"""
设备处理器模块 (Device Processor Module)
===================================

本模块定义了一个处理器步骤，用于将环境转移元组数据移动到指定的torch设备，
并转换其浮点精度。

主要用途：
- 将CPU上的数据转移到GPU进行加速计算
- 将数据从高精度转换为低精度以节省显存
- 支持多 GPU 场景
"""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.utils import get_safe_torch_device

from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceProcessorStep(ProcessorStep):
    """
    将 `EnvTransition` 中所有张量移动到指定设备并可选转换其浮点数据类型的处理器步骤。
    
    这对于在GPU等硬件上准备模型训练或推理的数据至关重要。
    
    属性:
        device: 张量的目标设备（例如 "cpu", "cuda", "cuda:0"）
        float_dtype: 目标浮点数据类型字符串（例如 "float32", "float16", "bfloat16"）
                     如果为None，则不更改数据类型
    """

    device: str = "cpu"  # 目标设备
    float_dtype: str | None = None  # 目标浮点类型

    # 数据类型映射表
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "half": torch.float16,
        "float": torch.float32,
        "double": torch.float64,
    }

    def __post_init__(self):
        """
        通过将字符串配置转换为torch对象来初始化处理器。
        
        该方法设置 `torch.device`，确定传输是否可以是非阻塞的，
        并验证 `float_dtype` 字符串，将其转换为 `torch.dtype` 对象。
        """
        self.tensor_device: torch.device = get_safe_torch_device(self.device)
        # 更新设备字符串，以防选择了特定GPU（例如 "cuda" -> "cuda:0"）
        self.device = self.tensor_device.type
        self.non_blocking = "cuda" in str(self.device)

        # 验证并将float_dtype字符串转换为torch dtype
        if self.float_dtype is not None:
            if self.float_dtype not in self.DTYPE_MAPPING:
                raise ValueError(
                    f"Invalid float_dtype '{self.float_dtype}'. Available options: {list(self.DTYPE_MAPPING.keys())}"
                )
            self._target_float_dtype = self.DTYPE_MAPPING[self.float_dtype]
        else:
            self._target_float_dtype = None

    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将单个张量移动到目标设备并转换其数据类型。
        
        处理多GPU场景：如果张量已经在不同于目标的CUDA设备上，
        则不移动它。这在使用Accelerate等框架时很有用。
        
        Args:
            tensor: 输入的torch.Tensor
        
        Returns:
            在正确设备上并具有正确数据类型的处理后张量
        """
        # 确定目标设备
        if tensor.is_cuda and self.tensor_device.type == "cuda":
            # 张量和目标都在GPU上 - 保留张量的GPU位置
            # 这处理了多GPU场景，其中Accelerate已经将张量
            # 放置在每个进程的正确GPU上
            target_device = tensor.device
        else:
            # 张量在CPU上，或者我们配置为CPU
            # 在两种情况下，使用配置的设备
            target_device = self.tensor_device

        # MPS解决方案：将float64转换为float32，因为MPS不支持float64
        if target_device.type == "mps" and tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)

        # 仅在必要时移动
        if tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=self.non_blocking)

        # 如果指定了目标浮点类型且张量是浮点类型，则转换
        if self._target_float_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=self._target_float_dtype)

        return tensor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        将设备和数据类型转换应用于环境转移元组中的所有张量。
        
        它遍历转移元组，找到所有 `torch.Tensor` 对象（包括嵌套在
        `observation` 等字典中的），并处理它们。
        
        Args:
            transition: 输入的 `EnvTransition` 对象
        
        Returns:
            所有张量都已移动到目标设备和数据类型的新 `EnvTransition` 对象
        """
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is not None and not isinstance(action, PolicyAction):
            raise ValueError(f"If action is not None should be a PolicyAction type got {type(action)}")

        simple_tensor_keys = [
            TransitionKey.ACTION,
            TransitionKey.REWARD,
            TransitionKey.DONE,
            TransitionKey.TRUNCATED,
        ]

        dict_tensor_keys = [
            TransitionKey.OBSERVATION,
            TransitionKey.COMPLEMENTARY_DATA,
        ]

        # 处理简单的顶层张量
        for key in simple_tensor_keys:
            value = transition.get(key)
            if isinstance(value, torch.Tensor):
                new_transition[key] = self._process_tensor(value)

        # 处理嵌套在字典中的张量
        for key in dict_tensor_keys:
            data_dict = transition.get(key)
            if data_dict is not None:
                new_data_dict = {
                    k: self._process_tensor(v) if isinstance(v, torch.Tensor) else v
                    for k, v in data_dict.items()
                }
                new_transition[key] = new_data_dict

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        返回处理器的可序列化配置。
        
        Returns:
            包含设备和浮点数据类型设置的字典
        """
        return {"device": self.device, "float_dtype": self.float_dtype}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        返回未更改的输入特征。
        
        设备和数据类型转换不会改变特征的基本定义（例如形状）。
        
        Args:
            features: 策略特征字典
        
        Returns:
            原始的策略特征字典
        """
        return features
