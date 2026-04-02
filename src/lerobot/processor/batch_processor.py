"""
批处理维度处理器模块 (Batch Dimension Processor Module)
=====================================================

本模块定义了用于向环境转移元组的各个组件添加批处理维度的处理步骤。

这些处理步骤用于处理动作、观测和补充数据，通过添加前导维度使它们适合批处理。
这是将数据送入神经网络模型前的常见需求（模型通常期望输入形状为 [batch_size, ...]）。

主要组件：
- AddBatchDimensionActionStep: 为动作张量添加批次维度
- AddBatchDimensionObservationStep: 为观测数据添加批次维度
- AddBatchDimensionComplementaryDataStep: 为补充数据添加批次维度
- AddBatchDimensionProcessorStep: 组合处理器，处理整个转移元组
"""

from dataclasses import dataclass, field

from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

from .core import EnvTransition, PolicyAction
from .pipeline import (
    ComplementaryDataProcessorStep,
    ObservationProcessorStep,
    PolicyActionProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_action")
class AddBatchDimensionActionStep(PolicyActionProcessorStep):
    """
    为1D张量动作添加批次维度的处理步骤。
    
    用于从单个动作样本创建批次大小为1的批次数据。
    例如：形状 (action_dim,) -> (1, action_dim)
    """

    def action(self, action: PolicyAction) -> PolicyAction:
        """
        如果动作是1D张量，则添加批次维度。
        
        Args:
            action: 动作张量
        
        Returns:
            添加了批次维度的动作张量
        """
        if action.dim() != 1:
            return action
        return action.unsqueeze(0)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        返回未更改的输入特征。
        
        添加批次维度不会改变特征定义。
        
        Args:
            features: 策略特征字典
        
        Returns:
            原始的策略特征字典
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_observation")
class AddBatchDimensionObservationStep(ObservationProcessorStep):
    """
    为观测数据添加批次维度的处理步骤。
    
    处理不同类型的观测数据：
    - 状态向量（1D张量）：(state_dim,) -> (1, state_dim)
    - 单张图像（3D张量）：(C, H, W) -> (1, C, H, W)
    - 多张图像字典（3D张量）：每张图像都添加批次维度
    """

    def observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        为观测字典中的张量观测数据添加批次维度。
        
        Args:
            observation: 观测字典
        
        Returns:
            添加了批次维度的观测字典
        """
        # 处理状态观测 - 如果是1D则添加批次维度
        for state_key in [OBS_STATE, OBS_ENV_STATE]:
            if state_key in observation:
                state_value = observation[state_key]
                if isinstance(state_value, Tensor) and state_value.dim() == 1:
                    observation[state_key] = state_value.unsqueeze(0)

        # 处理单张图像观测 - 如果是3D则添加批次维度
        if OBS_IMAGE in observation:
            image_value = observation[OBS_IMAGE]
            if isinstance(image_value, Tensor) and image_value.dim() == 3:
                observation[OBS_IMAGE] = image_value.unsqueeze(0)

        # 处理多张图像观测 - 如果是3D则添加批次维度
        for key, value in observation.items():
            if key.startswith(f"{OBS_IMAGES}.") and isinstance(value, Tensor) and value.dim() == 3:
                observation[key] = value.unsqueeze(0)
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        返回未更改的输入特征。
        
        添加批次维度不会改变特征定义。
        
        Args:
            features: 策略特征字典
        
        Returns:
            原始的策略特征字典
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_complementary_data")
class AddBatchDimensionComplementaryDataStep(ComplementaryDataProcessorStep):
    """
    为补充数据字段添加批次维度的处理步骤。
    
    处理特定的键，使它们变成批处理格式：
    - 'task'（字符串）：包装在列表中
    - 'index' 和 'task_index'（0D张量）：添加批次维度
    """

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        为补充数据字典中的特定字段添加批次维度。
        
        Args:
            complementary_data: 补充数据字典
        
        Returns:
            添加了批次维度的补充数据字典
        """
        # 处理task字段 - 将字符串包装在列表中以添加批次维度
        if "task" in complementary_data:
            task_value = complementary_data["task"]
            if isinstance(task_value, str):
                complementary_data["task"] = [task_value]

        # 处理index字段 - 如果是0D则添加批次维度
        if "index" in complementary_data:
            index_value = complementary_data["index"]
            if isinstance(index_value, Tensor) and index_value.dim() == 0:
                complementary_data["index"] = index_value.unsqueeze(0)

        # 处理task_index字段 - 如果是0D则添加批次维度
        if "task_index" in complementary_data:
            task_index_value = complementary_data["task_index"]
            if isinstance(task_index_value, Tensor) and task_index_value.dim() == 0:
                complementary_data["task_index"] = task_index_value.unsqueeze(0)
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        返回未更改的输入特征。
        
        添加批次维度不会改变特征定义。
        
        Args:
            features: 策略特征字典
        
        Returns:
            原始的策略特征字典
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor")
class AddBatchDimensionProcessorStep(ProcessorStep):
    """
    组合处理器步骤，为整个环境转移元组添加批次维度。
    
    该步骤组合了动作、观测和补充数据的各个处理器，
    从单实例转移元组创建批处理转移元组（批次大小为1）。
    
    属性:
        to_batch_action_processor: 动作组件的处理器
        to_batch_observation_processor: 观测组件的处理器
        to_batch_complementary_data_processor: 补充数据组件的处理器
    """

    to_batch_action_processor: AddBatchDimensionActionStep = field(
        default_factory=AddBatchDimensionActionStep
    )
    to_batch_observation_processor: AddBatchDimensionObservationStep = field(
        default_factory=AddBatchDimensionObservationStep
    )
    to_batch_complementary_data_processor: AddBatchDimensionComplementaryDataStep = field(
        default_factory=AddBatchDimensionComplementaryDataStep
    )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        将批处理过程应用于环境转移元组的所有相关部分。
        
        Args:
            transition: 要处理的环境转移元组
        
        Returns:
            添加了批次维度的环境转移元组
        """
        if transition[TransitionKey.ACTION] is not None:
            transition = self.to_batch_action_processor(transition)
        if transition[TransitionKey.OBSERVATION] is not None:
            transition = self.to_batch_observation_processor(transition)
        if transition[TransitionKey.COMPLEMENTARY_DATA] is not None:
            transition = self.to_batch_complementary_data_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        返回未更改的输入特征。
        
        添加批次维度不会改变特征定义。
        
        Args:
            features: 策略特征字典
        
        Returns:
            原始的策略特征字典
        """
        # 注意：转换特征时忽略批次维度
        return features
