"""
Gym动作处理器模块 (Gym Action Processor Module)
==============================================

提供在PyTorch张量和NumPy数组之间转换动作的处理器。

这对于Gym环境集成很重要，因为：
- 策略网络输出PyTorch张量
- Gym环境期望NumPy数组作为输入

主要组件：
- Torch2NumpyActionProcessorStep: 将PyTorch张量动作转换为NumPy数组
- Numpy2TorchActionProcessorStep: 将NumPy数组动作转换为PyTorch张量
"""

from dataclasses import dataclass

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .converters import to_tensor
from .core import EnvAction, EnvTransition, PolicyAction
from .hil_processor import TELEOP_ACTION_KEY
from .pipeline import ActionProcessorStep, ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("torch2numpy_action_processor")
@dataclass
class Torch2NumpyActionProcessorStep(ActionProcessorStep):
    """
    将PyTorch张量动作转换为NumPy数组。
    
    当策略的输出（通常是torch.Tensor）需要传递给
    期望NumPy数组的环境或组件时，此步骤很有用。
    
    属性:
        squeeze_batch_dim: 如果为True，当数组的第一维大小为1时删除它。
                           这对于将大小为 (1, D) 的批量动作转换为
                           大小为 (D,) 的单个动作很有用
    """

    squeeze_batch_dim: bool = True  # 是否压缩批次维度

    def action(self, action: PolicyAction) -> EnvAction:
        if not isinstance(action, PolicyAction):
            raise TypeError(
                f"Expected PolicyAction or None, got {type(action).__name__}. "
                "Use appropriate processor for non-tensor actions."
            )

        numpy_action = action.detach().cpu().numpy()  # 分离、转到CPU、转为numpy

        # 移除批次维度但保留动作维度
        # 仅当有批次维度时压缩（第一维 == 1）
        if (
            self.squeeze_batch_dim
            and numpy_action.shape
            and len(numpy_action.shape) > 1
            and numpy_action.shape[0] == 1
        ):
            numpy_action = numpy_action.squeeze(0)

        return numpy_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("numpy2torch_action_processor")
@dataclass
class Numpy2TorchActionProcessorStep(ProcessorStep):
    """
    当动作存在时，将NumPy数组动作转换为PyTorch张量。
    
    主要用于将环境返回的动作转换为可以存储在回放缓冲区的张量格式。
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        如果动作存在，将numpy动作转换为torch张量，否则直接透传。
        """
        from .core import TransitionKey

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, EnvAction):
                raise TypeError(
                    f"Expected np.ndarray or None, got {type(action).__name__}. "
                    "Use appropriate processor for non-tensor actions."
                )
            torch_action = to_tensor(action, dtype=None)  # Preserve original dtype
            new_transition[TransitionKey.ACTION] = torch_action

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        if TELEOP_ACTION_KEY in complementary_data:
            teleop_action = complementary_data[TELEOP_ACTION_KEY]
            if isinstance(teleop_action, EnvAction):
                complementary_data[TELEOP_ACTION_KEY] = to_tensor(teleop_action)
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
