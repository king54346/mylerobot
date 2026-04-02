"""
策略工具模块 (Policy Utils Module)
================================

提供策略模型使用的通用工具函数。

主要功能：
- 观测队列管理: 用于时序建模的历史观测队列
- 设备/类型推断: 从模型参数获取设备和数据类型
- 输出形状计算: 计算模块的输出形状
- 推理准备: 将观测数据准备为模型输入
- 动作转换: 将策略输出转换为机器人动作
- 特征验证: 验证数据集和策略特征的一致性
"""

import logging
from collections import deque

import numpy as np
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.utils import build_dataset_frame
from lerobot.processor import PolicyAction, RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STR


def populate_queues(
    queues: dict[str, deque], batch: dict[str, torch.Tensor], exclude_keys: list[str] | None = None
):
    """
    用批次数据填充观测队列。
    
    用于维护时序建模所需的历史观测窗口。
    
    Args:
        queues: 键到deque的字典，用于存储历史观测
        batch: 当前批次的张量字典
        exclude_keys: 要排除的键列表
    
    Returns:
        更新后的队列字典
    """
    if exclude_keys is None:
        exclude_keys = []
    for key in batch:
        # 忽略不在队列中的键（让调用者负责确保队列有需要的键）
        if key not in queues or key in exclude_keys:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # 通过复制第一个观测多次来初始化，直到队列满
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # 将最新的观测添加到队列
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """
    通过检查模块的一个参数获取其设备。
    
    注意: 假设所有参数都在相同设备上
    
    Args:
        module: PyTorch模块
    
    Returns:
        模块参数所在的设备
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """
    通过检查模块的一个参数获取其数据类型。
    
    注意: 假设所有参数都有相同的dtype
    
    Args:
        module: PyTorch模块
    
    Returns:
        模块参数的数据类型
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    给定输入形状，计算PyTorch模块的输出形状。
    
    Args:
        module: PyTorch模块
        input_shape: 表示输入形状的元组，例如 (batch_size, channels, height, width)
    
    Returns:
        模块的输出形状元组
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def log_model_loading_keys(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """
    记录加载模型时缺失和意外的键。
    
    Args:
        missing_keys: 期望但未找到的键
        unexpected_keys: 找到但未期望的键
    """
    if missing_keys:
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")
    if unexpected_keys:
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")


# TODO(Steven): 将此函数移动到适当的预处理器步骤
def prepare_observation_for_inference(
    observation: dict[str, np.ndarray],
    device: torch.device,
    task: str | None = None,
    robot_type: str | None = None,
) -> RobotObservation:
    """
    将观测数据转换为模型就绪的PyTorch张量。
    
    此函数接受NumPy数组字典，执行必要的预处理，
    并为模型推理做准备。步骤包括：
    1. 将NumPy数组转换为PyTorch张量
    2. 归一化和置换图像数据（如果有）
    3. 为每个张量添加批次维度
    4. 将所有张量移动到指定的计算设备
    5. 添加任务和机器人类型信息到字典
    
    Args:
        observation: 观测名称(字符串)到NumPy数组的字典。
            对于图像，期望格式为 (H, W, C)
        device: 张量将移动到的PyTorch设备（例如 'cpu' 或 'cuda'）
        task: 当前任务的可选字符串标识符
        robot_type: 使用的机器人的可选字符串标识符
    
    Returns:
        处理后用于推理的PyTorch张量字典，位于目标设备上。
        图像张量被重塑为 (C, H, W) 并归一化到 [0, 1] 范围
    """
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""

    return observation


def build_inference_frame(
    observation: RobotObservation,
    device: torch.device,
    ds_features: dict[str, dict],
    task: str | None = None,
    robot_type: str | None = None,
) -> RobotObservation:
    """
    从原始观测构建模型就绪的观测张量字典。
    
    此工具函数协调将来自环境的原始、非结构化观测
    转换为适合传递给策略模型的结构化、基于张量的格式的过程。
    
    Args:
        observation: 原始观测字典，可能包含多余的键
        device: 最终张量的目标PyTorch设备
        ds_features: 指定要从原始观测中提取哪些特征的配置字典
        task: 当前任务的可选字符串标识符
        robot_type: 使用的机器人的可选字符串标识符
    
    Returns:
        用于模型推理的预处理张量字典
    """
    # 从传入的原始观测中提取正确的键
    observation = build_dataset_frame(ds_features, observation, prefix=OBS_STR)

    # 对观测执行必要的转换
    observation = prepare_observation_for_inference(observation, device, task, robot_type)

    return observation


def make_robot_action(action_tensor: PolicyAction, ds_features: dict[str, dict]) -> RobotAction:
    """
    将策略的输出张量转换为命名动作的字典。
    
    此函数将策略模型的数值输出转换为
    人类可读和机器人可消费的格式，其中动作张量的每个维度
    都映射到一个命名的电机或执行器命令。
    
    Args:
        action_tensor: 表示策略动作的PyTorch张量，
            通常有批次维度（例如形状 [1, action_dim]）
        ds_features: 包含元数据的配置字典，包括
            对应动作张量每个索引的名称
    
    Returns:
        将动作名称（例如 "joint_1_motor"）映射到其
        对应浮点值的字典，准备发送给机器人控制器
    """
    # TODO(Steven): 检查这些步骤是否已经在所有后处理器策略中
    action_tensor = action_tensor.squeeze(0)
    action_tensor = action_tensor.to("cpu")

    action_names = ds_features[ACTION]["names"]
    act_processed_policy: RobotAction = {
        f"{name}": float(action_tensor[i]) for i, name in enumerate(action_names)
    }
    return act_processed_policy


def raise_feature_mismatch_error(
    provided_features: set[str],
    expected_features: set[str],
) -> None:
    """
    为数据集/环境和策略配置之间的特征不匹配抛出标准化的ValueError。
    
    Args:
        provided_features: 提供的特征集合
        expected_features: 期望的特征集合
    
    Raises:
        ValueError: 包含详细的不匹配信息和建议
    """
    missing = expected_features - provided_features
    extra = provided_features - expected_features
    # TODO (jadechoghari): 为用户提供动态的重命名映射建议
    raise ValueError(
        f"Feature mismatch between dataset/environment and policy config.\n"
        f"- Missing features: {sorted(missing) if missing else 'None'}\n"
        f"- Extra features: {sorted(extra) if extra else 'None'}\n\n"
        f"Please ensure your dataset and policy use consistent feature names.\n"
        f"If your dataset uses different observation keys (e.g., cameras named differently), "
        f"use the `--rename_map` argument, for example:\n"
        f'  --rename_map=\'{{"observation.images.left": "observation.images.camera1", '
        f'"observation.images.top": "observation.images.camera2"}}\''
    )


def validate_visual_features_consistency(
    cfg: PreTrainedConfig,
    features: dict[str, PolicyFeature],
) -> None:
    """
    验证策略配置和提供的数据集/环境特征之间的视觉特征一致性。
    
    如果满足以下任一条件，验证通过：
    - 策略期望的视觉特征是数据集的子集（策略使用部分相机，数据集有更多）
    - 数据集提供的视觉特征是策略的子集（策略声明额外的以提高灵活性）
    
    Args:
        cfg: 包含input_features和类型的模型或策略配置
        features: 特征名称到PolicyFeature对象的映射
    
    Raises:
        ValueError: 如果视觉特征不一致
    """
    expected_visuals = {k for k, v in cfg.input_features.items() if v.type == FeatureType.VISUAL}
    provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}

    # 如果任一方向是子集则接受
    policy_subset_of_dataset = expected_visuals.issubset(provided_visuals)
    dataset_subset_of_policy = provided_visuals.issubset(expected_visuals)

    if not (policy_subset_of_dataset or dataset_subset_of_policy):
        raise_feature_mismatch_error(provided_visuals, expected_visuals)
