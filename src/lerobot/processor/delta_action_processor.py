"""
增量动作处理器模块 (Delta Action Processor Module)
==============================================

提供处理增量/相对动作的处理器步骤。

增量动作 (Delta Action) 是指相对于当前位置的位置变化量，
而不是绝对目标位置。这种控制方式在以下场景中很常见：
- 游戏手柄/键盘控制
- 逆运动学控制
- 安全的朋增量运动

主要组件：
- MapTensorToDeltaActionDictStep: 将策略输出的张量转换为增量动作字典
- MapDeltaActionToRobotActionStep: 将增量动作转换为机器人目标动作
"""

from dataclasses import dataclass

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

from .core import PolicyAction, RobotAction
from .pipeline import ActionProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(ActionProcessorStep):
    """
    将策略输出的平坦动作张量映射为结构化的增量动作字典。
    
    此步骤通常在策略输出连续动作向量后使用。
    它将向量分解为末端执行器的命名增量分量 (x, y, z) 以及可选的夹爪。
    
    输入张量格式: [delta_x, delta_y, delta_z, gripper]
    输出字典格式: {"delta_x": ..., "delta_y": ..., "delta_z": ..., "gripper": ...}
    
    属性:
        use_gripper: 如果为True，假定张量的第4个元素是夹爪动作
    """

    use_gripper: bool = True  # 是否使用夹爪动作

    def action(self, action: PolicyAction) -> RobotAction:
        if not isinstance(action, PolicyAction):
            raise ValueError("Only PolicyAction is supported for this processor")

        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): 添加旋转支持
        delta_action = {
            "delta_x": action[0].item(),  # X方向增量
            "delta_y": action[1].item(),  # Y方向增量
            "delta_z": action[2].item(),  # Z方向增量
        }
        if self.use_gripper:
            delta_action["gripper"] = action[3].item()  # 夹爪动作
        return delta_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION][f"delta_{axis}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        if self.use_gripper:
            features[PipelineFeatureType.ACTION]["gripper"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(RobotActionProcessorStep):
    """
    将遥操作设备的增量动作映射为机器人的目标动作（用于逆运动学）。
    
    此步骤将增量移动字典（例如来自游戏手柄）转换为
    包含 "enabled" 标志和目标末端执行器位置的目标动作格式。
    它还处理缩放和噪声过滤。
    
    属性:
        position_scale: 缩放增量位置输入的系数
        noise_threshold: 低于此阈值的增量输入被视为噪声，
                         不会触发 "enabled" 状态
    """

    # 增量移动的缩放系数
    position_scale: float = 1.0
    noise_threshold: float = 1e-3  # 1毫米阈值，用于过滤噪声

    def action(self, action: RobotAction) -> RobotAction:
        # 注意 (maractingi): 动作可以是来自Teleop设备的字典，也可以是来自策略的张量
        # TODO (maractingi): 从Teleop设备更改target_xyz命名约定
        delta_x = action.pop("delta_x")
        delta_y = action.pop("delta_y")
        delta_z = action.pop("delta_z")
        gripper = action.pop("gripper")

        # 判断遥操作器是否正在积极提供输入
        # 如果检测到任何显著的移动增量，则认为已启用
        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5  # 使用欧几里得范数计算位置
        enabled = position_magnitude > self.noise_threshold  # 小阈值避免噪声

        # 适当缩放增量
        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        # 对于游戏手柄/键盘，我们没有旋转输入，所以设置为0
        # 未来可以为更复杂的遥操作器扩展
        target_wx = 0.0
        target_wy = 0.0
        target_wz = 0.0

        # 使用机器人目标格式更新动作
        action = {
            "enabled": enabled,           # 是否启用控制
            "target_x": scaled_delta_x,   # X方向目标增量
            "target_y": scaled_delta_y,   # Y方向目标增量
            "target_z": scaled_delta_z,   # Z方向目标增量
            "target_wx": target_wx,       # X轴旋转目标
            "target_wy": target_wy,       # Y轴旋转目标
            "target_wz": target_wz,       # Z轴旋转目标
            "gripper_vel": float(gripper),  # 夹爪速度
        }

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z", "gripper"]:
            features[PipelineFeatureType.ACTION].pop(f"delta_{axis}", None)

        for feat in ["enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
