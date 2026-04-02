"""
环境处理器模块 (Environment Processor Module)
============================================

提供将不同环境（如LIBERO、IsaacLab）的观测数据转换为LeRobot标准格式的处理器。

主要组件：
- LiberoProcessorStep: 处理LIBERO仿真环境的观测数据
- IsaaclabArenaProcessorStep: 处理IsaacLab Arena环境的观测数据
"""

from dataclasses import dataclass

import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE, OBS_STR

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="libero_processor")
class LiberoProcessorStep(ObservationProcessorStep):
    """
    将LIBERO观测数据处理为LeRobot格式。
    
    此步骤处理LIBERO环境的特定观测结构，
    包括嵌套的robot_state字典和图像观测。
    
    **状态处理:**
    - 处理包含嵌套末端执行器、夹爪和关节信息的 `robot_state` 字典
    - 提取并连接:
        - 末端执行器位置 (3D)
        - 末端执行器四元数转换为轴角 (3D)
        - 夹爪关节位置 (2D)
    - 将连接后的状态映射到 `"observation.state"`
    
    **图像处理:**
    - 通过翻转高度和宽度维度将图像旋转180度
    - 这是为了适应HuggingFaceVLA/libero的相机方向约定
    """

    def _process_observation(self, observation):
        """
        处理来自LIBERO的图像和robot_state观测数据。
        """
        processed_obs = observation.copy()
        for key in list(processed_obs.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]

                # 翻转H和W维度实现180度旋转
                img = torch.flip(img, dims=[2, 3])

                processed_obs[key] = img
        # 将robot_state处理为平坦的状态向量
        observation_robot_state_str = OBS_PREFIX + "robot_state"
        if observation_robot_state_str in processed_obs:
            robot_state = processed_obs.pop(observation_robot_state_str)

            # 提取各个组件
            eef_pos = robot_state["eef"]["pos"]  # (B, 3,) 末端执行器位置
            eef_quat = robot_state["eef"]["quat"]  # (B, 4,) 末端执行器四元数
            gripper_qpos = robot_state["gripper"]["qpos"]  # (B, 2,) 夹爪关节位置

            # 将四元数转换为轴角表示
            eef_axisangle = self._quat2axisangle(eef_quat)  # (B, 3)
            # 连接成单个状态向量
            state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)

            # 确保为float32类型
            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            processed_obs[OBS_STATE] = state
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        将特征键从LIBERO格式转换为LeRobot标准格式。
        """
        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}

        # 复制非 STATE 类型的特征
        for ft, feats in features.items():
            if ft != FeatureType.STATE:
                new_features[ft] = feats.copy()

        # 重建 STATE 特征
        state_feats = {}

        # 添加新的平坦化状态
        state_feats[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(8,),  # [eef_pos(3), axis_angle(3), gripper(2)] 末端位置+轴角+夹爪
        )

        new_features[FeatureType.STATE] = state_feats

        return new_features

    def observation(self, observation):
        return self._process_observation(observation)

    def _quat2axisangle(self, quat: torch.Tensor) -> torch.Tensor:
        """
        将批量四元数转换为轴角格式。
        仅接受形状为 (B, 4) 的torch张量。
        
        Args:
            quat (Tensor): (B, 4) 四元数张量，格式为 (x, y, z, w)
        
        Returns:
            Tensor: (B, 3) 轴角向量
        
        Raises:
            TypeError: 如果输入不是torch张量
            ValueError: 如果形状不是 (B, 4)
        """

        if not isinstance(quat, torch.Tensor):
            raise TypeError(f"_quat2axisangle expected a torch.Tensor, got {type(quat)}")

        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"_quat2axisangle expected shape (B, 4), got {tuple(quat.shape)}")

        quat = quat.to(dtype=torch.float32)
        device = quat.device
        batch_size = quat.shape[0]

        w = quat[:, 3].clamp(-1.0, 1.0)

        den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

        result = torch.zeros((batch_size, 3), device=device)

        mask = den > 1e-10

        if mask.any():
            angle = 2.0 * torch.acos(w[mask])  # (M,)
            axis = quat[mask, :3] / den[mask].unsqueeze(1)
            result[mask] = axis * angle.unsqueeze(1)

        return result


@dataclass
@ProcessorStepRegistry.register(name="isaaclab_arena_processor")
class IsaaclabArenaProcessorStep(ObservationProcessorStep):
    """
    将IsaacLab Arena观测数据处理为LeRobot格式。
    
    **状态处理:**
    - 根据 `state_keys` 从 obs["policy"] 中提取状态组件
    - 连接成平坦向量并映射到 "observation.state"
    
    **图像处理:**
    - 根据 `camera_keys` 从 obs["camera_obs"] 中提取图像
    - 从 (B, H, W, C) uint8 转换为 (B, C, H, W) float32 [0, 1]
    - 映射到 "observation.images.<camera_name>"
    """

    # 可从 IsaacLabEnv 配置 / CLI参数配置: --env.state_keys="robot_joint_pos,left_eef_pos"
    state_keys: tuple[str, ...]  # 要提取的状态键

    # 可从 IsaacLabEnv 配置 / CLI参数配置: --env.camera_keys="robot_pov_cam_rgb"
    camera_keys: tuple[str, ...]  # 要提取的相机键

    def _process_observation(self, observation):
        """
        处理来自IsaacLab Arena的图像和策略状态观测数据。
        """
        processed_obs = {}

        if f"{OBS_STR}.camera_obs" in observation:
            camera_obs = observation[f"{OBS_STR}.camera_obs"]

            for cam_name, img in camera_obs.items():
                if cam_name not in self.camera_keys:
                    continue

                img = img.permute(0, 3, 1, 2).contiguous()
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                elif img.dtype != torch.float32:
                    img = img.float()

                processed_obs[f"{OBS_IMAGES}.{cam_name}"] = img

        # 处理策略状态 -> observation.state
        if f"{OBS_STR}.policy" in observation:
            policy_obs = observation[f"{OBS_STR}.policy"]

            # 按顺序收集状态组件
            state_components = []
            for key in self.state_keys:
                if key in policy_obs:
                    component = policy_obs[key]
                    # 展平额外维度: (B, N, M) -> (B, N*M)
                    if component.dim() > 2:
                        batch_size = component.shape[0]
                        component = component.view(batch_size, -1)
                    state_components.append(component)

            if state_components:
                state = torch.cat(state_components, dim=-1)
                state = state.float()
                processed_obs[OBS_STATE] = state

        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """策略评估时不使用。"""
        return features

    def observation(self, observation):
        return self._process_observation(observation)
