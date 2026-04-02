"""
LeRobot 包初始化文件
===================

该文件定义了 LeRobot 库中所有可用的组件列表，包括：
- 可用环境 (available_envs): 支持的仿真环境，如 aloha、pusht 等
- 可用数据集 (available_datasets): 预训练数据集列表
- 可用策略 (available_policies): 支持的策略算法，如 ACT、Diffusion、TDMPC 等
- 可用机器人 (available_robots): 支持的机器人硬件
- 可用摄像头 (available_cameras): 支持的摄像头类型
- 可用电机 (available_motors): 支持的电机驱动类型

本文件采用轻量级设计，不导入所有依赖，以确保快速访问这些变量。

This file contains lists of available environments, dataset and policies to reflect the current state of LeRobot library.
We do not want to import all the dependencies, but instead we keep it lightweight to ensure fast access to these variables.

Example:
    ```python
        import lerobot
        print(lerobot.available_envs)
        print(lerobot.available_tasks_per_env)
        print(lerobot.available_datasets)
        print(lerobot.available_datasets_per_env)
        print(lerobot.available_real_world_datasets)
        print(lerobot.available_policies)
        print(lerobot.available_policies_per_env)
        print(lerobot.available_robots)
        print(lerobot.available_cameras)
        print(lerobot.available_motors)
    ```

When implementing a new dataset loadable with LeRobotDataset follow these steps:
- Update `available_datasets_per_env` in `lerobot/__init__.py`

When implementing a new environment (e.g. `gym_aloha`), follow these steps:
- Update `available_tasks_per_env` and `available_datasets_per_env` in `lerobot/__init__.py`

When implementing a new policy class (e.g. `DiffusionPolicy`) follow these steps:
- Update `available_policies` and `available_policies_per_env`, in `lerobot/__init__.py`
- Set the required `name` class attribute.
- Update variables in `tests/test_available.py` by importing your new Policy class
"""

import itertools

from lerobot.__version__ import __version__  # noqa: F401

# ==================== 环境与任务配置 ====================
# TODO(rcadene): 改进策略和环境的定义方式。目前 `available_policies` 中的项目
# 同时指向 yaml 文件和模型名称。`available_envs` 同样如此。
# 这种区别应该更加明确。

# 每个仿真环境对应的可用任务
# - aloha: 双臂协作机器人环境
#   - AlohaInsertion-v0: 插入任务
#   - AlohaTransferCube-v0: 方块转移任务
# - pusht: 推物体任务环境
available_tasks_per_env = {
    "aloha": [
        "AlohaInsertion-v0",
        "AlohaTransferCube-v0",
    ],
    "pusht": ["PushT-v0"],
}
# 所有可用仿真环境的列表
available_envs = list(available_tasks_per_env.keys())

# ==================== 数据集配置 ====================
# 每个环境对应的数据集，包含人类演示和脚本化演示
# - human: 人类遥操作收集的数据
# - scripted: 脚本自动生成的数据
# - image: 包含图像数据的版本
available_datasets_per_env = {
    "aloha": [
        "lerobot/aloha_sim_insertion_human",
        "lerobot/aloha_sim_insertion_scripted",
        "lerobot/aloha_sim_transfer_cube_human",
        "lerobot/aloha_sim_transfer_cube_scripted",
        "lerobot/aloha_sim_insertion_human_image",
        "lerobot/aloha_sim_insertion_scripted_image",
        "lerobot/aloha_sim_transfer_cube_human_image",
        "lerobot/aloha_sim_transfer_cube_scripted_image",
    ],
    # TODO(alexander-soare): Add "lerobot/pusht_keypoints". Right now we can't because this is too tightly
    # coupled with tests.
    "pusht": ["lerobot/pusht", "lerobot/pusht_image"],
}

# ==================== 真实世界数据集 ====================
# 来自各种研究机构的真实机器人数据集
# 包括：ALOHA、UMI、Unitree、NYU、UTokyo、CMU、Berkeley 等
available_real_world_datasets = [
    "lerobot/aloha_mobile_cabinet",
    "lerobot/aloha_mobile_chair",
    "lerobot/aloha_mobile_elevator",
    "lerobot/aloha_mobile_shrimp",
    "lerobot/aloha_mobile_wash_pan",
    "lerobot/aloha_mobile_wipe_wine",
    "lerobot/aloha_static_battery",
    "lerobot/aloha_static_candy",
    "lerobot/aloha_static_coffee",
    "lerobot/aloha_static_coffee_new",
    "lerobot/aloha_static_cups_open",
    "lerobot/aloha_static_fork_pick_up",
    "lerobot/aloha_static_pingpong_test",
    "lerobot/aloha_static_pro_pencil",
    "lerobot/aloha_static_screw_driver",
    "lerobot/aloha_static_tape",
    "lerobot/aloha_static_thread_velcro",
    "lerobot/aloha_static_towel",
    "lerobot/aloha_static_vinh_cup",
    "lerobot/aloha_static_vinh_cup_left",
    "lerobot/aloha_static_ziploc_slide",
    "lerobot/umi_cup_in_the_wild",
    "lerobot/unitreeh1_fold_clothes",
    "lerobot/unitreeh1_rearrange_objects",
    "lerobot/unitreeh1_two_robot_greeting",
    "lerobot/unitreeh1_warehouse",
    "lerobot/nyu_rot_dataset",
    "lerobot/utokyo_saytap",
    "lerobot/imperialcollege_sawyer_wrist_cam",
    "lerobot/utokyo_xarm_bimanual",
    "lerobot/tokyo_u_lsmo",
    "lerobot/utokyo_pr2_opening_fridge",
    "lerobot/cmu_franka_exploration_dataset",
    "lerobot/cmu_stretch",
    "lerobot/asu_table_top",
    "lerobot/utokyo_pr2_tabletop_manipulation",
    "lerobot/utokyo_xarm_pick_and_place",
    "lerobot/ucsd_kitchen_dataset",
    "lerobot/austin_buds_dataset",
    "lerobot/dlr_sara_grid_clamp",
    "lerobot/conq_hose_manipulation",
    "lerobot/columbia_cairlab_pusht_real",
    "lerobot/dlr_sara_pour",
    "lerobot/dlr_edan_shared_control",
    "lerobot/ucsd_pick_and_place_dataset",
    "lerobot/berkeley_cable_routing",
    "lerobot/nyu_franka_play_dataset",
    "lerobot/austin_sirius_dataset",
    "lerobot/cmu_play_fusion",
    "lerobot/berkeley_gnm_sac_son",
    "lerobot/nyu_door_opening_surprising_effectiveness",
    "lerobot/berkeley_fanuc_manipulation",
    "lerobot/jaco_play",
    "lerobot/viola",
    "lerobot/kaist_nonprehensile",
    "lerobot/berkeley_mvp",
    "lerobot/uiuc_d3field",
    "lerobot/berkeley_gnm_recon",
    "lerobot/austin_sailor_dataset",
    "lerobot/utaustin_mutex",
    "lerobot/roboturk",
    "lerobot/stanford_hydra_dataset",
    "lerobot/berkeley_autolab_ur5",
    "lerobot/stanford_robocook",
    "lerobot/toto",
    "lerobot/fmb",
    "lerobot/droid_100",
    "lerobot/berkeley_rpt",
    "lerobot/stanford_kuka_multimodal_dataset",
    "lerobot/iamlab_cmu_pickup_insert",
    "lerobot/taco_play",
    "lerobot/berkeley_gnm_cory_hall",
    "lerobot/usc_cloth_sim",
]

# 合并所有数据集（仿真 + 真实世界）并去重排序
available_datasets = sorted(
    set(itertools.chain(*available_datasets_per_env.values(), available_real_world_datasets))
)

# ==================== 可用策略算法 ====================
# 从 `lerobot/policies` 目录中列出所有可用的策略
# - act: Action Chunking with Transformers (ACT) 策略
# - diffusion: 扩散策略 (Diffusion Policy)
# - tdmpc: TD-MPC 强化学习策略
# - vqbet: VQ-BeT 向量量化行为 Transformer
available_policies = ["act", "diffusion", "tdmpc", "vqbet"]

# ==================== 可用机器人硬件 ====================
# 从 `lerobot/robots` 目录中列出所有支持的机器人
# - koch: Koch v1.1 低成本机械臂
# - koch_bimanual: 双臂 Koch 机器人
# - aloha: Google ALOHA 双臂系统
# - so100: SO-100 机械臂
# - so101: SO-101 机械臂
available_robots = [
    "koch",
    "koch_bimanual",
    "aloha",
    "so100",
    "so101",
]

# ==================== 可用摄像头类型 ====================
# 从 `lerobot/cameras` 目录中列出所有支持的摄像头
# - opencv: 基于 OpenCV 的通用摄像头支持
# - intelrealsense: Intel RealSense 深度摄像头
available_cameras = [
    "opencv",
    "intelrealsense",
]

# ==================== 可用电机驱动 ====================
# 从 `lerobot/motors` 目录中列出所有支持的电机类型
# - dynamixel: Dynamixel 系列伺服电机
# - feetech: Feetech 系列伺服电机
available_motors = [
    "dynamixel",
    "feetech",
]

# ==================== 环境与策略映射 ====================
# 定义每个环境推荐使用的策略配置
# 键和值都对应 yaml 配置文件
available_policies_per_env = {
    "aloha": ["act"],
    "pusht": ["diffusion", "vqbet"],
    "koch_real": ["act_koch_real"],
    "aloha_real": ["act_aloha_real"],
}

# ==================== 辅助数据结构 ====================
# 生成所有 (环境, 任务) 组合对
env_task_pairs = [(env, task) for env, tasks in available_tasks_per_env.items() for task in tasks]
# 生成所有 (环境, 数据集) 组合对
env_dataset_pairs = [
    (env, dataset) for env, datasets in available_datasets_per_env.items() for dataset in datasets
]
# 生成所有 (环境, 数据集, 策略) 三元组，用于测试和验证
env_dataset_policy_triplets = [
    (env, dataset, policy)
    for env, datasets in available_datasets_per_env.items()
    for dataset in datasets
    for policy in available_policies_per_env[env]
]
