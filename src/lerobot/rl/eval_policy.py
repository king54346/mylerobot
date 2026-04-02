# !/usr/bin/env python
"""
策略评估模块 (Policy Evaluation Module)
====================================

本模块提供了在环境中评估策略性能的工具。

主要函数:
    - eval_policy: 评估策略执行 n 个 episode 的奖励
    - main: 命令行入口点
"""

import logging

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    so_leader,  # noqa: F401
)

from .gym_manipulator import make_robot_env

logging.basicConfig(level=logging.INFO)


def eval_policy(env, policy, n_episodes):
    sum_reward_episode = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        while True:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        sum_reward_episode.append(episode_reward)

    logging.info(f"Success after 20 steps {sum_reward_episode}")
    logging.info(f"success rate {sum(sum_reward_episode) / len(sum_reward_episode)}")


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    env_cfg = cfg.env
    env = make_robot_env(env_cfg)
    dataset_cfg = cfg.dataset
    dataset = LeRobotDataset(repo_id=dataset_cfg.repo_id)
    dataset_meta = dataset.meta

    policy = make_policy(
        cfg=cfg.policy,
        # env_cfg=cfg.env,
        ds_meta=dataset_meta,
    )
    policy = policy.from_pretrained(env_cfg.pretrained_policy_name_or_path)
    policy.eval()

    eval_policy(env, policy=policy, n_episodes=10)


if __name__ == "__main__":
    main()
