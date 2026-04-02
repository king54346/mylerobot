# !/usr/bin/env python

"""
分布式 HILSerl 机器人策略训练的 Learner 服务器运行器。

本脚本实现了分布式 HILSerl 架构的 Learner 组件。
它初始化策略网络、维护回放缓冲区，并根据从 Actor 服务器
接收的转换更新策略。

使用示例:

- 启动 Learner 服务器进行训练:
```bash
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**注意**: 在启动 Actor 服务器之前先启动 Learner 服务器。
Learner 打开 gRPC 服务器与 Actor 通信。

**注意**: 如果配置中设置了 wandb.enable 为 true，
可以通过 Weights & Biases 监控训练进度。

**工作流程**:
1. 创建具有正确策略、数据集和环境设置的训练配置
2. 使用该配置启动此 Learner 服务器
3. 使用相同配置启动 Actor 服务器
4. 通过 wandb 仪表板监控训练进度

更多关于完整 HILSerl 训练工作流程的详细信息，请参见:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pformat

import grpc
import torch
from termcolor import colored
from torch import nn
from torch.multiprocessing import Queue
from torch.optim.optimizer import Optimizer

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2_grpc
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    bytes_to_python_object,
    bytes_to_transitions,
    state_to_bytes,
)
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state as utils_load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.transition import move_state_dict_to_device, move_transition_to_device
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_logging,
)

from .learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService


@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")

    # Use the job_name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )

    logging.info("[LEARNER] train_cli finished")


def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    初始化并运行训练过程的主训练函数。

    参数:
        cfg (TrainRLServerPipelineConfig): 训练配置
        job_name (str | None, optional): 用于日志记录的作业名称。默认为 None。
    """

    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as a parameter")

    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # 创建日志目录，确保目录存在
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # 使用显式日志文件初始化日志记录
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")
    logging.info(pformat(cfg.to_dict()))

    # 如果配置中启用了 wandb，则设置 WandB 日志记录
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.rl.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    # 处理恢复逻辑
    cfg = handle_resume_logic(cfg)

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )


def start_learner_threads(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
) -> None:
    """
    启动用于训练的 Learner 线程。

    参数:
        cfg (TrainRLServerPipelineConfig): 训练配置
        wandb_logger (WandBLogger | None): 用于指标的日志记录器
        shutdown_event: 用于信号关闭的事件
    """
    # 创建多进程队列
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
            parameters_queue,
            transition_queue,
            interaction_message_queue,
            shutdown_event,
            cfg,
        ),
        daemon=True,
    )
    communication_process.start()

    add_actor_information_and_train(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        parameters_queue=parameters_queue,
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")

    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")


# 核心算法函数


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,  # Event,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue,
):
    """
    处理从 Actor 到 Learner 的数据传输，管理训练更新，
    并在在线强化学习设置中记录训练进度。

    此函数持续:
    - 将转换从 Actor 传输到回放缓冲区。
    - 记录接收到的交互消息。
    - 确保仅当回放缓冲区有足够的转换时才开始训练。
    - 从回放缓冲区采样批次并执行多次 critic 更新。
    - 定期更新 actor、critic 和温度优化器。
    - 记录训练统计信息，包括损失值和优化频率。

    注意: 此函数没有单一职责，应该在将来拆分为多个函数。
    我们这样做的原因是 Python 的 GIL。它非常慢，性能会下降 200 倍。
    因此我们需要有一个单独的线程来完成所有工作。

    参数:
        cfg (TrainRLServerPipelineConfig): 包含超参数的配置对象。
        wandb_logger (WandBLogger | None): 用于跟踪训练进度的日志记录器。
        shutdown_event (Event): 用于信号关闭的事件。
        transition_queue (Queue): 用于从 Actor 接收转换的队列。
        interaction_message_queue (Queue): 用于从 Actor 接收交互消息的队列。
        parameters_queue (Queue): 用于向 Actor 发送策略参数的队列。
    """
    # 在开始时提取所有配置变量，可提高 7% 的速度性能
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch

    # 为多进程初始化日志记录
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("为 Actor 信息和训练进程初始化日志记录")

    logging.info("初始化策略")

    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )

    assert isinstance(policy, nn.Module)

    policy.train()

    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # 如果我们正在恢复，需要加载训练状态
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    if cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device,
        )
        batch_size: int = batch_size // 2  # 我们将从两个回放缓冲区采样

    logging.info("启动 Learner 线程")
    interaction_message = None
    optimization_step = resume_optimization_step if resume_optimization_step is not None else 0
    interaction_step_shift = resume_interaction_step if resume_interaction_step is not None else 0

    dataset_repo_id = None
    if cfg.dataset is not None:
        dataset_repo_id = cfg.dataset.repo_id

    # 初始化迭代器
    online_iterator = None
    offline_iterator = None

    # 注意: 这是 LEARNER 的主循环
    while True:
        # 如果请求关闭则退出训练循环
        if shutdown_event is not None and shutdown_event.is_set():
            logging.info("[LEARNER] Shutdown signal received. Exiting...")
            break

        # 处理所有可用的转换到回放缓冲区，由 Actor 服务器发送
        process_transitions(
            transition_queue=transition_queue,
            replay_buffer=replay_buffer,
            offline_replay_buffer=offline_replay_buffer,
            device=device,
            dataset_repo_id=dataset_repo_id,
            shutdown_event=shutdown_event,
        )

        # 处理所有可用的交互消息，由 Actor 服务器发送
        interaction_message = process_interaction_messages(
            interaction_message_queue=interaction_message_queue,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
            shutdown_event=shutdown_event,
        )

        # 等待回放缓冲区有足够的样本来开始训练
        if len(replay_buffer) < online_step_before_learning:
            continue

        if online_iterator is None:
            online_iterator = replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        if offline_replay_buffer is not None and offline_iterator is None:
            offline_iterator = offline_replay_buffer.get_iterator(
                batch_size=batch_size, async_prefetch=async_prefetch, queue_size=2
            )

        time_for_one_optimization_step = time.time()
        for _ in range(utd_ratio - 1):
            # 从迭代器采样
            batch = next(online_iterator)

            if dataset_repo_id is not None:
                batch_offline = next(offline_iterator)
                batch = concatenate_batch_transitions(
                    left_batch_transitions=batch, right_batch_transition=batch_offline
                )

            actions = batch[ACTION]
            rewards = batch["reward"]
            observations = batch["state"]
            next_observations = batch["next_state"]
            done = batch["done"]
            check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

            observation_features, next_observation_features = get_observation_features(
                policy=policy, observations=observations, next_observations=next_observations
            )

            # 创建包含 forward 方法所需所有元素的批次字典
            forward_batch = {
                ACTION: actions,
                "reward": rewards,
                "state": observations,
                "next_state": next_observations,
                "done": done,
                "observation_feature": observation_features,
                "next_observation_feature": next_observation_features,
                "complementary_info": batch["complementary_info"],
            }

            # 使用 forward 方法计算 critic 损失
            critic_output = policy.forward(forward_batch, model="critic")

            # 主 critic 优化
            loss_critic = critic_output["loss_critic"]
            optimizers["critic"].zero_grad()
            loss_critic.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
            )
            optimizers["critic"].step()

            # 离散 critic 优化（如果可用）
            if policy.config.num_discrete_actions is not None:
                discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
                loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
                optimizers["discrete_critic"].zero_grad()
                loss_discrete_critic.backward()
                discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
                )
                optimizers["discrete_critic"].step()

            # 更新目标网络（主网络和离散网络）
            policy.update_target_networks()

        # 为 UTD 比率中的最后一次更新采样
        batch = next(online_iterator)

        if dataset_repo_id is not None:
            batch_offline = next(offline_iterator)
            batch = concatenate_batch_transitions(
                left_batch_transitions=batch, right_batch_transition=batch_offline
            )

        actions = batch[ACTION]
        rewards = batch["reward"]
        observations = batch["state"]
        next_observations = batch["next_state"]
        done = batch["done"]

        check_nan_in_transition(observations=observations, actions=actions, next_state=next_observations)

        observation_features, next_observation_features = get_observation_features(
            policy=policy, observations=observations, next_observations=next_observations
        )

        # 创建包含 forward 方法所需所有元素的批次字典
        forward_batch = {
            ACTION: actions,
            "reward": rewards,
            "state": observations,
            "next_state": next_observations,
            "done": done,
            "observation_feature": observation_features,
            "next_observation_feature": next_observation_features,
        }

        critic_output = policy.forward(forward_batch, model="critic")

        loss_critic = critic_output["loss_critic"]
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters=policy.critic_ensemble.parameters(), max_norm=clip_grad_norm_value
        ).item()
        optimizers["critic"].step()

        # 初始化训练信息字典
        training_infos = {
            "loss_critic": loss_critic.item(),
            "critic_grad_norm": critic_grad_norm,
        }

        # 离散 critic 优化（如果可用）
        if policy.config.num_discrete_actions is not None:
            discrete_critic_output = policy.forward(forward_batch, model="discrete_critic")
            loss_discrete_critic = discrete_critic_output["loss_discrete_critic"]
            optimizers["discrete_critic"].zero_grad()
            loss_discrete_critic.backward()
            discrete_critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=policy.discrete_critic.parameters(), max_norm=clip_grad_norm_value
            ).item()
            optimizers["discrete_critic"].step()

            # 将离散 critic 信息添加到训练信息
            training_infos["loss_discrete_critic"] = loss_discrete_critic.item()
            training_infos["discrete_critic_grad_norm"] = discrete_critic_grad_norm

        # Actor 和温度优化（按指定频率）
        if optimization_step % policy_update_freq == 0:
            for _ in range(policy_update_freq):
                # Actor 优化
                actor_output = policy.forward(forward_batch, model="actor")
                loss_actor = actor_output["loss_actor"]
                optimizers["actor"].zero_grad()
                loss_actor.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=policy.actor.parameters(), max_norm=clip_grad_norm_value
                ).item()
                optimizers["actor"].step()

                # 将 actor 信息添加到训练信息
                training_infos["loss_actor"] = loss_actor.item()
                training_infos["actor_grad_norm"] = actor_grad_norm

                # 温度优化
                temperature_output = policy.forward(forward_batch, model="temperature")
                loss_temperature = temperature_output["loss_temperature"]
                optimizers["temperature"].zero_grad()
                loss_temperature.backward()
                temp_grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=[policy.log_alpha], max_norm=clip_grad_norm_value
                ).item()
                optimizers["temperature"].step()

                # 将温度信息添加到训练信息
                training_infos["loss_temperature"] = loss_temperature.item()
                training_infos["temperature_grad_norm"] = temp_grad_norm
                training_infos["temperature"] = policy.temperature

        # 如果需要，将策略推送到 Actor
        if time.time() - last_time_policy_pushed > policy_parameters_push_frequency:
            push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)
            last_time_policy_pushed = time.time()

        # 更新目标网络（主网络和离散网络）
        policy.update_target_networks()

        # 在指定间隔记录训练指标
        if optimization_step % log_freq == 0:
            training_infos["replay_buffer_size"] = len(replay_buffer)
            if offline_replay_buffer is not None:
                training_infos["offline_replay_buffer_size"] = len(offline_replay_buffer)
            training_infos["Optimization step"] = optimization_step

            # 记录训练指标
            if wandb_logger:
                wandb_logger.log_dict(d=training_infos, mode="train", custom_step_key="Optimization step")

        # 计算并记录优化频率
        time_for_one_optimization_step = time.time() - time_for_one_optimization_step
        frequency_for_one_optimization_step = 1 / (time_for_one_optimization_step + 1e-9)

        logging.info(f"[LEARNER] Optimization frequency loop [Hz]: {frequency_for_one_optimization_step}")

        # 记录优化频率
        if wandb_logger:
            wandb_logger.log_dict(
                {
                    "Optimization frequency loop [Hz]": frequency_for_one_optimization_step,
                    "Optimization step": optimization_step,
                },
                mode="train",
                custom_step_key="Optimization step",
            )

        optimization_step += 1
        if optimization_step % log_freq == 0:
            logging.info(f"[LEARNER] Number of optimization step: {optimization_step}")

        # 在指定间隔保存检查点
        if saving_checkpoint and (optimization_step % save_freq == 0 or optimization_step == online_steps):
            save_training_checkpoint(
                cfg=cfg,
                optimization_step=optimization_step,
                online_steps=online_steps,
                interaction_message=interaction_message,
                policy=policy,
                optimizers=optimizers,
                replay_buffer=replay_buffer,
                offline_replay_buffer=offline_replay_buffer,
                dataset_repo_id=dataset_repo_id,
                fps=fps,
            )


def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any,  # Event,
    cfg: TrainRLServerPipelineConfig,
):
    """
    启动用于训练的 Learner 服务器。
    它将从 Actor 服务器接收转换和交互消息，
    并向 Actor 服务器发送策略参数。

    参数:
        parameters_queue: 用于向 Actor 发送策略参数的队列
        transition_queue: 用于从 Actor 接收转换的队列
        interaction_message_queue: 用于从 Actor 接收交互消息的队列
        shutdown_event: 用于信号关闭的事件
        cfg: 训练配置
    """
    if not use_threads(cfg):
        # 创建进程特定的日志文件
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # 使用显式日志文件初始化日志记录
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner 服务器进程日志记录已初始化")

        # 设置进程信号处理程序来处理关闭信号
        # 但使用来自主进程的关闭事件
        # 为 MP 返回
        # TODO: 检查是否有用
        _ = ProcessSignalHandler(False, display_pid=True)

    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server,
    )

    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(SHUTDOWN_TIMEOUT)
    logging.info("[LEARNER] gRPC server stopped")


def save_training_checkpoint(
    cfg: TrainRLServerPipelineConfig,
    optimization_step: int,
    online_steps: int,
    interaction_message: dict | None,
    policy: nn.Module,
    optimizers: dict[str, Optimizer],
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer | None = None,
    dataset_repo_id: str | None = None,
    fps: int = 30,
) -> None:
    """
    保存训练检查点和相关数据。

    此函数执行以下步骤:
    1. 使用当前优化步骤创建检查点目录
    2. 保存策略模型、配置和优化器状态
    3. 保存当前交互步骤以便恢复训练
    4. 更新 "last" 检查点符号链接指向此检查点
    5. 将回放缓冲区保存为数据集以便以后使用
    6. 如果存在离线回放缓冲区，将其保存为单独的数据集

    参数:
        cfg: 训练配置
        optimization_step: 当前优化步骤
        online_steps: 总在线步骤数
        interaction_message: 包含交互信息的字典
        policy: 要保存的策略模型
        optimizers: 优化器字典
        replay_buffer: 要保存为数据集的回放缓冲区
        offline_replay_buffer: 可选的离线回放缓冲区要保存
        dataset_repo_id: 数据集的仓库 ID
        fps: 数据集的帧率
    """
    logging.info(f"Checkpoint policy after step {optimization_step}")
    _num_digits = max(6, len(str(online_steps)))
    interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

    # 创建检查点目录
    checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

    # 保存检查点
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=optimization_step,
        cfg=cfg,
        policy=policy,
        optimizer=optimizers,
        scheduler=None,
    )

    # 手动保存交互步骤
    training_state_dir = os.path.join(checkpoint_dir, TRAINING_STATE_DIR)
    os.makedirs(training_state_dir, exist_ok=True)
    training_state = {"step": optimization_step, "interaction_step": interaction_step}
    torch.save(training_state, os.path.join(training_state_dir, "training_state.pt"))

    # 更新 "last" 符号链接
    update_last_checkpoint(checkpoint_dir)

    # TODO: 临时在这里保存回放缓冲区，以后在机器人上时移除
    # 我们希望通过键盘输入来控制这个
    dataset_dir = os.path.join(cfg.output_dir, "dataset")
    if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)

    # 保存数据集
    # 注意: 处理配置中未指定数据集 repo id 的情况
    # 例如: 不使用演示数据的 RL 训练
    repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
    replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=dataset_dir)

    if offline_replay_buffer is not None:
        dataset_offline_dir = os.path.join(cfg.output_dir, "dataset_offline")
        if os.path.exists(dataset_offline_dir) and os.path.isdir(dataset_offline_dir):
            shutil.rmtree(dataset_offline_dir)

        offline_replay_buffer.to_lerobot_dataset(
            cfg.dataset.repo_id,
            fps=fps,
            root=dataset_offline_dir,
        )

    logging.info("恢复训练")


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    为强化学习策略的 actor、critic 和温度组件创建并返回优化器。

    此函数为以下组件设置 Adam 优化器:
    - **actor 网络**，确保只优化相关参数。
    - **critic 集成**，评估值函数。
    - **温度参数**，控制类 SAC 方法中的熵。

    它还初始化了学习率调度器，尽管目前设置为 `None`。

    注意:
    - 如果编码器是共享的，其参数将从 actor 的优化过程中排除。
    - 策略的对数温度 (`log_alpha`) 被包装在列表中以确保作为独立张量正确优化。

    参数:
        cfg: 包含超参数的配置对象。
        policy (nn.Module): 包含 actor、critic 和温度组件的策略模型。

    返回:
        Tuple[Dict[str, torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler]]:
        一个包含以下内容的元组:
        - `optimizers`: 一个将组件名称 ("actor", "critic", "temperature") 映射到各自 Adam 优化器的字典。
        - `lr_scheduler`: 目前设置为 `None`，但可以扩展以支持学习率调度。

    """
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    if cfg.policy.num_discrete_actions is not None:
        optimizer_discrete_critic = torch.optim.Adam(
            params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
        )
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
    lr_scheduler = None
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    if cfg.policy.num_discrete_actions is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic
    return optimizers, lr_scheduler


# 训练设置函数


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    处理训练的恢复逻辑。

    如果 resume 为 True:
    - 验证检查点是否存在
    - 加载检查点配置
    - 记录恢复详情
    - 返回检查点配置

    如果 resume 为 False:
    - 检查输出目录是否存在（以防止意外覆盖）
    - 返回原始配置

    参数:
        cfg (TrainRLServerPipelineConfig): 训练配置

    返回:
        TrainRLServerPipelineConfig: 更新后的配置

    异常:
        RuntimeError: 如果 resume 为 True 但未找到检查点，或者如果 resume 为 False 但目录存在
    """
    out_dir = cfg.output_dir

    # 情况 1: 不恢复，但需要检查目录是否存在以防止覆盖
    if not cfg.resume:
        checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg

    # 情况 2: 恢复训练
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")

    # 记录找到有效检查点并正在恢复
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"],
        )
    )

    # 使用 Draccus 加载配置
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # 确保返回的配置中设置了 resume 标志
    checkpoint_cfg.resume = True
    return checkpoint_cfg


def load_training_state(
    cfg: TrainRLServerPipelineConfig,
    optimizers: Optimizer | dict[str, Optimizer],
):
    """
    从检查点加载训练状态（优化器、步数等）。

    参数:
        cfg (TrainRLServerPipelineConfig): 训练配置
        optimizers (Optimizer | dict): 要加载状态的优化器

    返回:
        tuple: (optimization_step, interaction_step) 或 (None, None) 如果不恢复
    """
    if not cfg.resume:
        return None, None

    # 构建到最后一个检查点目录的路径
    checkpoint_dir = os.path.join(cfg.output_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)

    logging.info(f"Loading training state from {checkpoint_dir}")

    try:
        # 使用 train_utils 中的实用函数加载优化器状态
        step, optimizers, _ = utils_load_training_state(Path(checkpoint_dir), optimizers, None)

        # 从 training_state.pt 单独加载交互步骤
        training_state_path = os.path.join(checkpoint_dir, TRAINING_STATE_DIR, "training_state.pt")
        interaction_step = 0
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, weights_only=False)  # nosec B614: Safe usage of torch.load
            interaction_step = training_state.get("interaction_step", 0)

        logging.info(f"Resuming from step {step}, interaction step {interaction_step}")
        return step, interaction_step

    except Exception as e:
        logging.error(f"Failed to load training state: {e}")
        return None, None


def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module) -> None:
    """
    记录关于训练过程的信息。

    参数:
        cfg (TrainRLServerPipelineConfig): 训练配置
        policy (nn.Module): 策略模型
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
    cfg: TrainRLServerPipelineConfig, device: str, storage_device: str
) -> ReplayBuffer:
    """
    初始化回放缓冲区，如果恢复则从数据集初始化，否则初始化为空。

    参数:
        cfg (TrainRLServerPipelineConfig): 训练配置
        device (str): 存储张量的设备
        storage_device (str): 用于存储优化的设备

    返回:
        ReplayBuffer: 初始化的回放缓冲区
    """
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    logging.info("恢复训练，加载在线数据集")
    dataset_path = os.path.join(cfg.output_dir, "dataset")

    # 注意: 在 RL 中可能没有数据集。
    repo_id = None
    if cfg.dataset is not None:
        repo_id = cfg.dataset.repo_id
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_path,
    )
    return ReplayBuffer.from_lerobot_dataset(
        lerobot_dataset=dataset,
        capacity=cfg.policy.online_buffer_capacity,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        optimize_memory=True,
    )


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device: str,
) -> ReplayBuffer:
    """
    从数据集初始化离线回放缓冲区。

    参数:
        cfg (TrainRLServerPipelineConfig): 训练配置
        device (str): 存储张量的设备
        storage_device (str): 用于存储优化的设备

    返回:
        ReplayBuffer: 初始化的离线回放缓冲区
    """
    if not cfg.resume:
        logging.info("创建离线缓冲区的数据集")
        offline_dataset = make_dataset(cfg)
    else:
        logging.info("加载离线数据集")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=dataset_offline_path,
        )

    logging.info("转换为离线回放缓冲区")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    return offline_replay_buffer


# 实用工具/辅助函数


def get_observation_features(
    policy: SACPolicy, observations: torch.Tensor, next_observations: torch.Tensor
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    从策略编码器获取观测特征。它作为观测特征的缓存。
    当编码器被冻结时，观测特征不会更新。
    我们可以通过缓存观测特征来节省计算。

    参数:
        policy: 策略模型
        observations: 当前观测
        next_observations: 下一个观测

    返回:
        tuple: observation_features, next_observation_features
    """

    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        observation_features = policy.actor.encoder.get_cached_image_features(observations)
        next_observation_features = policy.actor.encoder.get_cached_image_features(next_observations)

    return observation_features, next_observation_features


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"


def check_nan_in_transition(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    raise_error: bool = False,
) -> bool:
    """
    检查转换数据中的 NaN 值。

    参数:
        observations: 观测张量字典
        actions: 动作张量
        next_state: 下一个状态张量字典
        raise_error: 如果为 True，当检测到 NaN 时抛出 ValueError

    返回:
        bool: 如果检测到 NaN 值则返回 True，否则返回 False
    """
    nan_detected = False

    # 检查观测
    for key, tensor in observations.items():
        if torch.isnan(tensor).any():
            logging.error(f"observations[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in observations[{key}]")

    # 检查下一个状态
    for key, tensor in next_state.items():
        if torch.isnan(tensor).any():
            logging.error(f"next_state[{key}] contains NaN values")
            nan_detected = True
            if raise_error:
                raise ValueError(f"NaN detected in next_state[{key}]")

    # 检查动作
    if torch.isnan(actions).any():
        logging.error("actions contains NaN values")
        nan_detected = True
        if raise_error:
            raise ValueError("NaN detected in actions")

    return nan_detected


def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # 创建一个字典来存储所有状态字典
    state_dicts = {"policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu")}

    # 如果存在离散 critic 则添加
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def process_interaction_message(
    message, interaction_step_shift: int, wandb_logger: WandBLogger | None = None
):
    """使用一致的处理方式处理单个交互消息。"""
    message = bytes_to_python_object(message)
    # 为与检查点状态保持一致，移动交互步骤
    message["Interaction step"] += interaction_step_shift

    # 如果日志记录器可用则记录
    if wandb_logger:
        wandb_logger.log_dict(d=message, mode="train", custom_step_key="Interaction step")

    return message


def process_transitions(
    transition_queue: Queue,
    replay_buffer: ReplayBuffer,
    offline_replay_buffer: ReplayBuffer,
    device: str,
    dataset_repo_id: str | None,
    shutdown_event: any,
):
    """处理队列中所有可用的转换。

    参数:
        transition_queue: 用于从 Actor 接收转换的队列
        replay_buffer: 要添加转换的回放缓冲区
        offline_replay_buffer: 要添加转换的离线回放缓冲区
        device: 要移动转换到的设备
        dataset_repo_id: 数据集的仓库 ID
        shutdown_event: 用于信号关闭的事件
    """
    while not transition_queue.empty() and not shutdown_event.is_set():
        transition_list = transition_queue.get()
        transition_list = bytes_to_transitions(buffer=transition_list)

        for transition in transition_list:
            transition = move_transition_to_device(transition=transition, device=device)

            # 跳过包含 NaN 值的转换
            if check_nan_in_transition(
                observations=transition["state"],
                actions=transition[ACTION],
                next_state=transition["next_state"],
            ):
                logging.warning("[LEARNER] NaN detected in transition, skipping")
                continue

            replay_buffer.add(**transition)

            # 如果是干预则添加到离线缓冲区
            if dataset_repo_id is not None and transition.get("complementary_info", {}).get(
                TeleopEvents.IS_INTERVENTION
            ):
                offline_replay_buffer.add(**transition)


def process_interaction_messages(
    interaction_message_queue: Queue,
    interaction_step_shift: int,
    wandb_logger: WandBLogger | None,
    shutdown_event: any,
) -> dict | None:
    """处理队列中所有可用的交互消息。

    参数:
        interaction_message_queue: 用于接收交互消息的队列
        interaction_step_shift: 要移动交互步骤的量
        wandb_logger: 用于跟踪进度的日志记录器
        shutdown_event: 用于信号关闭的事件

    返回:
        dict | None: 最后一个处理的交互消息，或者如果没有处理则返回 None
    """
    last_message = None
    while not interaction_message_queue.empty() and not shutdown_event.is_set():
        message = interaction_message_queue.get()
        last_message = process_interaction_message(
            message=message,
            interaction_step_shift=interaction_step_shift,
            wandb_logger=wandb_logger,
        )

    return last_message


if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")
