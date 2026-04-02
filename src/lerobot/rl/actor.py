"""
分布式 HILSerl 机器人策略训练的 Actor 服务器运行器。

本脚本实现了分布式 HILSerl 架构的 Actor 组件。
它在机器人环境中执行策略、收集经验，
并将转换发送到 Learner 服务器进行策略更新。

使用示例:

- 为带有人在回路干预的真实机器人训练启动 actor 服务器:
```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**注意**: Actor 服务器需要连接到正在运行的 Learner 服务器。
确保在启动 Actor 之前先启动 Learner 服务器。

**注意**: 人类干预是 HILSerl 训练的关键。按下游戏手柄的右上角扣动按钮
在训练期间控制机器人。初始时频繁干预，然后随着策略改进逐渐减少干预。

**工作流程**:
1. 使用 `lerobot-find-joint-limits` 确定机器人工作空间边界
2. 使用 `gym_manipulator.py` 在录制模式下录制演示
3. 使用 `crop_dataset_roi.py` 处理数据集并确定相机裁剪区域
4. 使用训练配置启动 Learner 服务器
5. 使用相同配置启动此 Actor 服务器
6. 使用人类干预指导策略学习

更多关于完整 HILSerl 训练工作流程的详细信息，请参见:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import time
from functools import lru_cache
from queue import Empty

import grpc
import torch
from torch import nn
from torch.multiprocessing import Event, Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.processor import TransitionKey
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

# 主入口点


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # 创建日志目录，确保目录存在
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # 使用显式日志文件初始化日志记录
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # 如果使用多线程，可以复用通道
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
    )
    logging.info("[ACTOR] Policy process joined")

    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


# 核心算法函数


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    在环境中执行策略交互。

    此函数在环境中展开策略，收集交互数据并将其推送到队列以流式传输到 Learner。
    一旦一个回合完成，从 Learner 接收的更新网络参数将从队列中获取并加载到网络中。

    参数:
        cfg: 交互过程的配置设置。
        shutdown_event: 用于检查进程是否应关闭的事件。
        parameters_queue: 用于从 Learner 接收更新网络参数的队列。
        transitions_queue: 用于向 Learner 发送转换的队列。
        interactions_queue: 用于向 Learner 发送交互信息的队列。
    """
    # 为多进程初始化日志记录
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor 策略进程日志记录已初始化")

    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### 在 Actor 和 Learner 进程中都实例化策略
    ### 为了避免通过端口发送 SACPolicy 对象，我们在两侧都创建策略实例，
    ### Learner 每 n 步发送更新的参数来更新 Actor 的参数
    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # 处理初始观测
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    # 注意: 目前我们只处理单个环境的情况
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # 添加计数器用于计算干预率
    episode_intervention_steps = 0
    episode_total_steps = 0

    policy_timer = TimerManager("Policy inference", log=False)

    for interaction_step in range(cfg.policy.online_steps):
        start_time = time.perf_counter()
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return

        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }

        # 计时策略推理并检查是否满足 FPS 要求
        with policy_timer:
            # 从转换中提取观测用于策略
            action = policy.select_action(batch=observation)
        policy_fps = policy_timer.fps_last

        log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

        # 使用新的 step 函数
        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        # 从处理后的转换中提取值
        next_observation = {
            k: v
            for k, v in new_transition[TransitionKey.OBSERVATION].items()
            if k in cfg.policy.input_features
        }

        # 遥操作动作是在环境中执行的动作
        # 它可能是来自遥操作设备的动作或来自策略的动作
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        sum_reward_episode += float(reward)
        episode_total_steps += 1

        # 从转换信息中检查干预
        intervention_info = new_transition[TransitionKey.INFO]
        if intervention_info.get(TeleopEvents.IS_INTERVENTION, False):
            episode_intervention = True
            episode_intervention_steps += 1

        complementary_info = {
            "discrete_penalty": torch.tensor(
                [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
            ),
        }
        # 为 Learner 创建转换（转换为旧格式）
        list_transition_to_send_to_learner.append(
            Transition(
                state=observation,
                action=executed_action,
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
        )

        # 更新转换用于下一次迭代
        transition = new_transition

        if done or truncated:
            logging.info(f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

            if len(list_transition_to_send_to_learner) > 0:
                push_transitions_to_transport_queue(
                    transitions=list_transition_to_send_to_learner,
                    transitions_queue=transitions_queue,
                )
                list_transition_to_send_to_learner = []

            stats = get_frequency_stats(policy_timer)
            policy_timer.reset()

            # 计算干预率
            intervention_rate = 0.0
            if episode_total_steps > 0:
                intervention_rate = episode_intervention_steps / episode_total_steps

            # 向 Learner 发送回合奖励
            interactions_queue.put(
                python_object_to_bytes(
                    {
                        "Episodic reward": sum_reward_episode,
                        "Interaction step": interaction_step,
                        "Episode intervention": int(episode_intervention),
                        "Intervention rate": intervention_rate,
                        **stats,
                    }
                )
            )

            # 重置干预计数器和环境
            sum_reward_episode = 0.0
            episode_intervention = False
            episode_intervention_steps = 0
            episode_total_steps = 0

            # 重置环境和处理器
            obs, info = online_env.reset()
            env_processor.reset()
            action_processor.reset()

            # 处理初始观测
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        if cfg.env.fps is not None:
            dt_time = time.perf_counter() - start_time
            precise_sleep(max(1 / cfg.env.fps - dt_time, 0.0))


#  通信函数 - 组织所有 gRPC/消息传递函数


def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: Event,  # type: ignore
    attempts: int = 30,
):
    """与 Learner 建立连接。

    参数:
        stub (services_pb2_grpc.LearnerServiceStub): 用于连接的存根。
        shutdown_event (Event): 用于检查是否应建立连接的事件。
        attempts (int): 尝试建立连接的次数。
    返回:
        bool: 如果连接建立成功则返回 True，否则返回 False。
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # 强制连接尝试并检查状态
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    """
    返回 Learner 服务的客户端。

    gRPC 使用 HTTP/2，这是一个二进制协议，并在单个连接上多路复用请求。
    因此我们只需要创建一个客户端并复用它。
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    """从 Learner 接收参数。

    参数:
        cfg (TrainRLServerPipelineConfig): Actor 的配置。
        parameters_queue (Queue): 用于接收参数的队列。
        shutdown_event (Event): 用于检查进程是否应关闭的事件。
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # 创建进程特定的日志文件
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # 使用显式日志文件初始化日志记录
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor 接收策略进程日志记录已初始化")

        # 设置进程信号处理程序来处理关闭信号
        # 但使用来自主进程的关闭事件
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    向 Learner 发送转换。

    此函数持续从队列中获取消息并处理:

    - 转换数据:
        - 收集一批转换（观测、动作、奖励、下一个观测）。
        - 转换被移动到 CPU 并使用 PyTorch 序列化。
        - 序列化的数据被包装在 `services_pb2.Transition` 消息中并发送到 Learner。
    """

    if not use_threads(cfg):
        # 创建进程特定的日志文件
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # 使用显式日志文件初始化日志记录
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor 转换进程日志记录已初始化")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    向 Learner 发送交互信息。

    此函数持续从队列中获取消息并处理:

    - 交互消息:
        - 包含关于回合奖励和策略时序的有用统计信息。
        - 消息使用 `pickle` 序列化并发送到 Learner。
    """

    if not use_threads(cfg):
        # 创建进程特定的日志文件
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # 使用显式日志文件初始化日志记录
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor 交互进程日志记录已初始化")

        # 设置进程信号处理程序来处理关闭信号
        # 但使用来自主进程的关闭事件
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


def transitions_stream(shutdown_event: Event, transitions_queue: Queue, timeout: float) -> services_pb2.Empty:  # type: ignore
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: Event,
    interactions_queue: Queue,
    timeout: float,  # type: ignore
) -> services_pb2.Empty:
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


#  策略函数


def update_policy_parameters(policy: SACPolicy, parameters_queue: Queue, device):
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] 从 Learner 加载新参数。")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # TODO: 检查编码器参数同步可能存在的问题:
        # 1. 当 shared_encoder=True 时，我们从 actor 的 state_dict 加载过时的编码器参数
        #    而不是从 critic 获取更新的编码器参数（critic 是单独优化的）
        # 2. 当 freeze_vision_encoder=True 时，我们浪费带宽发送/加载冻结的参数
        # 3. 需要正确处理 actor 和 discrete_critic 的编码器参数
        # 可能的修复:
        # - 当 shared_encoder=True 时发送 critic 的编码器状态
        # - 当 freeze_vision_encoder=True 时完全跳过编码器参数
        # - 确保 discrete_critic 获取正确的编码器状态（目前使用 encoder_critic）

        # 加载 actor 状态字典
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        policy.actor.load_state_dict(actor_state_dict)

        # 如果存在离散 critic 则加载
        if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
            discrete_critic_state_dict = move_state_dict_to_device(
                state_dicts["discrete_critic"], device=device
            )
            policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
            logging.info("[ACTOR] Loaded discrete critic parameters from Learner.")


#  实用工具函数


def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """将转换分成较小的块发送到 Learner，以避免网络问题。

    参数:
        transitions: 要发送的转换列表
        transitions_queue: 用于向 Learner 发送消息的队列
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        tr = move_transition_to_device(transition=transition, device="cpu")
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)

    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """获取策略的频率统计信息。

    参数:
        timer (TimerManager): 包含收集指标的计时器。

    返回:
        dict[str, float]: 策略的频率统计信息。
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"


if __name__ == "__main__":
    actor_cli()
