"""
经验回放缓冲区模块 (Replay Buffer Module)
==========================================

本模块实现了用于强化学习的经验回放缓冲区。

主要组件:
    - ReplayBuffer: 存储转换的缓冲区类
    - BatchTransition: 批量转换的类型定义
    - random_crop_vectorized: 向量化的随机裁剪
    - random_shift: 随机平移数据增强
"""

import functools
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, REWARD
from lerobot.utils.transition import Transition


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    对一批图像以向量化方式执行每张图像的随机裁剪。
    
    参数:
        images: 输入图像张量 (B, C, H, W)
        output_size: 输出尺寸 (crop_h, crop_w)
    
    返回:
        裁剪后的图像张量 (B, C, crop_h, crop_w)
    """
    B, C, H, W = images.shape  # noqa: N806
    crop_h, crop_w = output_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Requested crop size ({crop_h}, {crop_w}) is bigger than the image size ({H}, {W})."
        )

    tops = torch.randint(0, H - crop_h + 1, (B,), device=images.device)
    lefts = torch.randint(0, W - crop_w + 1, (B,), device=images.device)

    rows = torch.arange(crop_h, device=images.device).unsqueeze(0) + tops.unsqueeze(1)
    cols = torch.arange(crop_w, device=images.device).unsqueeze(0) + lefts.unsqueeze(1)

    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)  # (B, crop_h, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)  # (B, crop_h, crop_w)

    images_hwcn = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    # Gather pixels
    cropped_hwcn = images_hwcn[torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :]
    # cropped_hwcn => (B, crop_h, crop_w, C)

    cropped = cropped_hwcn.permute(0, 3, 1, 2)  # (B, C, crop_h, crop_w)
    return cropped


def random_shift(images: torch.Tensor, pad: int = 4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = images.shape
    images = F.pad(input=images, pad=(pad, pad, pad, pad), mode="replicate")
    return random_crop_vectorized(images=images, output_size=(h, w))


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ):
        """
        Replay buffer for storing transitions.
        It will allocate tensors on the specified device, when the first transition is added.
        NOTE: If you encounter memory issues, you can try to use the `optimize_memory` flag to save memory or
        and use the `storage_device` flag to store the buffer on a different device.
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            device (str): The device where the tensors will be moved when sampling ("cuda:0" or "cpu").
            state_keys (List[str]): The list of keys that appear in `state` and `next_state`.
            image_augmentation_function (Optional[Callable]): A function that takes a batch of images
                and returns a batch of augmented images. If None, a default augmentation function is used.
            use_drq (bool): Whether to use the default DRQ image augmentation style, when sampling in the buffer.
            storage_device: The device (e.g. "cpu" or "cuda:0") where the data will be stored.
                Using "cpu" can help save GPU memory.
            optimize_memory (bool): If True, optimizes memory by not storing duplicate next_states when
                they can be derived from states. This is useful for large datasets where next_state[i] = state[i+1].
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device
        self.position = 0
        self.size = 0
        self.initialized = False
        self.optimize_memory = optimize_memory

        # Track episode boundaries for memory optimization
        self.episode_ends = torch.zeros(capacity, dtype=torch.bool, device=storage_device)

        # If no state_keys provided, default to an empty list
        self.state_keys = state_keys if state_keys is not None else []

        self.image_augmentation_function = image_augmentation_function

        if image_augmentation_function is None:
            base_function = functools.partial(random_shift, pad=4)
            self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq

    def _initialize_storage(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """根据第一个转换初始化存储张量。"""
        # 从第一个转换确定形状
        state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
        action_shape = action.squeeze(0).shape

        # 为存储预分配张量
        self.states = {
            key: torch.empty((self.capacity, *shape), device=self.storage_device)
            for key, shape in state_shapes.items()
        }
        self.actions = torch.empty((self.capacity, *action_shape), device=self.storage_device)
        self.rewards = torch.empty((self.capacity,), device=self.storage_device)

        if not self.optimize_memory:
            # 标准方法: 分别存储 states 和 next_states
            self.next_states = {
                key: torch.empty((self.capacity, *shape), device=self.storage_device)
                for key, shape in state_shapes.items()
            }
        else:
            # 内存优化方法: 不分配 next_states 缓冲区
            # 只创建对 states 的引用以保持 API 一致性
            self.next_states = self.states  # 仅用于 API 一致性的引用

        self.dones = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
        self.truncateds = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)

        # 初始化 complementary_info 的存储
        self.has_complementary_info = complementary_info is not None
        self.complementary_info_keys = []
        self.complementary_info = {}

        if self.has_complementary_info:
            self.complementary_info_keys = list(complementary_info.keys())
            # 为 complementary_info 中的每个键预分配张量
            for key, value in complementary_info.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.squeeze(0).shape
                    self.complementary_info[key] = torch.empty(
                        (self.capacity, *value_shape), device=self.storage_device
                    )
                elif isinstance(value, (int | float)):
                    # 像处理 reward 一样处理标量值
                    self.complementary_info[key] = torch.empty((self.capacity,), device=self.storage_device)
                else:
                    raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

    def __len__(self):
        return self.size

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor],
        done: bool,
        truncated: bool,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """保存一个转换，确保张量存储在指定的存储设备上。"""
        # 如果是第一个转换，初始化存储
        if not self.initialized:
            self._initialize_storage(state=state, action=action, complementary_info=complementary_info)

        # 将转换存储在预分配的张量中
        for key in self.states:
            self.states[key][self.position].copy_(state[key].squeeze(dim=0))

            if not self.optimize_memory:
                # 仅在未优化内存时存储 next_states
                self.next_states[key][self.position].copy_(next_state[key].squeeze(dim=0))

        self.actions[self.position].copy_(action.squeeze(dim=0))
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.truncateds[self.position] = truncated

        # 如果提供了 complementary_info 且存储已初始化，则处理它
        if complementary_info is not None and self.has_complementary_info:
            # 存储 complementary_info
            for key in self.complementary_info_keys:
                if key in complementary_info:
                    value = complementary_info[key]
                    if isinstance(value, torch.Tensor):
                        self.complementary_info[key][self.position].copy_(value.squeeze(dim=0))
                    elif isinstance(value, (int | float)):
                        self.complementary_info[key][self.position] = value

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> BatchTransition:
        """随机采样一批转换并将它们整理成批量张量。"""
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        batch_size = min(batch_size, self.size)
        high = max(0, self.size - 1) if self.optimize_memory and self.size < self.capacity else self.size

        # 用于采样的随机索引 - 在与存储相同的设备上创建
        idx = torch.randint(low=0, high=high, size=(batch_size,), device=self.storage_device)

        # 识别需要增强的图像键
        image_keys = [k for k in self.states if k.startswith(OBS_IMAGE)] if self.use_drq else []

        # 创建批量 state 和 next_state
        batch_state = {}
        batch_next_state = {}

        # 第一遍: 将所有 state 张量加载到目标设备
        for key in self.states:
            batch_state[key] = self.states[key][idx].to(self.device)

            if not self.optimize_memory:
                # 标准方法 - 直接加载 next_states
                batch_next_state[key] = self.next_states[key][idx].to(self.device)
            else:
                # 内存优化方法 - 从下一个索引获取 next_state
                next_idx = (idx + 1) % self.capacity
                batch_next_state[key] = self.states[key][next_idx].to(self.device)

        # 如果需要，以批量方式应用图像增强
        if self.use_drq and image_keys:
            # 连接 state 和 next_state 中的所有图像
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])

            # 优化: 批量处理所有图像并一次性应用增强
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # 将增强后的图像拆分回各自的来源
            for i, key in enumerate(image_keys):
                # 计算当前图像键的偏移量:
                # 对于每个键，我们有 2*batch_size 张图像 (batch_size 个 states，batch_size 个 next_states)
                # States 从索引 i*2*batch_size 开始，占用 batch_size 个槽位
                batch_state[key] = augmented_images[i * 2 * batch_size : (i * 2 + 1) * batch_size]
                # Next states 在 states 之后，从索引 (i*2+1)*batch_size 开始，也占用 batch_size 个槽位
                batch_next_state[key] = augmented_images[(i * 2 + 1) * batch_size : (i + 1) * 2 * batch_size]

        # 采样其他张量
        batch_actions = self.actions[idx].to(self.device)
        batch_rewards = self.rewards[idx].to(self.device)
        batch_dones = self.dones[idx].to(self.device).float()
        batch_truncateds = self.truncateds[idx].to(self.device).float()

        # 如果可用，采样 complementary_info
        batch_complementary_info = None
        if self.has_complementary_info:
            batch_complementary_info = {}
            for key in self.complementary_info_keys:
                batch_complementary_info[key] = self.complementary_info[key][idx].to(self.device)

        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
            truncated=batch_truncateds,
            complementary_info=batch_complementary_info,
        )

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        创建一个生成转换批次的无限迭代器。
        当内部迭代器耗尽时会自动重新开始。

        参数:
            batch_size (int): 要采样的批次大小
            async_prefetch (bool): 是否使用线程异步预取 (默认: True)
            queue_size (int): 预取的批次数量 (默认: 2)

        生成:
            BatchTransition: 批量转换
        """
        while True:  # Create an infinite loop
            if async_prefetch:
                # 获取标准迭代器
                iterator = self._get_async_iterator(queue_size=queue_size, batch_size=batch_size)
            else:
                iterator = self._get_naive_iterator(batch_size=batch_size, queue_size=queue_size)

            # 从迭代器生成所有项
            with suppress(StopIteration):
                yield from iterator

    def _get_async_iterator(self, batch_size: int, queue_size: int = 2):
        """
        创建一个在后台线程中持续生成预取批次的迭代器。
        设计故意简单，避免忙等待/复杂的状态管理。

        参数:
            batch_size (int): 要采样的批次大小
            queue_size (int): 内存中保留的最大预取批次数

        生成:
            BatchTransition: 从回放缓冲区采样的批次
        """
        import queue
        import threading

        data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer() -> None:
            """持续将采样的批次放入队列，直到关闭。"""
            while not shutdown_event.is_set():
                try:
                    batch = self.sample(batch_size)
                    # 超时确保队列满时线程不会阻塞
                    # 并且同时可以设置 shutdown 事件
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    # 队列已满 – 再次循环 (将重新检查 shutdown_event)
                    continue
                except Exception:
                    # 遇到意外错误，终止生产者
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True)
                except Exception:
                    # 如果生产者已经设置了 shutdown 标志，我们退出
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            # 快速清空队列，帮助线程在阻塞于 `put` 时退出
            while not data_queue.empty():
                _ = data_queue.get_nowait()
            # 给生产者线程一点时间完成
            producer_thread.join(timeout=1.0)

    def _get_naive_iterator(self, batch_size: int, queue_size: int = 2):
        """
        创建一个简单的非线程迭代器，生成批次。

        参数:
            batch_size (int): 要采样的批次大小
            queue_size (int): 初始预取的批次数量

        生成:
            BatchTransition: 批量转换
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        capacity: int | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
    ) -> "ReplayBuffer":
        """
        将 LeRobotDataset 转换为 ReplayBuffer。

        参数:
            lerobot_dataset (LeRobotDataset): 要转换的数据集
            device (str): 采样张量的设备。默认 "cuda:0"
            state_keys (Sequence[str] | None): 出现在 `state` 和 `next_state` 中的键列表
            capacity (int | None): 缓冲区容量。如果为 None，则使用数据集长度
            image_augmentation_function (Callable | None): 图像增强函数。
                如果为 None，则使用默认的 pad=4 随机平移
            use_drq (bool): 采样时是否使用 DrQ 图像增强
            storage_device (str): 存储张量数据的设备。使用 "cpu" 可节省 GPU 内存
            optimize_memory (bool): 如果为 True，通过不复制 state 数据来减少内存使用

        返回:
            ReplayBuffer: 包含数据集转换的回放缓冲区
        """
        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        # 使用图像增强和 DrQ 设置创建回放缓冲区
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
        )

        # 将数据集转换为转换
        list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)

        # 用第一个转换初始化缓冲区以设置存储张量
        if list_transition:
            first_transition = list_transition[0]
            first_state = {k: v.to(device) for k, v in first_transition["state"].items()}
            first_action = first_transition[ACTION].to(device)

            # 如果可用，获取 complementary info
            first_complementary_info = None
            if (
                "complementary_info" in first_transition
                and first_transition["complementary_info"] is not None
            ):
                first_complementary_info = {
                    k: v.to(device) for k, v in first_transition["complementary_info"].items()
                }

            replay_buffer._initialize_storage(
                state=first_state, action=first_action, complementary_info=first_complementary_info
            )

        # 用所有转换填充缓冲区
        for data in list_transition:
            for k, v in data.items():
                if isinstance(v, dict):
                    for key, tensor in v.items():
                        v[key] = tensor.to(storage_device)
                elif isinstance(v, torch.Tensor):
                    data[k] = v.to(storage_device)

            action = data[ACTION]

            replay_buffer.add(
                state=data["state"],
                action=action,
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=False,  # NOTE: Truncation are not supported yet in lerobot dataset
                complementary_info=data.get("complementary_info", None),
            )

        return replay_buffer

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        将此 ReplayBuffer 中的所有转换转换为单个 LeRobotDataset 对象。
        """
        if self.size == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # 为数据集创建特征字典
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # 跨 episode 的全局索引
            "episode_index": {"dtype": "int64", "shape": [1]},  # 哪个 episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # episode 内的索引
            "timestamp": {"dtype": "float32", "shape": [1]},  # 目前存储虚拟值
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # 添加 "action"
        sample_action = self.actions[0]
        act_info = guess_feature_info(t=sample_action, name=ACTION)
        features[ACTION] = act_info

        # 添加 "reward" 和 "done"
        features[REWARD] = {"dtype": "float32", "shape": (1,)}
        features[DONE] = {"dtype": "bool", "shape": (1,)}

        # 添加 state 键
        for key in self.states:
            sample_val = self.states[key][0]
            f_info = guess_feature_info(t=sample_val, name=key)
            features[key] = f_info

        # 如果可用，添加 complementary_info 键
        if self.has_complementary_info:
            for key in self.complementary_info_keys:
                sample_val = self.complementary_info[key][0]
                if isinstance(sample_val, torch.Tensor) and sample_val.ndim == 0:
                    sample_val = sample_val.unsqueeze(0)
                f_info = guess_feature_info(t=sample_val, name=f"complementary_info.{key}")
                features[f"complementary_info.{key}"] = f_info

        # 创建一个空的 LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=None,
            features=features,
            use_videos=True,
        )

        # 如果需要，开始写入图像
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=3)

        # 将转换转换为 episodes 和 frames

        for idx in range(self.size):
            actual_idx = (self.position - self.size + idx) % self.capacity

            frame_dict = {}

            # Fill the data for state keys
            for key in self.states:
                frame_dict[key] = self.states[key][actual_idx].cpu()

            # Fill action, reward, done
            frame_dict[ACTION] = self.actions[actual_idx].cpu()
            frame_dict[REWARD] = torch.tensor([self.rewards[actual_idx]], dtype=torch.float32).cpu()
            frame_dict[DONE] = torch.tensor([self.dones[actual_idx]], dtype=torch.bool).cpu()
            frame_dict["task"] = task_name

            # Add complementary_info if available
            if self.has_complementary_info:
                for key in self.complementary_info_keys:
                    val = self.complementary_info[key][actual_idx]
                    # Convert tensors to CPU
                    if isinstance(val, torch.Tensor):
                        if val.ndim == 0:
                            val = val.unsqueeze(0)
                        frame_dict[f"complementary_info.{key}"] = val.cpu()
                    # Non-tensor values can be used directly
                    else:
                        frame_dict[f"complementary_info.{key}"] = val

            # Add to the dataset's buffer
            lerobot_dataset.add_frame(frame_dict)

            # If we reached an episode boundary, call save_episode, reset counters
            if self.dones[actual_idx] or self.truncateds[actual_idx]:
                lerobot_dataset.save_episode()

        # Save any remaining frames in the buffer
        if lerobot_dataset.episode_buffer["size"] > 0:
            lerobot_dataset.save_episode()

        lerobot_dataset.stop_image_writer()
        lerobot_dataset.finalize()

        return lerobot_dataset

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
    ) -> list[Transition]:
        """
        Convert a LeRobotDataset into a list of RL (s, a, r, s', done) transitions.

        Args:
            dataset (LeRobotDataset):
                The dataset to convert. Each item in the dataset is expected to have
                at least the following keys:
                {
                    "action": ...
                    "next.reward": ...
                    "next.done": ...
                    "episode_index": ...
                }
                plus whatever your 'state_keys' specify.

            state_keys (Sequence[str] | None):
                The dataset keys to include in 'state' and 'next_state'. Their names
                will be kept as-is in the output transitions. E.g.
                ["observation.state", "observation.environment_state"].
                If None, you must handle or define default keys.

        Returns:
            transitions (List[Transition]):
                A list of Transition dictionaries with the same length as `dataset`.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        transitions = []
        num_frames = len(dataset)

        # Check if the dataset has "next.done" key
        sample = dataset[0]
        has_done_key = DONE in sample

        # Check for complementary_info keys
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        has_complementary_info = len(complementary_info_keys) > 0

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")

        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample[ACTION].unsqueeze(0)  # Add batch dimension

            # ----- 3) Reward and done -----
            reward = float(current_sample[REWARD].item())  # ensure float

            # Determine done flag - use next.done if available, otherwise infer from episode boundaries
            if has_done_key:
                done = bool(current_sample[DONE].item())  # ensure bool
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            # TODO: (azouitine) Handle truncation (using the same value as done for now)
            truncated = done

            # ----- 4) Next state -----
            # If not done and the next sample is in the same episode, we pull the next sample's state.
            # Otherwise (done=True or next sample crosses to a new episode), next_state = current_state.
            next_state = current_state  # default
            if not done and (i < num_frames - 1):
                next_sample = dataset[i + 1]
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # Build next_state from the same keys
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # Add batch dimension
                    next_state = next_state_data

            # ----- 5) Complementary info (if available) -----
            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    # Strip the "complementary_info." prefix to get the actual key
                    clean_key = key[len("complementary_info.") :]
                    val = current_sample[key]
                    # Handle tensor and non-tensor values differently
                    if isinstance(val, torch.Tensor):
                        complementary_info[clean_key] = val.unsqueeze(0)  # Add batch dimension
                    else:
                        # TODO: (azouitine) Check if it's necessary to convert to tensor
                        # For non-tensor values, use directly
                        complementary_info[clean_key] = val

            # ----- Construct the Transition -----
            transition = Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            transitions.append(transition)

        return transitions


# Utility function to guess shapes/dtypes from a tensor
def guess_feature_info(t, name: str):
    """
    Return a dictionary with the 'dtype' and 'shape' for a given tensor or scalar value.
    If it looks like a 3D (C,H,W) shape, we might consider it an 'image'.
    Otherwise default to appropriate dtype for numeric.
    """

    shape = tuple(t.shape)
    # Basic guess: if we have exactly 3 dims and shape[0] in {1, 3}, guess 'image'
    if len(shape) == 3 and shape[0] in [1, 3]:
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # Otherwise treat as numeric
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition, right_batch_transition: BatchTransition
) -> BatchTransition:
    """
    Concatenates two BatchTransition objects into one.

    This function merges the right BatchTransition into the left one by concatenating
    all corresponding tensors along dimension 0. The operation modifies the left_batch_transitions
    in place and also returns it.

    Args:
        left_batch_transitions (BatchTransition): The first batch to concatenate and the one
            that will be modified in place.
        right_batch_transition (BatchTransition): The second batch to append to the first one.

    Returns:
        BatchTransition: The concatenated batch (same object as left_batch_transitions).

    Warning:
        This function modifies the left_batch_transitions object in place.
    """
    # Concatenate state fields
    left_batch_transitions["state"] = {
        key: torch.cat(
            [left_batch_transitions["state"][key], right_batch_transition["state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    # Concatenate basic fields
    left_batch_transitions[ACTION] = torch.cat(
        [left_batch_transitions[ACTION], right_batch_transition[ACTION]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )

    # Concatenate next_state fields
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # Concatenate done and truncated fields
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]],
        dim=0,
    )

    # Handle complementary_info
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    # Only process if right_info exists
    if right_info is not None:
        # Initialize left complementary_info if needed
        if left_info is None:
            left_batch_transitions["complementary_info"] = right_info
        else:
            # Concatenate each field
            for key in right_info:
                if key in left_info:
                    left_info[key] = torch.cat([left_info[key], right_info[key]], dim=0)
                else:
                    left_info[key] = right_info[key]

    return left_batch_transitions
