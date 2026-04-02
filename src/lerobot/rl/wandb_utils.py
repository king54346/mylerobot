"""
Weights & Biases 日志工具模块 (WandB Utilities Module)
================================================

本模块提供了与 Weights & Biases 进行日志记录的工具类。

主要组件:
    - WandBLogger: W&B 日志记录器类
    - cfg_to_group: 将配置转换为组名
    - get_wandb_run_id_from_filesystem: 从文件系统获取 W&B 运行 ID
    - get_safe_wandb_artifact_name: 获取安全的工件名称
"""

import logging
import os
import re
from glob import glob
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


def cfg_to_group(
    cfg: TrainPipelineConfig, return_list: bool = False, truncate_tags: bool = False, max_tag_length: int = 64
) -> list[str] | str:
    """返回用于日志记录的组名。可选择以列表形式返回组名。"""

    def _maybe_truncate(tag: str) -> str:
        """如果需要，将标签截断为 max_tag_length 个字符。

        wandb 拒绝长度超过 64 个字符的标签。
        参见: https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py
        """
        if len(tag) <= max_tag_length:
            return tag
        return tag[:max_tag_length]

    lst = [
        f"policy:{cfg.policy.type}",
        f"seed:{cfg.seed}",
    ]
    if cfg.dataset is not None:
        lst.append(f"dataset:{cfg.dataset.repo_id}")
    if cfg.env is not None:
        lst.append(f"env:{cfg.env.type}")
    if truncate_tags:
        lst = [_maybe_truncate(tag) for tag in lst]
    return lst if return_list else "-".join(lst)


def get_wandb_run_id_from_filesystem(log_dir: Path) -> str:
    # 获取 WandB 运行 ID。
    paths = glob(str(log_dir / "wandb/latest-run/run-*"))
    if len(paths) != 1:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    match = re.search(r"run-([^\.]+).wandb", paths[0].split("/")[-1])
    if match is None:
        raise RuntimeError("Couldn't get the previous WandB run ID for run resumption.")
    wandb_run_id = match.groups(0)[0]
    return wandb_run_id


def get_safe_wandb_artifact_name(name: str):
    """WandB 工件名称不接受 ":" 或 "/"。"""
    return name.replace(":", "_").replace("/", "_")


class WandBLogger:
    """用于使用 wandb 记录对象的辅助类。"""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.wandb
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        self._group = cfg_to_group(cfg)

        # 设置 WandB。
        os.environ["WANDB_SILENT"] = "True"
        import wandb

        wandb_run_id = (
            cfg.wandb.run_id
            if cfg.wandb.run_id
            else get_wandb_run_id_from_filesystem(self.log_dir)
            if cfg.resume
            else None
        )
        wandb.init(
            id=wandb_run_id,
            project=self.cfg.project,
            entity=self.cfg.entity,
            name=self.job_name,
            notes=self.cfg.notes,
            tags=cfg_to_group(cfg, return_list=True, truncate_tags=True),
            dir=self.log_dir,
            config=cfg.to_dict(),
            # TODO(rcadene): 尝试设置为 True
            save_code=False,
            # TODO(rcadene): 分离 train 和 eval，并使用 job_type="eval" 运行异步 eval
            job_type="train_eval",
            resume="must" if cfg.resume else None,
            mode=self.cfg.mode if self.cfg.mode in ["online", "offline", "disabled"] else "online",
        )
        run_id = wandb.run.id
        # 注意: 我们将用 wandb run id 覆盖 cfg.wandb.run_id。
        # 这是因为我们希望能够从 wandb run id 恢复运行。
        cfg.wandb.run_id = run_id
        # 处理 rl 异步训练的自定义步骤键。
        self._wandb_custom_step_key: set[str] | None = None
        logging.info(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
        self._wandb = wandb

    def log_policy(self, checkpoint_dir: Path):
        """将策略检查点到 wandb。"""
        if self.cfg.disable_artifact:
            return

        step_id = checkpoint_dir.name
        artifact_name = f"{self._group}-{step_id}"
        artifact_name = get_safe_wandb_artifact_name(artifact_name)
        artifact = self._wandb.Artifact(artifact_name, type="model")
        pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

        # 检查这是否是 PEFT 模型（有 adapter 文件而不是 model.safetensors）
        adapter_model_file = pretrained_model_dir / "adapter_model.safetensors"
        standard_model_file = pretrained_model_dir / SAFETENSORS_SINGLE_FILE

        if adapter_model_file.exists():
            # PEFT 模型: 添加 adapter 文件和配置
            artifact.add_file(adapter_model_file)
            adapter_config_file = pretrained_model_dir / "adapter_config.json"
            if adapter_config_file.exists():
                artifact.add_file(adapter_config_file)
            # 同时添加加载所需的策略配置
            config_file = pretrained_model_dir / "config.json"
            if config_file.exists():
                artifact.add_file(config_file)
        elif standard_model_file.exists():
            # 标准模型: 添加单个 safetensors 文件
            artifact.add_file(standard_model_file)
        else:
            logging.warning(
                f"No {SAFETENSORS_SINGLE_FILE} or adapter_model.safetensors found in {pretrained_model_dir}. "
                "Skipping model artifact upload to WandB."
            )
            return

        self._wandb.log_artifact(artifact)

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # 注意: 这不简单。Wandb 步骤必须始终单调递增，并且每次 wandb.log 调用都会增加，
        # 但在异步 RL 的情况下，可能有多个时间步。例如，与环境的交互步、
        # 训练步、评估步等。因此我们需要定义自定义步骤键
        # 来为每个指标记录正确的步骤。
        if custom_step_key is not None:
            if self._wandb_custom_step_key is None:
                self._wandb_custom_step_key = set()
            new_custom_key = f"{mode}/{custom_step_key}"
            if new_custom_key not in self._wandb_custom_step_key:
                self._wandb_custom_step_key.add(new_custom_key)
                self._wandb.define_metric(new_custom_key, hidden=True)

        for k, v in d.items():
            if not isinstance(v, (int | float | str)):
                logging.warning(
                    f'键 "{k}" 的 WandB 日志记录被忽略，因为其类型 "{type(v)}" 未被此包装器处理。'
                )
                continue

            # 不记录自定义步骤键本身。
            if self._wandb_custom_step_key is not None and k in self._wandb_custom_step_key:
                continue

            if custom_step_key is not None:
                value_custom_step = d[custom_step_key]
                data = {f"{mode}/{k}": v, f"{mode}/{custom_step_key}": value_custom_step}
                self._wandb.log(data)
                continue

            self._wandb.log(data={f"{mode}/{k}": v}, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        wandb_video = self._wandb.Video(video_path, fps=self.env_fps, format="mp4")
        self._wandb.log({f"{mode}/video": wandb_video}, step=step)
