"""
LeRobot 实时分块 (RTC) 实现。

RTC 建模模块
======================

基于 Physical Intelligence 的 Kinetix 实现:
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L214

本模块实现了 RTC 技术，包括:
- 速度计算 (velocity calculation)
- 前缀注意力 (prefix attention)
- 自适应块处理 (adaptive chunk processing)
- 引导校正 (guidance correction)
"""

import logging
import math

import torch
from torch import Tensor

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.debug_tracker import Tracker

logger = logging.getLogger(__name__)


class RTCProcessor:
    """动作分块策略的实时分块处理器。

    该类实现了 RTC 技术，包括速度计算、前缀注意力和自适应块处理ing.
    """

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config

        self.tracker = None

        if rtc_config.debug:
            self.tracker = Tracker(
                enabled=rtc_config.debug,
                maxlen=rtc_config.debug_maxlen,
            )

    # ====================== 追踪器代理方法 ======================
    def track(
        self,
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        """追踪调试信息的代理方法。

        如果追踪器为 None 或已禁用，此方法不执行任何操作。
        否则，将调用转发给 tracker.track()。
        """
        if self.tracker is not None:
            self.tracker.track(
                time=time,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                **metadata,
            )

    def get_all_debug_steps(self) -> list:
        """从追踪器获取所有调试步骤。

        如果追踪器已禁用或为 None，返回空列表。
        """
        if self.tracker is not None:
            return self.tracker.get_all_steps()
        return []

    def is_debug_enabled(self) -> bool:
        """检查调试追踪是否已启用。

        如果追踪器存在且已启用，返回 True。
        """
        return self.tracker is not None and self.tracker.enabled

    def reset_tracker(self) -> None:
        """重置追踪器，清除所有记录的步骤。

        如果追踪器为 None，则不执行任何操作。
        """
        if self.tracker is not None:
            self.tracker.reset()

    # ====================== 追踪器代理方法结束 ======================

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ) -> Tensor:
        """现有去噪器的 RTC 引导包装器。

        此方法包装一个原始去噪可调用对象，该对象仅接受 ``x_t`` 并返回基础去噪速度 ``v_t``。
        然后使用前一块的剩余前缀应用实时分块 (RTC) 前缀引导。

        参数:
            x_t (Tensor): 待去噪的当前潜在状态。形状为 ``(B, T, A)`` 或 ``(T, A)``。
            prev_chunk_left_over (Tensor | None): 前一块未执行的前缀。
                形状为 ``(B, T_prev, A)`` 或 ``(T_prev, A)``。如果为 ``None``，
                则不应用引导，方法返回原始去噪器的 ``v_t``。
            inference_delay (int): 用于引导的前缀时间步数。
            time (float | Tensor): [0, 1] 范围内表示归一化时间的标量。
                必须可以与 ``x_t`` 广播。
            original_denoise_step_partial (Callable[[Tensor], Tensor]): 仅给定 ``x_t``
                计算基础去噪速度的可调用对象。
            execution_horizon (int | None): 用于构建前缀权重的视界。
                如果为 ``None``，默认使用 ``self.rtc_config.execution_horizon``。

        返回:
            Tensor: 与 ``v_t`` 形状相同的引导速度。

        注意:
            - 如果输入是 2D，会临时添加批次维度，最后再移除。
            - 如果 ``prev_chunk_left_over`` 比当前块长度 ``T`` 短，
              会用零右填充以匹配 ``T``。
            - 前缀权重通过 ``get_prefix_weights(inference_delay, execution_horizon, T)``
              构建并广播到 ``(B, T, A)``。
            - 引导校正通过 autograd 计算，使用 ``x1_t = x_t + time * v_t`` 和
              ``error = (prev_chunk_left_over - x1_t) * weights``。
            - 最终引导权重由配置中的 ``max_guidance_weight`` 限制。

        参考:
            https://www.physicalintelligence.company/download/real_time_chunking.pdf
        """

        # 在原始实现中，时间从 0 到 1
        # 在我们的实现中，时间从 1 到 0
        # 所以需要反转时间
        tau = 1 - time

        if prev_chunk_left_over is None:
            # 第一步，无引导 - 返回 v_t
            v_t = original_denoise_step_partial(x_t)
            return v_t

        x_t = x_t.clone().detach()

        squeezed = False
        if len(x_t.shape) < 3:
            # 添加批次维度
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if len(prev_chunk_left_over.shape) < 3:
            # 添加批次维度
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        # If the previous action chunk is to short then it doesn't make sense to use long execution horizon
        # because there is nothing to merge
        if execution_horizon > prev_chunk_left_over.shape[1]:
            execution_horizon = prev_chunk_left_over.shape[1]

        batch_size = x_t.shape[0]
        action_chunk_size = x_t.shape[1]
        action_dim = x_t.shape[2]

        if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            padded = torch.zeros(batch_size, action_chunk_size, action_dim).to(x_t.device)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        assert prev_chunk_left_over.shape == x_t.shape, (
            "The padded previous chunk must be the same size as the input tensor"
        )

        weights = (
            self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
            .to(x_t.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t.requires_grad_(True)

            x1_t = x_t - time * v_t  # noqa: N806
            err = (prev_chunk_left_over - x1_t) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)
        tau_tensor = torch.as_tensor(tau)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction

        # 如果添加了批次维度则移除
        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)

        self.track(
            time=time,
            x1_t=x1_t,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )

        return result

    def get_prefix_weights(self, start, end, total):
        start = min(start, end)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR:
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP:
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)

        return weights

    def _linweights(self, start, end, total):
        skip_steps_at_end = max(total - end, 0)

        linspace_steps = total - skip_steps_at_end - start

        if end <= start or linspace_steps <= 0:
            return torch.tensor([])

        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total, end):
        zeros_len = total - end

        if zeros_len <= 0:
            return weights

        zeros = torch.zeros(zeros_len)
        return torch.cat([weights, zeros])

    def _add_leading_ones(self, weights, start, total):
        ones_len = min(start, total)

        if ones_len <= 0:
            return weights

        ones = torch.ones(ones_len)
        return torch.cat([ones, weights])
