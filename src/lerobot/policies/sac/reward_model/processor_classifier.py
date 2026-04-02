# !/usr/bin/env python


from typing import Any

import torch

from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.processor import (
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action


def make_classifier_processor(
    config: RewardClassifierConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the reward classifier.

    The pre-processing pipeline prepares input data for the classifier by:
    1. Normalizing both input and output features based on dataset statistics.
    2. Moving the data to the specified device.

    The post-processing pipeline handles the classifier's output by:
    1. Moving the data to the CPU.
    2. Applying an identity step, as no unnormalization is needed for the output logits.

    Args:
        config: The configuration object for the RewardClassifier.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        NormalizerProcessorStep(
            features=config.input_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        NormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [DeviceProcessorStep(device="cpu"), IdentityProcessorStep()]

    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name="classifier_preprocessor",
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name="classifier_postprocessor",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
