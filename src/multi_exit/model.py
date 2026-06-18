"""Small multi-exit CNN used to validate the stage-two pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TimedExitOutput:
    exit_level: int
    logits: torch.Tensor
    cumulative_ms: float


def _conv_stage(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class TinyMultiExitCNN(nn.Module):
    """Four-exit CNN baseline; each head receives raw logits."""

    def __init__(self, num_classes: int, *, input_channels: int = 3) -> None:
        super().__init__()
        channels = (16, 32, 64, 96)
        self.stages = nn.ModuleList(
            [
                _conv_stage(input_channels, channels[0]),
                _conv_stage(channels[0], channels[1]),
                _conv_stage(channels[1], channels[2]),
                _conv_stage(channels[2], channels[3]),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.exit_heads = nn.ModuleList(
            [nn.Linear(channel_count, num_classes) for channel_count in channels]
        )

    def _exit_logits(self, features: torch.Tensor, exit_index: int) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.exit_heads[exit_index](pooled)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        logits: list[torch.Tensor] = []
        features = inputs
        for exit_index, stage in enumerate(self.stages):
            features = stage(features)
            logits.append(self._exit_logits(features, exit_index))
        return logits

    def forward_with_exit_times(self, inputs: torch.Tensor) -> list[TimedExitOutput]:
        """Return cumulative per-exit timings for profiling smoke runs."""

        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)
        start = time.perf_counter()
        outputs: list[TimedExitOutput] = []
        features = inputs
        for exit_index, stage in enumerate(self.stages):
            features = stage(features)
            logits = self._exit_logits(features, exit_index)
            if inputs.is_cuda:
                torch.cuda.synchronize(inputs.device)
            cumulative_ms = (time.perf_counter() - start) * 1000.0
            outputs.append(
                TimedExitOutput(
                    exit_level=exit_index + 1,
                    logits=logits,
                    cumulative_ms=cumulative_ms,
                )
            )
        return outputs


def multi_exit_cross_entropy(
    exit_logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    *,
    exit_weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Average CrossEntropyLoss over exits using raw logits, not softmax."""

    if not exit_logits:
        raise ValueError("exit_logits must contain at least one tensor")
    if exit_weights is None:
        exit_weights = [1.0] * len(exit_logits)
    if len(exit_weights) != len(exit_logits):
        raise ValueError("exit_weights length must match exit_logits length")

    weighted_losses = [
        float(weight) * F.cross_entropy(logits, targets)
        for weight, logits in zip(exit_weights, exit_logits)
    ]
    return sum(weighted_losses) / max(sum(float(weight) for weight in exit_weights), 1e-12)
