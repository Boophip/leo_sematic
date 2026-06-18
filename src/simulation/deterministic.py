"""Composable deterministic task evaluation for pre-RL validation."""

from __future__ import annotations

from dataclasses import dataclass

from src.simulation.compute import (
    ComputeNode,
    compute_delay_ms,
    compute_energy_joules,
    cycles_from_inference_ms,
)
from src.simulation.link import (
    LinkState,
    communication_energy_joules,
    offload_communication_delay_ms,
    transmission_delay_ms,
)


@dataclass(frozen=True)
class TaskCostProfile:
    compressed_bytes: int
    encode_ms: float
    decode_ms: float
    inference_ms: float
    result_bytes: int = 64


@dataclass(frozen=True)
class TaskEvaluation:
    total_delay_ms: float
    communication_delay_ms: float
    queue_wait_ms: float
    compute_delay_ms: float
    compute_energy_j: float
    communication_energy_j: float
    task_cycles: float

    @property
    def total_energy_j(self) -> float:
        return self.compute_energy_j + self.communication_energy_j


def _validate_task_profile(profile: TaskCostProfile) -> None:
    if profile.compressed_bytes < 0 or profile.result_bytes < 0:
        raise ValueError("byte counts must be non-negative")
    if profile.encode_ms < 0 or profile.decode_ms < 0 or profile.inference_ms < 0:
        raise ValueError("timings must be non-negative")


def evaluate_local_task(profile: TaskCostProfile, node: ComputeNode) -> TaskEvaluation:
    _validate_task_profile(profile)
    task_cycles = cycles_from_inference_ms(
        profile.inference_ms,
        node.frequency_cycles_per_s,
    )
    compute_ms = compute_delay_ms(task_cycles, node)
    wait_ms = compute_ms - profile.inference_ms
    return TaskEvaluation(
        total_delay_ms=compute_ms,
        communication_delay_ms=0.0,
        queue_wait_ms=wait_ms,
        compute_delay_ms=compute_ms,
        compute_energy_j=compute_energy_joules(task_cycles, node),
        communication_energy_j=0.0,
        task_cycles=task_cycles,
    )


def evaluate_offload_task(
    profile: TaskCostProfile,
    link: LinkState,
    node: ComputeNode,
    *,
    tx_power_w: float = 5.0,
) -> TaskEvaluation:
    _validate_task_profile(profile)
    task_cycles = cycles_from_inference_ms(
        profile.inference_ms,
        node.frequency_cycles_per_s,
    )
    compute_ms = compute_delay_ms(task_cycles, node)
    wait_ms = compute_ms - profile.inference_ms
    radio_ms = offload_communication_delay_ms(
        profile.compressed_bytes,
        link,
        result_bytes=profile.result_bytes,
    )
    tx_ms = transmission_delay_ms(profile.compressed_bytes, link) + transmission_delay_ms(
        profile.result_bytes,
        link,
    )
    communication_ms = profile.encode_ms + profile.decode_ms + radio_ms
    return TaskEvaluation(
        total_delay_ms=communication_ms + compute_ms,
        communication_delay_ms=communication_ms,
        queue_wait_ms=wait_ms,
        compute_delay_ms=compute_ms,
        compute_energy_j=compute_energy_joules(task_cycles, node),
        communication_energy_j=communication_energy_joules(tx_power_w, tx_ms),
        task_cycles=task_cycles,
    )
