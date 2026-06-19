"""Fixed-frequency compute queue and energy primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeNode:
    """Fixed-frequency compute node with a queue measured in CPU cycles."""

    frequency_cycles_per_s: float
    queue_cycles: float = 0.0
    energy_per_cycle_j: float = 1e-9


def validate_compute_node(node: ComputeNode) -> None:
    if node.frequency_cycles_per_s <= 0:
        raise ValueError("frequency_cycles_per_s must be positive")
    if node.queue_cycles < 0:
        raise ValueError("queue_cycles must be non-negative")
    if node.energy_per_cycle_j < 0:
        raise ValueError("energy_per_cycle_j must be non-negative")


def queue_wait_ms(queue_cycles: float, frequency_cycles_per_s: float) -> float:
    """Convert queued cycles into waiting time under fixed-frequency service."""

    if queue_cycles < 0:
        raise ValueError("queue_cycles must be non-negative")
    if frequency_cycles_per_s <= 0:
        raise ValueError("frequency_cycles_per_s must be positive")
    return (queue_cycles / frequency_cycles_per_s) * 1000.0


def cycles_from_inference_ms(inference_ms: float, frequency_cycles_per_s: float) -> float:
    """Convert profiled inference time into cycles for queue accounting."""

    if inference_ms < 0:
        raise ValueError("inference_ms must be non-negative")
    if frequency_cycles_per_s <= 0:
        raise ValueError("frequency_cycles_per_s must be positive")
    return (inference_ms / 1000.0) * frequency_cycles_per_s


def compute_delay_ms(task_cycles: float, node: ComputeNode) -> float:
    """Return queue waiting time plus deterministic service time."""

    validate_compute_node(node)
    if task_cycles < 0:
        raise ValueError("task_cycles must be non-negative")
    service_ms = (task_cycles / node.frequency_cycles_per_s) * 1000.0
    return queue_wait_ms(node.queue_cycles, node.frequency_cycles_per_s) + service_ms


def evolve_compute_queue(
    queue_cycles: float,
    frequency_cycles_per_s: float,
    delta_t_s: float,
    arrival_cycles: float,
) -> float:
    """Update Q_i(t+1) = max(Q_i(t) - f_i * delta_t, 0) + arrivals."""

    if queue_cycles < 0 or arrival_cycles < 0:
        raise ValueError("queue and arrival cycles must be non-negative")
    if frequency_cycles_per_s <= 0:
        raise ValueError("frequency_cycles_per_s must be positive")
    if delta_t_s < 0:
        raise ValueError("delta_t_s must be non-negative")
    served = frequency_cycles_per_s * delta_t_s
    return max(queue_cycles - served, 0.0) + arrival_cycles


def compute_energy_joules(task_cycles: float, node: ComputeNode) -> float:
    """Linear fixed-frequency compute energy for the selected task cycles."""

    validate_compute_node(node)
    if task_cycles < 0:
        raise ValueError("task_cycles must be non-negative")
    return task_cycles * node.energy_per_cycle_j
