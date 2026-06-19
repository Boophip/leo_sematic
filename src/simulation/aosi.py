"""Grid-level semantic age of information primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AosiCellState:
    """One grid cell's semantic value, age, change factor, and final AoSI cost."""

    grid_value: float
    age_s: float
    change_factor: float
    cost: float


def clip01(value: float) -> float:
    """Clamp a semantic value or achievement rate to the normalized range."""

    return min(max(float(value), 0.0), 1.0)


def grid_semantic_value(values: Iterable[float]) -> float:
    """Aggregate ROI values with the project probability-OR formula."""

    product = 1.0
    for value in values:
        product *= 1.0 - clip01(value)
    return 1.0 - product


def semantic_change_factor(
    current_value: float,
    previous_value: float,
    *,
    alpha: float = 0.2,
) -> float:
    """Compute C_g from the current and previous grid semantic values."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    return alpha + (1.0 - alpha) * abs(clip01(current_value) - clip01(previous_value))


def update_aosi_cell(
    *,
    previous_value: float,
    current_value: float,
    previous_age_s: float,
    delta_t_s: float,
    updated: bool,
    alpha: float = 0.2,
) -> AosiCellState:
    """Legacy hard-refresh AoSI update used by smoke tests and diagnostics."""

    if previous_age_s < 0 or delta_t_s < 0:
        raise ValueError("age and delta_t_s must be non-negative")
    value = clip01(current_value)
    age_s = 0.0 if updated else previous_age_s + delta_t_s
    change_factor = semantic_change_factor(value, previous_value, alpha=alpha)
    return AosiCellState(
        grid_value=value,
        age_s=age_s,
        change_factor=change_factor,
        cost=value * change_factor * age_s,
    )


def update_aosi_cell_soft(
    *,
    previous_value: float,
    current_value: float,
    previous_age_s: float,
    delta_t_s: float,
    achievement_rate: float,
    alpha: float = 0.2,
) -> AosiCellState:
    """Update AoSI using the paper's grid-level task-achievement soft reset."""

    if previous_age_s < 0 or delta_t_s < 0:
        raise ValueError("age and delta_t_s must be non-negative")
    rate = clip01(achievement_rate)
    value = clip01(current_value)
    # A perfect grid update still advances by one slot, matching the paper's
    # A_g(t + 1) = A_g(t) * (1 - R_g(t)) + delta_t soft-reset rule.
    age_s = previous_age_s * (1.0 - rate) + delta_t_s
    change_factor = semantic_change_factor(value, previous_value, alpha=alpha)
    return AosiCellState(
        grid_value=value,
        age_s=age_s,
        change_factor=change_factor,
        cost=value * change_factor * age_s,
    )
