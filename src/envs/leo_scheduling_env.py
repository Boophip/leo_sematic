"""Gymnasium environment for single-agent LEO ROI scheduling.

This module is the RL environment layer. It reuses the deterministic
simulation primitives and offline profiling/proxy tables; it does not run
vision models online.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.simulation.aosi import grid_semantic_value, update_aosi_cell_soft
from src.simulation.compute import ComputeNode, evolve_compute_queue
from src.simulation.deterministic import (
    TaskEvaluation,
    evaluate_local_task,
    evaluate_offload_task,
)
from src.simulation.episode import (
    DROP_ACTION,
    LOCAL_ACTION,
    OFFLOAD_ACTION,
    SOURCE_NODE,
    ActionProfile,
    ActionProfileTable,
    DecisionResult,
    NodeRuntimeState,
    PolicyRunResult,
    SatelliteNodeConfig,
    SchedulingAction,
    SimulationConfig,
    SlotSummary,
    build_slot_summary,
    summarize_decisions,
)
from src.simulation.link import (
    IllegalActionError,
    LinkState,
    offload_communication_delay_ms,
    spectral_efficiency_for_snr,
)


EXIT_LEVELS = (1, 2, 3, 4)
OFFLOAD_COMPRESSION_LEVELS = ("beta_0", "beta_1", "beta_2", "beta_3")
ACTION_FEATURES = ("legal", "quality", "delay", "bytes")
EMPTY_EVALUATION = TaskEvaluation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class FixedActionSpec:
    """Stable discrete action slot exposed to PPO."""

    kind: str
    target_node: str = ""
    exit_level: int = 0
    compression_level: str = ""

    @property
    def label(self) -> str:
        return self.to_action().label

    def to_action(self) -> SchedulingAction:
        return SchedulingAction(
            kind=self.kind,
            target_node=self.target_node,
            exit_level=self.exit_level,
            compression_level=self.compression_level,
        )


def build_fixed_action_space(target_nodes: Sequence[str]) -> tuple[FixedActionSpec, ...]:
    """Build drop, local, and offload actions with a stable index order."""

    actions: list[FixedActionSpec] = [FixedActionSpec(kind=DROP_ACTION)]
    for exit_level in EXIT_LEVELS:
        actions.append(
            FixedActionSpec(
                kind=LOCAL_ACTION,
                target_node=SOURCE_NODE,
                exit_level=exit_level,
                compression_level="local",
            )
        )
    for target_node in sorted(target_nodes):
        for exit_level in EXIT_LEVELS:
            for compression_level in OFFLOAD_COMPRESSION_LEVELS:
                actions.append(
                    FixedActionSpec(
                        kind=OFFLOAD_ACTION,
                        target_node=target_node,
                        exit_level=exit_level,
                        compression_level=compression_level,
                    )
                )
    return tuple(actions)


class LeoSchedulingEnv(gym.Env):
    """Single-agent PPO environment over one deterministic ROI stream."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        profiles: ActionProfileTable,
        *,
        config: SimulationConfig,
        node_configs: Sequence[SatelliteNodeConfig],
        link_trace: Mapping[int, Mapping[str, LinkState]],
        target_nodes: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if config.rois_per_slot <= 0:
            raise ValueError("rois_per_slot must be positive")
        if not profiles.roi_order:
            raise ValueError("profiles must contain at least one ROI")

        self.profiles = profiles
        self.config = config
        self.node_configs = tuple(node_configs)
        self.link_trace = {int(slot): dict(links) for slot, links in link_trace.items()}
        if SOURCE_NODE not in {node.name for node in self.node_configs}:
            raise ValueError(f"node_configs must include {SOURCE_NODE!r}")

        inferred_targets = [
            node.name
            for node in self.node_configs
            if node.name != SOURCE_NODE
        ]
        self.target_nodes = tuple(target_nodes or inferred_targets)
        self.action_specs = build_fixed_action_space(self.target_nodes)
        self.action_space = spaces.Discrete(len(self.action_specs))
        self.observation_feature_names = self._build_feature_names()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )

        self.node_states: dict[str, NodeRuntimeState] = {}
        self.pending_cycles: dict[str, float] = {}
        self.grid_ages: dict[int, float] = {}
        self.previous_grid_values: dict[int, float] = {}
        self.current_index = 0
        self.current_slot_index = 0
        self._terminated = False
        self._slot_total_value: dict[int, float] = {}
        self._slot_success_value: dict[int, float] = {}
        self.training_reward_total = 0.0
        self.quality_virtual_queue = 0.0
        self.delay_virtual_queue_ms = 0.0
        self.decisions: list[DecisionResult] = []
        self.slots: list[SlotSummary] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Reset queue, AoSI, and ROI pointer while preserving the fixed scenario."""

        super().reset(seed=seed)
        self.node_states = {
            node.name: NodeRuntimeState(config=node, queue_cycles=node.initial_queue_cycles)
            for node in self.node_configs
        }
        self.pending_cycles = {name: 0.0 for name in self.node_states}
        self.grid_ages = {}
        self.previous_grid_values = {}
        self.current_index = 0
        self.current_slot_index = 0
        self._terminated = False
        self._slot_total_value = {}
        self._slot_success_value = {}
        self.training_reward_total = 0.0
        self.quality_virtual_queue = 0.0
        self.delay_virtual_queue_ms = 0.0
        self.decisions = []
        self.slots = []
        return self._observation(), {"roi_id": self._current_roi_id()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Execute one atomic ROI action and advance to the next ROI."""

        if self._terminated:
            raise RuntimeError("step called after episode termination; call reset first")
        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError(f"action index out of range: {action_index}")

        slot_index = self._slot_index_for_current_roi()
        self.current_slot_index = slot_index
        roi_id = self._current_roi_id()
        roi = self.profiles.roi(roi_id)
        links = self._links_for_slot(slot_index)
        spec = self.action_specs[action_index]
        decision = self._evaluate_action(spec.to_action(), roi, slot_index, links)
        self.decisions.append(decision)
        self._slot_total_value[roi.grid_id] = (
            self._slot_total_value.get(roi.grid_id, 0.0) + roi.semantic_value
        )
        if decision.success:
            self._slot_success_value[roi.grid_id] = (
                self._slot_success_value.get(roi.grid_id, 0.0) + roi.semantic_value
            )

        reward = self._decision_reward(decision)
        self.current_index += 1
        slot_ended = self._slot_completed(slot_index)
        aosi_cost = 0.0
        canonical_slot_reward = 0.0
        if slot_ended:
            slot_summary = self._finish_slot(slot_index)
            aosi_cost = slot_summary.aosi_cost
            canonical_slot_reward = slot_summary.reward
            reward += canonical_slot_reward

        self._terminated = self.current_index >= len(self.profiles.roi_order)
        self.training_reward_total += reward
        info = decision.as_row()
        info.update(
            {
                "reward": reward,
                "slot_ended": slot_ended,
                "aosi_cost": aosi_cost,
                "canonical_slot_reward": canonical_slot_reward,
                "training_reward_total": self.training_reward_total,
                "quality_virtual_queue": self.quality_virtual_queue,
                "delay_virtual_queue_ms": self.delay_virtual_queue_ms,
                "action_index": action_index,
                "action_label": spec.label,
            }
        )
        return self._observation(), float(reward), self._terminated, False, info

    def action_index(
        self,
        *,
        kind: str,
        target_node: str = "",
        exit_level: int = 0,
        compression_level: str = "",
    ) -> int:
        """Return the stable action index for tests and scripted evaluation."""

        target = SOURCE_NODE if kind == LOCAL_ACTION and not target_node else target_node
        for index, spec in enumerate(self.action_specs):
            if (
                spec.kind == kind
                and spec.target_node == target
                and spec.exit_level == exit_level
                and spec.compression_level == compression_level
            ):
                return index
        raise KeyError((kind, target_node, exit_level, compression_level))

    def metrics(self, *, policy_name: str = "Proposed-RL") -> dict[str, object]:
        """Aggregate decisions using the same report contract as deterministic baselines."""

        metrics = summarize_decisions(policy_name, self.decisions, self.slots)
        metrics["quality_virtual_queue"] = self.quality_virtual_queue
        metrics["delay_virtual_queue_ms"] = self.delay_virtual_queue_ms
        return metrics

    def as_policy_run_result(self, *, policy_name: str = "Proposed-RL") -> PolicyRunResult:
        """Return a policy result object compatible with comparison writers."""

        return PolicyRunResult(
            policy=policy_name,
            decisions=list(self.decisions),
            slots=list(self.slots),
            metrics=self.metrics(policy_name=policy_name),
        )

    def _evaluate_action(
        self,
        action: SchedulingAction,
        roi,
        slot_index: int,
        links: Mapping[str, LinkState],
    ) -> DecisionResult:
        if action.kind == DROP_ACTION:
            return self._decision_from_evaluation(
                slot_index=slot_index,
                roi=roi,
                action=action,
                profile=None,
                evaluation=EMPTY_EVALUATION,
                illegal=False,
            )

        profile = self._profile_for_action(roi.roi_id, action)
        if profile is None or not self._is_action_legal(action, profile, links):
            return self._decision_from_evaluation(
                slot_index=slot_index,
                roi=roi,
                action=action,
                profile=profile,
                evaluation=EMPTY_EVALUATION,
                illegal=True,
            )

        target_node = SOURCE_NODE if action.kind == LOCAL_ACTION else action.target_node
        node = self._compute_node_with_pending(target_node)
        try:
            if action.kind == LOCAL_ACTION:
                evaluation = evaluate_local_task(profile.to_task_profile(result_bytes=self.config.result_bytes), node)
            else:
                evaluation = evaluate_offload_task(
                    profile.to_task_profile(result_bytes=self.config.result_bytes),
                    links[action.target_node],
                    node,
                    tx_power_w=self.config.tx_power_w,
                )
        except (IllegalActionError, KeyError):
            return self._decision_from_evaluation(
                slot_index=slot_index,
                roi=roi,
                action=action,
                profile=profile,
                evaluation=EMPTY_EVALUATION,
                illegal=True,
            )

        self.pending_cycles[target_node] += evaluation.task_cycles
        return self._decision_from_evaluation(
            slot_index=slot_index,
            roi=roi,
            action=action,
            profile=profile,
            evaluation=evaluation,
            illegal=False,
        )

    def _decision_from_evaluation(
        self,
        *,
        slot_index: int,
        roi,
        action: SchedulingAction,
        profile: ActionProfile | None,
        evaluation: TaskEvaluation,
        illegal: bool,
    ) -> DecisionResult:
        predicted_quality = 0.0 if profile is None or illegal else profile.predicted_quality
        true_quality = 0.0 if profile is None or illegal else profile.task_quality
        timeout = (
            action.kind != DROP_ACTION
            and not illegal
            and evaluation.total_delay_ms > self.config.deadline_ms
        )
        quality_violation = (
            action.kind != DROP_ACTION
            and not illegal
            and predicted_quality < self.config.quality_threshold
        )
        success = action.kind != DROP_ACTION and not illegal and not timeout and not quality_violation
        return DecisionResult(
            policy="Proposed-RL",
            slot_index=slot_index,
            roi_id=roi.roi_id,
            image_id=roi.image_id,
            grid_id=roi.grid_id,
            semantic_value=roi.semantic_value,
            action_label=action.label,
            action_kind=action.kind,
            target_node=SOURCE_NODE if action.kind == LOCAL_ACTION else action.target_node,
            exit_level=action.exit_level,
            compression_level=action.compression_level,
            predicted_quality=predicted_quality,
            true_task_quality=true_quality,
            total_delay_ms=evaluation.total_delay_ms,
            communication_delay_ms=evaluation.communication_delay_ms,
            queue_wait_ms=evaluation.queue_wait_ms,
            compute_delay_ms=evaluation.compute_delay_ms,
            total_energy_j=evaluation.total_energy_j,
            communication_energy_j=evaluation.communication_energy_j,
            compute_energy_j=evaluation.compute_energy_j,
            compressed_bytes=0 if profile is None else profile.compressed_bytes,
            task_cycles=evaluation.task_cycles,
            success=success,
            timeout=timeout,
            quality_violation=quality_violation,
            illegal=illegal,
        )

    def _decision_reward(self, decision: DecisionResult) -> float:
        reward = 0.0
        if decision.action_kind != DROP_ACTION and not decision.illegal:
            reward += decision.semantic_value * decision.predicted_quality
            reward -= self.config.delay_weight * _safe_div(
                decision.total_delay_ms,
                self.config.delay_ref_ms,
            )
            reward -= self.config.energy_weight * _safe_div(
                decision.total_energy_j,
                self.config.energy_ref_j,
            )
        if decision.illegal:
            reward -= self.config.illegal_penalty
        if decision.timeout:
            reward -= self.config.timeout_penalty
        if decision.quality_violation:
            reward -= self.config.quality_penalty
        return reward

    def _finish_slot(self, slot_index: int) -> SlotSummary:
        for name, state in self.node_states.items():
            state.queue_cycles = evolve_compute_queue(
                queue_cycles=state.queue_cycles,
                frequency_cycles_per_s=state.config.frequency_cycles_per_s,
                delta_t_s=self.config.delta_t_s,
                arrival_cycles=self.pending_cycles[name],
            )
        self.pending_cycles = {name: 0.0 for name in self.node_states}

        current_values = self._grid_values_for_slot(slot_index)
        aosi_cost = self._update_aosi(current_values)
        previous_keys = set(self.previous_grid_values) | set(current_values)
        self.previous_grid_values = {
            grid_id: current_values.get(grid_id, 0.0)
            for grid_id in previous_keys
        }
        slot_decisions = [
            row
            for row in self.decisions
            if row.slot_index == slot_index
        ]
        self._update_virtual_queues(slot_decisions)
        slot_summary = self._slot_summary(slot_index, slot_decisions, aosi_cost)
        self.slots.append(slot_summary)
        self._slot_total_value = {}
        self._slot_success_value = {}
        return slot_summary

    def _update_virtual_queues(self, decisions: Sequence[DecisionResult]) -> None:
        quality_deficit = sum(
            self.config.quality_threshold - row.predicted_quality
            for row in decisions
        )
        delay_excess = sum(
            max(row.total_delay_ms - self.config.deadline_ms, 0.0)
            for row in decisions
        )
        self.quality_virtual_queue = max(self.quality_virtual_queue + quality_deficit, 0.0)
        self.delay_virtual_queue_ms = max(self.delay_virtual_queue_ms + delay_excess, 0.0)

    def _update_aosi(self, current_values: Mapping[int, float]) -> float:
        total_cost = 0.0
        active_grids = set(self.previous_grid_values) | set(current_values)
        for grid_id in active_grids:
            total_value = self._slot_total_value.get(grid_id, 0.0)
            achievement_rate = _safe_div(self._slot_success_value.get(grid_id, 0.0), total_value)
            cell = update_aosi_cell_soft(
                previous_value=self.previous_grid_values.get(grid_id, 0.0),
                current_value=current_values.get(grid_id, 0.0),
                previous_age_s=self.grid_ages.get(grid_id, 0.0),
                delta_t_s=self.config.delta_t_s,
                achievement_rate=achievement_rate,
                alpha=self.config.aosi_alpha,
            )
            self.grid_ages[grid_id] = cell.age_s
            total_cost += cell.cost
        return total_cost

    def _slot_summary(
        self,
        slot_index: int,
        decisions: Sequence[DecisionResult],
        aosi_cost: float,
    ) -> SlotSummary:
        return build_slot_summary(
            "Proposed-RL",
            slot_index,
            decisions,
            aosi_cost=aosi_cost,
            config=self.config,
        )

    def _observation(self) -> np.ndarray:
        if self.current_index >= len(self.profiles.roi_order):
            return np.zeros(len(self.observation_feature_names), dtype=np.float32)
        roi = self.profiles.roi(self._current_roi_id())
        slot_index = self._slot_index_for_current_roi()
        links = self._links_for_slot(slot_index)
        grid_value = self._grid_values_for_slot(slot_index).get(roi.grid_id, 0.0)
        values: list[float] = [
            _norm(roi.class_id, 14.0),
            _clip01(roi.detector_confidence),
            _norm(roi.area_ratio, 0.1),
            _clip01(roi.semantic_value),
            _clip01(roi.class_priority),
            _norm(roi.grid_id, 63.0),
            _safe_div(self.current_index, max(len(self.profiles.roi_order) - 1, 1)),
            _safe_div(self.current_index % self.config.rois_per_slot, self.config.rois_per_slot),
            _norm(self.grid_ages.get(roi.grid_id, 0.0), 64.0),
            _clip01(grid_value),
            _norm(
                self.quality_virtual_queue,
                max(self.config.rois_per_slot * max(self.config.quality_threshold, 1.0), 1.0),
            ),
            _norm(
                self.delay_virtual_queue_ms,
                max(self.config.rois_per_slot * max(self.config.deadline_ms, 1.0), 1.0),
            ),
        ]
        for node_name in self._node_feature_order():
            queue_cycles = self.node_states.get(node_name)
            values.append(
                _norm(0.0 if queue_cycles is None else queue_cycles.queue_cycles, 10.0e9)
            )
        for target_node in self.target_nodes:
            link = links.get(target_node, LinkState(False, -1.0, 1.0))
            spectral_efficiency = spectral_efficiency_for_snr(link.snr_db) if link.visible else 0.0
            values.extend(
                [
                    1.0 if link.visible else 0.0,
                    _norm(max(link.snr_db, 0.0), 30.0),
                    _norm(spectral_efficiency, 6.0),
                    _norm(link.propagation_delay_ms, 100.0),
                ]
            )
        for spec in self.action_specs:
            values.extend(self._action_observation_features(spec, roi.roi_id, links))
        return np.asarray([_clip01(value) for value in values], dtype=np.float32)

    def _action_observation_features(
        self,
        spec: FixedActionSpec,
        roi_id: str,
        links: Mapping[str, LinkState],
    ) -> tuple[float, float, float, float]:
        action = spec.to_action()
        if action.kind == DROP_ACTION:
            return (1.0, 0.0, 0.0, 0.0)
        profile = self._profile_for_action(roi_id, action)
        legal = profile is not None and self._is_action_legal(action, profile, links)
        if profile is None:
            return (0.0, 0.0, 1.0, 0.0)
        delay_ms = self._delay_proxy_ms(action, profile, links) if legal else self.config.deadline_ms * 2.0
        return (
            1.0 if legal else 0.0,
            _clip01(profile.predicted_quality),
            _norm(delay_ms, max(self.config.deadline_ms, 1.0) * 2.0),
            _norm(profile.compressed_bytes, 500_000.0),
        )

    def _delay_proxy_ms(
        self,
        action: SchedulingAction,
        profile: ActionProfile,
        links: Mapping[str, LinkState],
    ) -> float:
        target_node = SOURCE_NODE if action.kind == LOCAL_ACTION else action.target_node
        node = self._compute_node_with_pending(target_node)
        compute_ms = profile.inference_ms + (
            node.queue_cycles / node.frequency_cycles_per_s * 1000.0
        )
        if action.kind == LOCAL_ACTION:
            return compute_ms
        try:
            communication_ms = (
                profile.encode_ms
                + profile.decode_ms
                + offload_communication_delay_ms(
                    profile.compressed_bytes,
                    links[action.target_node],
                    result_bytes=self.config.result_bytes,
                )
            )
        except (IllegalActionError, KeyError, ValueError):
            return float("inf")
        return communication_ms + compute_ms

    def _profile_for_action(self, roi_id: str, action: SchedulingAction) -> ActionProfile | None:
        if action.kind == LOCAL_ACTION:
            return self.profiles.maybe_profile(roi_id, action.exit_level, "local")
        if action.kind == OFFLOAD_ACTION:
            return self.profiles.maybe_profile(roi_id, action.exit_level, action.compression_level)
        return None

    def _is_action_legal(
        self,
        action: SchedulingAction,
        profile: ActionProfile,
        links: Mapping[str, LinkState],
    ) -> bool:
        if action.kind == LOCAL_ACTION:
            return profile.compression_level == "local"
        if action.kind != OFFLOAD_ACTION:
            return action.kind == DROP_ACTION
        link = links.get(action.target_node)
        return bool(
            link is not None
            and link.visible
            and spectral_efficiency_for_snr(link.snr_db) > 0
            and profile.compression_level != "local"
        )

    def _compute_node_with_pending(self, node_name: str) -> ComputeNode:
        state = self.node_states[node_name]
        return state.as_compute_node(extra_queue_cycles=self.pending_cycles[node_name])

    def _links_for_slot(self, slot_index: int) -> Mapping[str, LinkState]:
        return self.link_trace.get(slot_index, {})

    def _current_roi_id(self) -> str:
        if self.current_index >= len(self.profiles.roi_order):
            return ""
        return self.profiles.roi_order[self.current_index]

    def _slot_index_for_current_roi(self) -> int:
        return self.current_index // self.config.rois_per_slot

    def _slot_completed(self, slot_index: int) -> bool:
        if self.current_index >= len(self.profiles.roi_order):
            return True
        return self._slot_index_for_current_roi() != slot_index

    def _grid_values_for_slot(self, slot_index: int) -> dict[int, float]:
        start = slot_index * self.config.rois_per_slot
        end = min(start + self.config.rois_per_slot, len(self.profiles.roi_order))
        values_by_grid: dict[int, list[float]] = {}
        for roi_id in self.profiles.roi_order[start:end]:
            roi = self.profiles.roi(roi_id)
            values_by_grid.setdefault(roi.grid_id, []).append(roi.semantic_value)
        return {
            grid_id: grid_semantic_value(values)
            for grid_id, values in values_by_grid.items()
        }

    def _node_feature_order(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.node_configs)

    def _build_feature_names(self) -> tuple[str, ...]:
        names = [
            "roi_class_id_norm",
            "roi_detector_confidence",
            "roi_area_ratio_norm",
            "roi_semantic_value",
            "roi_class_priority",
            "roi_grid_id_norm",
            "episode_progress",
            "slot_progress",
            "grid_age_norm",
            "grid_value",
            "quality_virtual_queue_norm",
            "delay_virtual_queue_norm",
        ]
        for node in self._node_feature_order():
            names.append(f"queue_{node}_norm")
        for target_node in self.target_nodes:
            names.extend(
                [
                    f"link_{target_node}_visible",
                    f"link_{target_node}_snr_norm",
                    f"link_{target_node}_amc_norm",
                    f"link_{target_node}_propagation_norm",
                ]
            )
        for index, spec in enumerate(self.action_specs):
            for feature in ACTION_FEATURES:
                names.append(f"action_{index}_{spec.label}_{feature}")
        return tuple(names)


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _norm(value: float, scale: float) -> float:
    if scale <= 0:
        raise ValueError("scale must be positive")
    return _clip01(float(value) / scale)
