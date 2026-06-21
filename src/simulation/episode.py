"""Deterministic stage-three scheduling episodes and baseline policies."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.proxy.quality_proxy import QualityProxyModel, load_quality_proxy
from src.simulation.aosi import grid_semantic_value, update_aosi_cell_soft
from src.simulation.compute import ComputeNode, evolve_compute_queue
from src.simulation.deterministic import (
    TaskCostProfile,
    TaskEvaluation,
    evaluate_local_task,
    evaluate_offload_task,
)
from src.simulation.link import (
    IllegalActionError,
    LinkState,
    offload_communication_delay_ms,
    spectral_efficiency_for_snr,
)


SOURCE_NODE = "source"
DROP_ACTION = "drop"
LOCAL_ACTION = "local"
OFFLOAD_ACTION = "offload"
DEFAULT_LINK_BANDWIDTH_HZ = 20e6
SCENARIO_NAMES = ("default", "low_snr", "compute_congested", "tight_deadline")


@dataclass(frozen=True)
class SatelliteNodeConfig:
    """Static per-satellite compute settings used by deterministic episodes."""

    name: str
    frequency_cycles_per_s: float
    initial_queue_cycles: float = 0.0
    energy_per_cycle_j: float = 1e-9


@dataclass(frozen=True)
class SimulationConfig:
    """Scenario constants; these are fixed during one comparable baseline run."""

    delta_t_s: float = 1.0
    rois_per_slot: int = 64
    deadline_ms: float = 500.0
    quality_threshold: float = 0.5
    result_bytes: int = 64
    tx_power_w: float = 5.0
    delay_ref_ms: float = 1000.0
    energy_ref_j: float = 1.0
    aosi_ref: float = 64.0
    quality_weight: float = 1.0
    delay_weight: float = 0.15
    energy_weight: float = 0.05
    aosi_weight: float = 0.05
    illegal_penalty: float = 1.0
    timeout_penalty: float = 0.2
    quality_penalty: float = 0.2
    aosi_alpha: float = 0.2


@dataclass(frozen=True)
class ScenarioBundle:
    """Concrete stage-three scenario assembled from config, nodes, and link trace."""

    name: str
    config: SimulationConfig
    node_configs: tuple[SatelliteNodeConfig, ...]
    link_trace: dict[int, dict[str, LinkState]]
    bandwidth_hz: float
    snr_offset_db: float = 0.0


@dataclass(frozen=True)
class RoiState:
    """Online-available ROI metadata carried from the offline profiling table."""

    roi_id: str
    image_id: str
    predicted_class: str
    class_id: int
    detector_confidence: float
    area_ratio: float
    semantic_value: float
    grid_id: int
    class_priority: float


@dataclass(frozen=True)
class ActionProfile:
    """Offline cost/quality row for one ROI, one exit, and one compression level."""

    roi_id: str
    exit_level: int
    compression_level: str
    compressed_bytes: int
    encode_ms: float
    decode_ms: float
    inference_ms: float
    task_quality: float
    predicted_quality: float
    output_width: int
    output_height: int

    def to_task_profile(self, *, result_bytes: int) -> TaskCostProfile:
        return TaskCostProfile(
            compressed_bytes=self.compressed_bytes,
            encode_ms=self.encode_ms,
            decode_ms=self.decode_ms,
            inference_ms=self.inference_ms,
            result_bytes=result_bytes,
        )


@dataclass(frozen=True)
class SchedulingAction:
    """Atomic scheduling choice for one ROI inside one time slot."""

    kind: str
    target_node: str = ""
    exit_level: int = 0
    compression_level: str = ""

    @property
    def label(self) -> str:
        if self.kind == DROP_ACTION:
            return DROP_ACTION
        if self.kind == LOCAL_ACTION:
            return f"local:e{self.exit_level}"
        return f"{self.target_node}:e{self.exit_level}:{self.compression_level}"


@dataclass(frozen=True)
class PolicyContext:
    """Read-only state visible to deterministic baseline policies."""

    slot_index: int
    links: Mapping[str, LinkState]
    node_queues: Mapping[str, float]
    grid_ages: Mapping[int, float]
    config: SimulationConfig
    profiles: "ActionProfileTable"


@dataclass
class NodeRuntimeState:
    """Mutable queue state for one satellite during an episode."""

    config: SatelliteNodeConfig
    queue_cycles: float

    def as_compute_node(self, *, extra_queue_cycles: float = 0.0) -> ComputeNode:
        return ComputeNode(
            frequency_cycles_per_s=self.config.frequency_cycles_per_s,
            queue_cycles=self.queue_cycles + extra_queue_cycles,
            energy_per_cycle_j=self.config.energy_per_cycle_j,
        )


@dataclass(frozen=True)
class DecisionResult:
    """Per-ROI accounting record written to the detailed decisions CSV."""

    policy: str
    slot_index: int
    roi_id: str
    image_id: str
    grid_id: int
    semantic_value: float
    action_label: str
    action_kind: str
    target_node: str
    exit_level: int
    compression_level: str
    predicted_quality: float
    true_task_quality: float
    total_delay_ms: float
    communication_delay_ms: float
    queue_wait_ms: float
    compute_delay_ms: float
    total_energy_j: float
    communication_energy_j: float
    compute_energy_j: float
    compressed_bytes: int
    task_cycles: float
    success: bool
    timeout: bool
    quality_violation: bool
    illegal: bool
    raw_action_index: int = -1
    mapped_action_index: int = -1
    candidate_mode: str = "fixed"
    candidate_count: int = 0
    candidate_labels: str = ""
    candidate_remapped: bool = False

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SlotSummary:
    """Per-slot aggregate reward and cost terms."""

    policy: str
    slot_index: int
    roi_count: int
    reward: float
    aosi_cost: float
    semantic_value_sum: float
    semantic_quality_sum: float
    total_delay_ms: float
    total_energy_j: float


@dataclass(frozen=True)
class PolicyRunResult:
    """Complete deterministic run for a single baseline policy."""

    policy: str
    decisions: list[DecisionResult]
    slots: list[SlotSummary]
    metrics: dict[str, object]


class ActionProfileTable:
    """Index offline ROI x exit x compression rows for online table lookup."""

    def __init__(
        self,
        rois: Mapping[str, RoiState],
        profiles: Mapping[tuple[str, int, str], ActionProfile],
        roi_order: Sequence[str],
    ) -> None:
        self._rois = dict(rois)
        self._profiles = dict(profiles)
        self.roi_order = tuple(roi_order)
        # Legal-action generation is called for every ROI, so precompute compact
        # indexes instead of repeatedly scanning the full profiling table.
        compression_by_roi: dict[str, set[str]] = {}
        exits_by_roi_compression: dict[tuple[str, str], set[int]] = {}
        for roi_id, exit_level, compression_level in self._profiles:
            compression_by_roi.setdefault(roi_id, set()).add(compression_level)
            exits_by_roi_compression.setdefault((roi_id, compression_level), set()).add(exit_level)
        self._compression_levels_by_roi = {
            roi_id: tuple(sorted(levels))
            for roi_id, levels in compression_by_roi.items()
        }
        self._exit_levels_by_roi_compression = {
            key: tuple(sorted(levels))
            for key, levels in exits_by_roi_compression.items()
        }

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        quality_proxy_path: Path | None = None,
        limit_rois: int | None = None,
        use_oracle_quality: bool = False,
    ) -> "ActionProfileTable":
        """Load profiling data and optionally replace oracle quality with proxy predictions."""

        proxy = None
        if quality_proxy_path is not None and not use_oracle_quality:
            proxy = load_quality_proxy(quality_proxy_path)
        return cls.from_frame(
            pd.read_csv(path),
            quality_proxy=proxy,
            limit_rois=limit_rois,
        )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        quality_proxy: QualityProxyModel | None = None,
        limit_rois: int | None = None,
    ) -> "ActionProfileTable":
        """Build the online action table while preserving the established schema."""

        required = {
            "roi_id",
            "image_id",
            "predicted_class",
            "class_id",
            "detector_confidence",
            "area_ratio",
            "semantic_value",
            "grid_id",
            "class_priority",
            "exit_level",
            "compression_level",
            "compressed_bytes",
            "encode_ms",
            "decode_ms",
            "inference_ms",
            "task_quality",
            "output_width",
            "output_height",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"profile frame missing fields: {missing}")

        frame = frame.copy()
        if limit_rois is not None:
            if limit_rois <= 0:
                raise ValueError("limit_rois must be positive")
            selected = list(dict.fromkeys(frame["roi_id"].astype(str)))[:limit_rois]
            frame = frame[frame["roi_id"].astype(str).isin(selected)].copy()

        if quality_proxy is None:
            # Tests and diagnostic runs can use true profiled quality directly.
            frame["_predicted_quality"] = pd.to_numeric(frame["task_quality"], errors="raise")
        else:
            # The deterministic environment uses the same online-available
            # features as the future RL environment; leakage fields stay unused.
            frame["_predicted_quality"] = quality_proxy.predict_quality(frame)

        rois: dict[str, RoiState] = {}
        profiles: dict[tuple[str, int, str], ActionProfile] = {}
        roi_order: list[str] = []

        for row in frame.to_dict("records"):
            roi_id = str(row["roi_id"])
            if roi_id not in rois:
                rois[roi_id] = RoiState(
                    roi_id=roi_id,
                    image_id=str(row["image_id"]),
                    predicted_class=str(row["predicted_class"]),
                    class_id=int(row["class_id"]),
                    detector_confidence=float(row["detector_confidence"]),
                    area_ratio=float(row["area_ratio"]),
                    semantic_value=float(row["semantic_value"]),
                    grid_id=int(row["grid_id"]),
                    class_priority=float(row["class_priority"]),
                )
                roi_order.append(roi_id)

            exit_level = int(row["exit_level"])
            compression_level = str(row["compression_level"])
            profiles[(roi_id, exit_level, compression_level)] = ActionProfile(
                roi_id=roi_id,
                exit_level=exit_level,
                compression_level=compression_level,
                compressed_bytes=int(float(row["compressed_bytes"])),
                encode_ms=float(row["encode_ms"]),
                decode_ms=float(row["decode_ms"]),
                inference_ms=float(row["inference_ms"]),
                task_quality=float(row["task_quality"]),
                predicted_quality=float(row["_predicted_quality"]),
                output_width=int(float(row["output_width"])),
                output_height=int(float(row["output_height"])),
            )

        if not roi_order:
            raise ValueError("profile frame has no ROI rows")
        return cls(rois=rois, profiles=profiles, roi_order=roi_order)

    def roi(self, roi_id: str) -> RoiState:
        return self._rois[roi_id]

    def profile(self, roi_id: str, exit_level: int, compression_level: str) -> ActionProfile:
        return self._profiles[(roi_id, exit_level, compression_level)]

    def maybe_profile(
        self,
        roi_id: str,
        exit_level: int,
        compression_level: str,
    ) -> ActionProfile | None:
        return self._profiles.get((roi_id, exit_level, compression_level))

    def compression_levels(self, roi_id: str) -> tuple[str, ...]:
        return self._compression_levels_by_roi.get(roi_id, ())

    def exit_levels(self, roi_id: str, compression_level: str) -> tuple[int, ...]:
        return self._exit_levels_by_roi_compression.get((roi_id, compression_level), ())


class SchedulingPolicy:
    """Minimal policy interface shared by hand-written deterministic baselines."""

    name = "base"

    def choose(
        self,
        roi: RoiState,
        legal_actions: Sequence[SchedulingAction],
        context: PolicyContext,
    ) -> SchedulingAction:
        raise NotImplementedError


class RandomLegalPolicy(SchedulingPolicy):
    name = "Random"

    def __init__(self, *, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def choose(
        self,
        roi: RoiState,
        legal_actions: Sequence[SchedulingAction],
        context: PolicyContext,
    ) -> SchedulingAction:
        return self._rng.choice(list(legal_actions))


class LocalDeepPolicy(SchedulingPolicy):
    name = "Local-Deep"

    def choose(
        self,
        roi: RoiState,
        legal_actions: Sequence[SchedulingAction],
        context: PolicyContext,
    ) -> SchedulingAction:
        local = [action for action in legal_actions if action.kind == LOCAL_ACTION]
        return max(local, key=lambda action: action.exit_level) if local else legal_actions[0]


class LocalAdaptivePolicy(SchedulingPolicy):
    name = "Local-Adaptive"

    def choose(
        self,
        roi: RoiState,
        legal_actions: Sequence[SchedulingAction],
        context: PolicyContext,
    ) -> SchedulingAction:
        local = sorted(
            [action for action in legal_actions if action.kind == LOCAL_ACTION],
            key=lambda action: action.exit_level,
        )
        for action in local:
            profile = context.profiles.profile(roi.roi_id, action.exit_level, action.compression_level)
            if profile.predicted_quality >= context.config.quality_threshold:
                return action
        return local[-1] if local else legal_actions[0]


class BestSnrPolicy(SchedulingPolicy):
    name = "Best-SNR"

    def choose(
        self,
        roi: RoiState,
        legal_actions: Sequence[SchedulingAction],
        context: PolicyContext,
    ) -> SchedulingAction:
        offload = [action for action in legal_actions if action.kind == OFFLOAD_ACTION]
        if not offload:
            return LocalDeepPolicy().choose(roi, legal_actions, context)

        def rank(action: SchedulingAction) -> tuple[float, int, int]:
            link = context.links[action.target_node]
            # Prefer beta_1 as a stable high-quality offload setting when SNR ties.
            preferred_beta = 1 if action.compression_level == "beta_1" else 0
            return (link.snr_db, preferred_beta, action.exit_level)

        best_target = max({action.target_node for action in offload}, key=lambda node: context.links[node].snr_db)
        target_actions = [action for action in offload if action.target_node == best_target]
        return max(target_actions, key=rank)


class SemanticGreedyPolicy(SchedulingPolicy):
    name = "Semantic-Greedy"

    def __init__(self, *, drop_threshold: float = 0.25) -> None:
        self.drop_threshold = drop_threshold

    def choose(
        self,
        roi: RoiState,
        legal_actions: Sequence[SchedulingAction],
        context: PolicyContext,
    ) -> SchedulingAction:
        if roi.semantic_value < self.drop_threshold:
            return SchedulingAction(kind=DROP_ACTION)

        scored: list[tuple[float, SchedulingAction]] = []
        for action in legal_actions:
            if action.kind == DROP_ACTION:
                scored.append((0.0, action))
                continue
            profile = context.profiles.profile(
                roi.roi_id,
                action.exit_level,
                action.compression_level,
            )
            # This is a deterministic baseline score, not the final RL reward.
            byte_penalty = profile.compressed_bytes / 100_000.0
            delay_penalty = profile.inference_ms / max(context.config.deadline_ms, 1.0)
            score = roi.semantic_value * profile.predicted_quality - 0.02 * byte_penalty - 0.05 * delay_penalty
            if action.kind == LOCAL_ACTION:
                score += 0.02
            scored.append((score, action))
        return max(scored, key=lambda item: (item[0], item[1].exit_level))[1]


class NoAosiGreedyPolicy(SchedulingPolicy):
    name = "No-AoSI"

    def __init__(self, *, drop_threshold: float = 0.25) -> None:
        self.drop_threshold = drop_threshold

    def choose(
        self,
        roi: RoiState,
        legal_actions: Sequence[SchedulingAction],
        context: PolicyContext,
    ) -> SchedulingAction:
        if roi.semantic_value < self.drop_threshold:
            return SchedulingAction(kind=DROP_ACTION)

        scored: list[tuple[float, float, float, SchedulingAction]] = []
        for action in legal_actions:
            if action.kind == DROP_ACTION:
                scored.append((0.0, 0.0, 0.0, action))
                continue
            profile = context.profiles.profile(
                roi.roi_id,
                action.exit_level,
                action.compression_level,
            )
            # No-AoSI is the short-sighted ablation: it uses quality, delay and
            # traffic cost, but deliberately never reads context.grid_ages.
            delay_ms = _policy_delay_proxy_ms(action, profile, context)
            delay_penalty = delay_ms / max(context.config.deadline_ms, 1.0)
            byte_penalty = profile.compressed_bytes / 100_000.0
            score = (
                roi.semantic_value * profile.predicted_quality
                - 0.12 * delay_penalty
                - 0.02 * byte_penalty
            )
            scored.append((score, profile.predicted_quality, -float(profile.compressed_bytes), action))
        return max(scored, key=lambda item: (item[0], item[1], item[2], item[3].exit_level))[3]


class AosiGreedyPolicy(SchedulingPolicy):
    name = "AoSI-Greedy"

    def choose(
        self,
        roi: RoiState,
        legal_actions: Sequence[SchedulingAction],
        context: PolicyContext,
    ) -> SchedulingAction:
        age = context.grid_ages.get(roi.grid_id, 0.0)
        scored: list[tuple[float, SchedulingAction]] = []
        for action in legal_actions:
            if action.kind == DROP_ACTION:
                scored.append((-roi.semantic_value * (1.0 + age), action))
                continue
            profile = context.profiles.profile(
                roi.roi_id,
                action.exit_level,
                action.compression_level,
            )
            # Older grids receive a small priority lift so the baseline exercises
            # the AoSI state instead of behaving like plain semantic greedy.
            quality_term = roi.semantic_value * profile.predicted_quality * (1.0 + 0.2 * age)
            delay_term = _policy_delay_proxy_ms(action, profile, context) / max(
                context.config.deadline_ms,
                1.0,
            )
            # Stale grids value quick refresh more than marginal offload savings;
            # this makes the AoSI baseline observably depend on grid age.
            freshness_pressure = min(age / 5.0, 2.0)
            offload_bonus = 0.03 if action.kind == OFFLOAD_ACTION else 0.0
            scored.append(
                (
                    quality_term
                    + offload_bonus
                    - (0.03 + 0.18 * freshness_pressure) * delay_term,
                    action,
                )
            )
        return max(scored, key=lambda item: (item[0], item[1].exit_level))[1]


def default_node_configs() -> tuple[SatelliteNodeConfig, ...]:
    return (
        SatelliteNodeConfig(name=SOURCE_NODE, frequency_cycles_per_s=2.0e9, energy_per_cycle_j=1e-9),
        SatelliteNodeConfig(name="sat_1", frequency_cycles_per_s=2.5e9, energy_per_cycle_j=1e-9),
        SatelliteNodeConfig(name="sat_2", frequency_cycles_per_s=1.8e9, energy_per_cycle_j=1.2e-9),
    )


def default_policies(seed: int = 42) -> tuple[SchedulingPolicy, ...]:
    return (
        RandomLegalPolicy(seed=seed),
        LocalDeepPolicy(),
        LocalAdaptivePolicy(),
        BestSnrPolicy(),
        SemanticGreedyPolicy(),
        NoAosiGreedyPolicy(),
        AosiGreedyPolicy(),
    )


def build_scenario(
    scenario: str,
    *,
    base_config: SimulationConfig,
    node_configs: Sequence[SatelliteNodeConfig] | None = None,
    slot_count: int,
    bandwidth_hz: float = DEFAULT_LINK_BANDWIDTH_HZ,
) -> ScenarioBundle:
    """Create reproducible deterministic stress scenarios without external simulators."""

    if slot_count <= 0:
        raise ValueError("slot_count must be positive")
    scenario_name = scenario.lower()
    if scenario_name not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIO_NAMES}")

    config = base_config
    nodes = tuple(node_configs) if node_configs is not None else default_node_configs()
    snr_offset_db = 0.0

    if scenario_name == "low_snr":
        # A uniform SNR offset preserves visibility timing while forcing AMC to
        # choose lower spectral-efficiency modes, exposing communication costs.
        snr_offset_db = -8.0
    elif scenario_name == "compute_congested":
        nodes = tuple(
            replace(node, initial_queue_cycles=node.initial_queue_cycles + 5.0e9)
            if node.name != SOURCE_NODE
            else node
            for node in nodes
        )
    elif scenario_name == "tight_deadline":
        config = replace(config, deadline_ms=min(config.deadline_ms, 120.0))

    target_nodes = [node.name for node in nodes if node.name != SOURCE_NODE]
    link_trace = generate_deterministic_link_trace(
        slot_count,
        target_nodes,
        bandwidth_hz=bandwidth_hz,
        snr_offset_db=snr_offset_db,
    )
    return ScenarioBundle(
        name=scenario_name,
        config=config,
        node_configs=nodes,
        link_trace=link_trace,
        bandwidth_hz=bandwidth_hz,
        snr_offset_db=snr_offset_db,
    )


def generate_deterministic_link_trace(
    time_slots: int,
    target_nodes: Sequence[str],
    *,
    bandwidth_hz: float = DEFAULT_LINK_BANDWIDTH_HZ,
    snr_offset_db: float = 0.0,
) -> dict[int, dict[str, LinkState]]:
    """Generate a repeatable synthetic visibility/SNR trace for stage-three validation."""

    trace: dict[int, dict[str, LinkState]] = {}
    for slot in range(time_slots):
        links: dict[str, LinkState] = {}
        for index, node in enumerate(target_nodes):
            period = 4 + index
            visible = ((slot + index) % period) != period - 1
            snr_db = 4.0 + ((slot * (5 + index) + index * 7) % 18) + snr_offset_db
            if not visible:
                snr_db = -1.0
            links[node] = LinkState(
                visible=visible,
                snr_db=snr_db,
                bandwidth_hz=bandwidth_hz,
                propagation_delay_ms=2.0 + index,
            )
        trace[slot] = links
    return trace


def legal_actions_for_roi(
    profiles: ActionProfileTable,
    roi_id: str,
    links: Mapping[str, LinkState],
) -> tuple[SchedulingAction, ...]:
    """Enumerate physically legal actions; invisible offload links are excluded."""

    actions: list[SchedulingAction] = [SchedulingAction(kind=DROP_ACTION)]
    for exit_level in profiles.exit_levels(roi_id, "local"):
        actions.append(
            SchedulingAction(
                kind=LOCAL_ACTION,
                exit_level=exit_level,
                compression_level="local",
            )
        )

    for target_node, link in sorted(links.items()):
        if not link.visible or spectral_efficiency_for_snr(link.snr_db) <= 0:
            continue
        for compression_level in profiles.compression_levels(roi_id):
            if compression_level == "local":
                continue
            for exit_level in profiles.exit_levels(roi_id, compression_level):
                actions.append(
                    SchedulingAction(
                        kind=OFFLOAD_ACTION,
                        target_node=target_node,
                        exit_level=exit_level,
                        compression_level=compression_level,
                    )
                )
    return tuple(actions)


def _policy_delay_proxy_ms(
    action: SchedulingAction,
    profile: ActionProfile,
    context: PolicyContext,
) -> float:
    """Estimate action delay for greedy ranking without mutating queues or AoSI."""

    if action.kind == LOCAL_ACTION:
        return profile.inference_ms
    try:
        link = context.links[action.target_node]
        communication_ms = (
            profile.encode_ms
            + profile.decode_ms
            + offload_communication_delay_ms(
                profile.compressed_bytes,
                link,
                result_bytes=context.config.result_bytes,
            )
        )
    except (IllegalActionError, KeyError, ValueError):
        return float("inf")
    return communication_ms + profile.inference_ms


def run_policy_episode(
    profiles: ActionProfileTable,
    policy: SchedulingPolicy,
    *,
    config: SimulationConfig,
    node_configs: Sequence[SatelliteNodeConfig],
    link_trace: Mapping[int, Mapping[str, LinkState]],
) -> PolicyRunResult:
    """Run one baseline over the ROI stream without invoking any vision model online."""

    if config.rois_per_slot <= 0:
        raise ValueError("rois_per_slot must be positive")
    node_states = {
        node.name: NodeRuntimeState(config=node, queue_cycles=node.initial_queue_cycles)
        for node in node_configs
    }
    if SOURCE_NODE not in node_states:
        raise ValueError(f"node_configs must include {SOURCE_NODE!r}")

    decisions: list[DecisionResult] = []
    slots: list[SlotSummary] = []
    grid_ages: dict[int, float] = {}
    previous_grid_values: dict[int, float] = {}
    total_slots = math.ceil(len(profiles.roi_order) / config.rois_per_slot)

    for slot_index in range(total_slots):
        slot_roi_ids = profiles.roi_order[
            slot_index * config.rois_per_slot : (slot_index + 1) * config.rois_per_slot
        ]
        links = dict(link_trace.get(slot_index, {}))
        pending_cycles = {name: 0.0 for name in node_states}
        current_values = _grid_values_for_slot(profiles, slot_roi_ids)
        grid_total_value: dict[int, float] = {}
        grid_success_value: dict[int, float] = {}
        slot_decisions: list[DecisionResult] = []

        context = PolicyContext(
            slot_index=slot_index,
            links=links,
            node_queues={name: state.queue_cycles for name, state in node_states.items()},
            grid_ages=grid_ages,
            config=config,
            profiles=profiles,
        )

        for roi_id in slot_roi_ids:
            roi = profiles.roi(roi_id)
            grid_total_value[roi.grid_id] = grid_total_value.get(roi.grid_id, 0.0) + roi.semantic_value
            legal_actions = legal_actions_for_roi(profiles, roi_id, links)
            action = policy.choose(roi, legal_actions, context)
            if action not in legal_actions:
                # Baselines should not do this, but the guard keeps future policies
                # from silently converting illegal choices into plausible delays.
                action = SchedulingAction(kind=DROP_ACTION)
                illegal = True
            else:
                illegal = False
            result = _evaluate_action(
                policy_name=policy.name,
                slot_index=slot_index,
                roi=roi,
                action=action,
                profiles=profiles,
                node_states=node_states,
                pending_cycles=pending_cycles,
                links=links,
                config=config,
                forced_illegal=illegal,
            )
            if result.success:
                grid_success_value[roi.grid_id] = (
                    grid_success_value.get(roi.grid_id, 0.0) + roi.semantic_value
                )
            slot_decisions.append(result)

        for name, state in node_states.items():
            # Queue evolution follows PROJECT_CONTEXT.md: serve fixed-frequency
            # cycles for one slot, then append arrivals selected in this slot.
            state.queue_cycles = evolve_compute_queue(
                queue_cycles=state.queue_cycles,
                frequency_cycles_per_s=state.config.frequency_cycles_per_s,
                delta_t_s=config.delta_t_s,
                arrival_cycles=pending_cycles[name],
            )

        aosi_cost = _update_aosi(
            current_values=current_values,
            previous_values=previous_grid_values,
            grid_ages=grid_ages,
            grid_total_value=grid_total_value,
            grid_success_value=grid_success_value,
            config=config,
        )
        previous_grid_values = {
            grid_id: current_values.get(grid_id, 0.0)
            for grid_id in set(previous_grid_values) | set(current_values)
        }
        slot_summary = build_slot_summary(
            policy.name,
            slot_index,
            slot_decisions,
            aosi_cost=aosi_cost,
            config=config,
        )
        slots.append(slot_summary)
        decisions.extend(slot_decisions)

    return PolicyRunResult(
        policy=policy.name,
        decisions=decisions,
        slots=slots,
        metrics=summarize_decisions(policy.name, decisions, slots),
    )


def run_baseline_suite(
    profiles: ActionProfileTable,
    *,
    config: SimulationConfig,
    node_configs: Sequence[SatelliteNodeConfig],
    policies: Sequence[SchedulingPolicy],
    link_trace: Mapping[int, Mapping[str, LinkState]],
) -> list[PolicyRunResult]:
    """Evaluate all deterministic baselines on the same ROI stream and link trace."""

    return [
        run_policy_episode(
            profiles,
            policy,
            config=config,
            node_configs=node_configs,
            link_trace=link_trace,
        )
        for policy in policies
    ]


def summarize_decisions(
    policy_name: str,
    decisions: Sequence[DecisionResult],
    slots: Sequence[SlotSummary],
) -> dict[str, object]:
    """Aggregate policy-level metrics used by reports and comparison tables."""

    processed = [row for row in decisions if row.action_kind != DROP_ACTION and not row.illegal]
    delays = [row.total_delay_ms for row in processed]
    qualities = [row.predicted_quality for row in processed]
    semantic_total = sum(row.semantic_value for row in decisions)
    semantic_quality = sum(row.semantic_value * row.predicted_quality for row in processed)
    return {
        "policy": policy_name,
        "roi_count": len(decisions),
        "processed_count": len(processed),
        "drop_count": sum(1 for row in decisions if row.action_kind == DROP_ACTION),
        "offload_count": sum(1 for row in processed if row.action_kind == OFFLOAD_ACTION),
        "local_count": sum(1 for row in processed if row.action_kind == LOCAL_ACTION),
        "illegal_count": sum(1 for row in decisions if row.illegal),
        "executed_illegal_count": sum(1 for row in decisions if row.illegal),
        "success_count": sum(1 for row in decisions if row.success),
        "timeout_count": sum(1 for row in decisions if row.timeout),
        "quality_violation_count": sum(1 for row in decisions if row.quality_violation),
        "candidate_remap_count": sum(1 for row in decisions if row.candidate_remapped),
        "mean_candidate_count": _mean([row.candidate_count for row in decisions if row.candidate_count > 0]),
        "success_rate": _safe_div(sum(1 for row in decisions if row.success), len(decisions)),
        "semantic_success_rate": _safe_div(
            sum(row.semantic_value for row in decisions if row.success),
            semantic_total,
        ),
        "mean_quality": _mean(qualities),
        "semantic_quality_sum": semantic_quality,
        "mean_delay_ms": _mean(delays),
        "p95_delay_ms": _percentile(delays, 95),
        "total_energy_j": sum(row.total_energy_j for row in processed),
        "total_compressed_bytes": sum(row.compressed_bytes for row in processed),
        "total_aosi_cost": sum(slot.aosi_cost for slot in slots),
        "qoe_total": sum(slot.reward for slot in slots),
    }


def _evaluate_action(
    *,
    policy_name: str,
    slot_index: int,
    roi: RoiState,
    action: SchedulingAction,
    profiles: ActionProfileTable,
    node_states: Mapping[str, NodeRuntimeState],
    pending_cycles: dict[str, float],
    links: Mapping[str, LinkState],
    config: SimulationConfig,
    forced_illegal: bool = False,
) -> DecisionResult:
    """Convert a selected action into delay, energy, quality, and queue arrivals."""

    empty_evaluation = TaskEvaluation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if action.kind == DROP_ACTION or forced_illegal:
        return _decision_from_evaluation(
            policy_name=policy_name,
            slot_index=slot_index,
            roi=roi,
            action=action,
            profile=None,
            evaluation=empty_evaluation,
            config=config,
            illegal=forced_illegal,
        )

    try:
        profile = profiles.profile(roi.roi_id, action.exit_level, action.compression_level)
    except KeyError:
        return _decision_from_evaluation(
            policy_name=policy_name,
            slot_index=slot_index,
            roi=roi,
            action=action,
            profile=None,
            evaluation=empty_evaluation,
            config=config,
            illegal=True,
        )

    target_node = SOURCE_NODE if action.kind == LOCAL_ACTION else action.target_node
    if target_node not in node_states:
        return _decision_from_evaluation(
            policy_name=policy_name,
            slot_index=slot_index,
            roi=roi,
            action=action,
            profile=profile,
            evaluation=empty_evaluation,
            config=config,
            illegal=True,
        )
    node = node_states[target_node].as_compute_node(extra_queue_cycles=pending_cycles[target_node])
    task_profile = profile.to_task_profile(result_bytes=config.result_bytes)

    try:
        if action.kind == LOCAL_ACTION:
            evaluation = evaluate_local_task(task_profile, node)
        else:
            evaluation = evaluate_offload_task(
                task_profile,
                links[action.target_node],
                node,
                tx_power_w=config.tx_power_w,
            )
    except (IllegalActionError, KeyError):
        return _decision_from_evaluation(
            policy_name=policy_name,
            slot_index=slot_index,
            roi=roi,
            action=action,
            profile=profile,
            evaluation=empty_evaluation,
            config=config,
            illegal=True,
        )

    pending_cycles[target_node] += evaluation.task_cycles
    return _decision_from_evaluation(
        policy_name=policy_name,
        slot_index=slot_index,
        roi=roi,
        action=action,
        profile=profile,
        evaluation=evaluation,
        config=config,
        illegal=False,
    )


def _decision_from_evaluation(
    *,
    policy_name: str,
    slot_index: int,
    roi: RoiState,
    action: SchedulingAction,
    profile: ActionProfile | None,
    evaluation: TaskEvaluation,
    config: SimulationConfig,
    illegal: bool,
) -> DecisionResult:
    """Apply deadline and quality constraints to a physical task evaluation."""

    predicted_quality = 0.0 if profile is None or illegal else profile.predicted_quality
    true_quality = 0.0 if profile is None or illegal else profile.task_quality
    timeout = (
        action.kind != DROP_ACTION
        and not illegal
        and evaluation.total_delay_ms > config.deadline_ms
    )
    quality_violation = (
        action.kind != DROP_ACTION
        and not illegal
        and predicted_quality < config.quality_threshold
    )
    success = action.kind != DROP_ACTION and not illegal and not timeout and not quality_violation
    return DecisionResult(
        policy=policy_name,
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


def _grid_values_for_slot(
    profiles: ActionProfileTable,
    roi_ids: Iterable[str],
) -> dict[int, float]:
    """Compute probability-OR grid values from ROI semantic values."""

    values_by_grid: dict[int, list[float]] = {}
    for roi_id in roi_ids:
        roi = profiles.roi(roi_id)
        values_by_grid.setdefault(roi.grid_id, []).append(roi.semantic_value)
    return {
        grid_id: grid_semantic_value(values)
        for grid_id, values in values_by_grid.items()
    }


def _update_aosi(
    *,
    current_values: Mapping[int, float],
    previous_values: Mapping[int, float],
    grid_ages: dict[int, float],
    grid_total_value: Mapping[int, float],
    grid_success_value: Mapping[int, float],
    config: SimulationConfig,
) -> float:
    """Apply grid-level soft reset based on successful semantic value share."""

    total_cost = 0.0
    active_grids = set(previous_values) | set(current_values)
    for grid_id in active_grids:
        total_value = grid_total_value.get(grid_id, 0.0)
        achievement_rate = _safe_div(grid_success_value.get(grid_id, 0.0), total_value)
        cell = update_aosi_cell_soft(
            previous_value=previous_values.get(grid_id, 0.0),
            current_value=current_values.get(grid_id, 0.0),
            previous_age_s=grid_ages.get(grid_id, 0.0),
            delta_t_s=config.delta_t_s,
            achievement_rate=achievement_rate,
            alpha=config.aosi_alpha,
        )
        grid_ages[grid_id] = cell.age_s
        total_cost += cell.cost
    return total_cost


def build_slot_summary(
    policy_name: str,
    slot_index: int,
    decisions: Sequence[DecisionResult],
    *,
    aosi_cost: float,
    config: SimulationConfig,
) -> SlotSummary:
    """Build the canonical slot QoE from quality, delay, energy, and AoSI."""

    processed = [row for row in decisions if row.action_kind != DROP_ACTION and not row.illegal]
    semantic_value_sum = sum(row.semantic_value for row in decisions)
    semantic_quality_sum = sum(row.semantic_value * row.predicted_quality for row in processed)
    total_delay_ms = sum(row.total_delay_ms for row in processed)
    total_energy_j = sum(row.total_energy_j for row in processed)
    timeout_count = sum(1 for row in processed if row.timeout)
    quality_violations = sum(1 for row in processed if row.quality_violation)
    illegal_count = sum(1 for row in decisions if row.illegal)
    reward = (
        config.quality_weight * _safe_div(semantic_quality_sum, semantic_value_sum)
        - config.delay_weight * _safe_div(total_delay_ms, config.delay_ref_ms)
        - config.energy_weight * _safe_div(total_energy_j, config.energy_ref_j)
        - config.aosi_weight * _safe_div(aosi_cost, config.aosi_ref)
        - config.illegal_penalty * illegal_count
        - config.timeout_penalty * timeout_count
        - config.quality_penalty * quality_violations
    )
    return SlotSummary(
        policy=policy_name,
        slot_index=slot_index,
        roi_count=len(decisions),
        reward=reward,
        aosi_cost=aosi_cost,
        semantic_value_sum=semantic_value_sum,
        semantic_quality_sum=semantic_quality_sum,
        total_delay_ms=total_delay_ms,
        total_energy_j=total_energy_j,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else float(sum(values) / len(values))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[int(index)])
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
