"""Train a minimal PPO smoke policy for LEO semantic scheduling."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.envs import LeoSchedulingEnv  # noqa: E402
from src.simulation.episode import (  # noqa: E402
    DEFAULT_LINK_BANDWIDTH_HZ,
    SCENARIO_NAMES,
    ActionProfileTable,
    PolicyRunResult,
    SatelliteNodeConfig,
    SchedulingPolicy,
    SimulationConfig,
    build_scenario,
    default_node_configs,
    default_policies,
    run_baseline_suite,
)


DEFAULT_CONFIG_PATH = ROOT / "configs" / "satellite_env.yaml"
DEFAULT_PROFILE_CSV = (
    ROOT
    / "data"
    / "profiling"
    / "dota_v1_lite_300_100_100"
    / "test_full"
    / "roi_profile_smoke.csv"
)
DEFAULT_QUALITY_PROXY = ROOT / "outputs" / "proxy" / "test_full" / "quality_proxy.joblib"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "rl" / "ppo_smoke"
REPORT_COLUMNS = (
    ("policy", "Policy"),
    ("success_rate", "Success"),
    ("semantic_success_rate", "Semantic Success"),
    ("mean_delay_ms", "Mean Delay ms"),
    ("p95_delay_ms", "P95 Delay ms"),
    ("total_energy_j", "Energy J"),
    ("total_compressed_bytes", "Traffic Bytes"),
    ("total_aosi_cost", "AoSI Cost"),
    ("qoe_total", "QoE"),
)
SCENARIO_SUMMARY_COLUMNS = (
    ("scenario", "Scenario"),
    ("best_policy", "Best Policy"),
    ("best_qoe", "Best QoE"),
    ("proposed_rl_qoe", "PPO QoE"),
    ("proposed_rl_rank", "PPO Rank"),
    ("proposed_rl_gap_to_best_qoe", "PPO Gap"),
)
POLICY_AGGREGATE_COLUMNS = (
    ("policy", "Policy"),
    ("scenario_count", "Scenarios"),
    ("mean_rank", "Mean Rank"),
    ("best_scenario_count", "Best Count"),
    ("mean_qoe", "Mean QoE"),
    ("mean_qoe_gap_to_best", "Mean Gap"),
    ("mean_success_rate", "Mean Success"),
    ("mean_semantic_success_rate", "Mean Semantic Success"),
    ("mean_delay_ms", "Mean Delay ms"),
    ("mean_total_energy_j", "Mean Energy J"),
    ("mean_total_aosi_cost", "Mean AoSI Cost"),
)


@dataclass(frozen=True)
class PpoSmokeSettings:
    """Resolved training contract for one PPO smoke run."""

    config_path: Path | None
    config_loaded: bool
    profile_csv: Path
    quality_proxy: Path
    output_dir: Path
    limit_rois: int
    total_timesteps: int
    seed: int
    scenario: str
    all_scenarios: bool
    eval_only: bool
    model_path: Path | None
    exist_ok: bool
    ppo_n_steps: int | None
    ppo_batch_size: int | None
    ppo_n_epochs: int | None
    ppo_learning_rate: float | None
    ppo_gamma: float | None
    ppo_ent_coef: float | None
    eval_frequency: int
    checkpoint_frequency: int
    select_best_checkpoint: bool
    best_checkpoint_min_success_rate: float
    reward_quality_deficit_weight: float
    reward_delay_excess_weight: float
    reward_virtual_queue_weight: float
    bandwidth_hz: float
    base_config: SimulationConfig
    node_configs: tuple[SatelliteNodeConfig, ...]
    policy_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profile-csv", type=Path)
    parser.add_argument("--quality-proxy", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-rois", type=int, default=512)
    parser.add_argument("--total-timesteps", type=int, default=5000)
    parser.add_argument("--scenario", choices=SCENARIO_NAMES)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rois-per-slot", type=int)
    parser.add_argument("--deadline-ms", type=float)
    parser.add_argument("--quality-threshold", type=float)
    parser.add_argument("--ppo-n-steps", type=int)
    parser.add_argument("--ppo-batch-size", type=int)
    parser.add_argument("--ppo-n-epochs", type=int)
    parser.add_argument("--ppo-learning-rate", type=float)
    parser.add_argument("--ppo-gamma", type=float)
    parser.add_argument("--ppo-ent-coef", type=float)
    parser.add_argument("--eval-frequency", type=int, default=0)
    parser.add_argument("--checkpoint-frequency", type=int, default=0)
    parser.add_argument("--select-best-checkpoint", action="store_true")
    parser.add_argument("--best-checkpoint-min-success-rate", type=float, default=0.0)
    parser.add_argument("--reward-quality-deficit-weight", type=float, default=0.0)
    parser.add_argument("--reward-delay-excess-weight", type=float, default=0.0)
    parser.add_argument("--reward-virtual-queue-weight", type=float, default=0.0)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _load_yaml_config(path: Path) -> tuple[dict[str, object], bool]:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        if resolved == DEFAULT_CONFIG_PATH:
            return {}, False
        raise FileNotFoundError(f"config file not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {resolved}")
    return dict(payload), True


def _resolve_settings(args: argparse.Namespace) -> PpoSmokeSettings:
    payload, config_loaded = _load_yaml_config(args.config)
    paths = _section(payload, "paths")
    run = _section(payload, "run")
    link = _section(payload, "link")
    reward = _section(payload, "reward")
    aosi = _section(payload, "aosi")

    limit_rois = int(args.limit_rois)
    total_timesteps = int(args.total_timesteps)
    rois_per_slot = int(args.rois_per_slot if args.rois_per_slot is not None else run.get("rois_per_slot", 64))
    deadline_ms = float(args.deadline_ms if args.deadline_ms is not None else run.get("deadline_ms", 500.0))
    quality_threshold = float(
        args.quality_threshold
        if args.quality_threshold is not None
        else run.get("quality_threshold", 0.5)
    )
    seed = int(args.seed if args.seed is not None else run.get("seed", 42))
    scenario = str(args.scenario or run.get("scenario", "default")).lower()
    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIO_NAMES}")

    base_config = SimulationConfig(
        delta_t_s=float(run.get("delta_t_s", 1.0)),
        rois_per_slot=rois_per_slot,
        deadline_ms=deadline_ms,
        quality_threshold=quality_threshold,
        result_bytes=int(link.get("result_bytes", 64)),
        tx_power_w=float(link.get("tx_power_w", 5.0)),
        delay_ref_ms=float(reward.get("delay_ref_ms", 1000.0)),
        energy_ref_j=float(reward.get("energy_ref_j", 1.0)),
        aosi_ref=float(reward.get("aosi_ref", aosi.get("ref", 64.0))),
        quality_weight=float(reward.get("quality_weight", 1.0)),
        delay_weight=float(reward.get("delay_weight", 0.15)),
        energy_weight=float(reward.get("energy_weight", 0.05)),
        aosi_weight=float(reward.get("aosi_weight", 0.05)),
        illegal_penalty=float(reward.get("illegal_penalty", 1.0)),
        timeout_penalty=float(reward.get("timeout_penalty", 0.2)),
        quality_penalty=float(reward.get("quality_penalty", 0.2)),
        aosi_alpha=float(aosi.get("alpha", 0.2)),
    )
    policy_names = tuple(str(name) for name in payload.get("policies", []) or [])
    if not policy_names:
        policy_names = tuple(policy.name for policy in default_policies(seed=seed))

    return PpoSmokeSettings(
        config_path=_resolve_path(args.config) if config_loaded else None,
        config_loaded=config_loaded,
        profile_csv=_resolve_path(args.profile_csv or paths.get("profile_csv") or DEFAULT_PROFILE_CSV),
        quality_proxy=_resolve_path(args.quality_proxy or paths.get("quality_proxy") or DEFAULT_QUALITY_PROXY),
        output_dir=_resolve_path(args.output_dir),
        limit_rois=limit_rois,
        total_timesteps=total_timesteps,
        seed=seed,
        scenario=scenario,
        all_scenarios=bool(args.all_scenarios),
        eval_only=bool(args.eval_only),
        model_path=_resolve_path(args.model_path) if args.model_path is not None else None,
        exist_ok=bool(args.exist_ok),
        ppo_n_steps=args.ppo_n_steps,
        ppo_batch_size=args.ppo_batch_size,
        ppo_n_epochs=args.ppo_n_epochs,
        ppo_learning_rate=args.ppo_learning_rate,
        ppo_gamma=args.ppo_gamma,
        ppo_ent_coef=args.ppo_ent_coef,
        eval_frequency=int(args.eval_frequency),
        checkpoint_frequency=int(args.checkpoint_frequency),
        select_best_checkpoint=bool(args.select_best_checkpoint),
        best_checkpoint_min_success_rate=float(args.best_checkpoint_min_success_rate),
        reward_quality_deficit_weight=float(args.reward_quality_deficit_weight),
        reward_delay_excess_weight=float(args.reward_delay_excess_weight),
        reward_virtual_queue_weight=float(args.reward_virtual_queue_weight),
        bandwidth_hz=float(link.get("bandwidth_hz", DEFAULT_LINK_BANDWIDTH_HZ)),
        base_config=base_config,
        node_configs=_node_configs_from_payload(payload),
        policy_names=policy_names,
    )


def _section(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {key!r} must be a mapping")
    return dict(value)


def _resolve_path(value: object) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _node_configs_from_payload(payload: Mapping[str, object]) -> tuple[SatelliteNodeConfig, ...]:
    raw_nodes = payload.get("nodes")
    if raw_nodes is None:
        return default_node_configs()
    if not isinstance(raw_nodes, list):
        raise ValueError("config section 'nodes' must be a list")
    nodes: list[SatelliteNodeConfig] = []
    for item in raw_nodes:
        if not isinstance(item, dict):
            raise ValueError("each node entry must be a mapping")
        nodes.append(
            SatelliteNodeConfig(
                name=str(item["name"]),
                frequency_cycles_per_s=float(item["frequency_cycles_per_s"]),
                initial_queue_cycles=float(item.get("initial_queue_cycles", 0.0)),
                energy_per_cycle_j=float(item.get("energy_per_cycle_j", 1e-9)),
            )
        )
    return tuple(nodes)


def _validate_settings(settings: PpoSmokeSettings) -> None:
    if settings.limit_rois <= 0:
        raise ValueError("--limit-rois must be positive")
    if settings.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if not settings.profile_csv.is_file():
        raise FileNotFoundError(f"profile CSV not found: {settings.profile_csv}")
    if not settings.quality_proxy.is_file():
        raise FileNotFoundError(f"quality proxy not found: {settings.quality_proxy}")
    if settings.eval_only:
        model_path = settings.model_path or settings.output_dir / "model.zip"
        if not model_path.is_file():
            raise FileNotFoundError(f"PPO model not found for --eval-only: {model_path}")
    if settings.base_config.rois_per_slot <= 0:
        raise ValueError("rois_per_slot must be positive")
    for name, value in (
        ("--ppo-n-steps", settings.ppo_n_steps),
        ("--ppo-batch-size", settings.ppo_batch_size),
        ("--ppo-n-epochs", settings.ppo_n_epochs),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("--ppo-learning-rate", settings.ppo_learning_rate),
        ("--ppo-gamma", settings.ppo_gamma),
        ("--ppo-ent-coef", settings.ppo_ent_coef),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative")
    if settings.eval_frequency < 0:
        raise ValueError("--eval-frequency must be non-negative")
    if settings.checkpoint_frequency < 0:
        raise ValueError("--checkpoint-frequency must be non-negative")
    if settings.select_best_checkpoint and settings.eval_frequency <= 0:
        raise ValueError("--select-best-checkpoint requires --eval-frequency > 0")
    if not 0.0 <= settings.best_checkpoint_min_success_rate <= 1.0:
        raise ValueError("--best-checkpoint-min-success-rate must be in [0, 1]")
    for name, value in (
        ("--reward-quality-deficit-weight", settings.reward_quality_deficit_weight),
        ("--reward-delay-excess-weight", settings.reward_delay_excess_weight),
        ("--reward-virtual-queue-weight", settings.reward_virtual_queue_weight),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _select_policies(policy_names: Sequence[str], *, seed: int) -> tuple[SchedulingPolicy, ...]:
    available = {policy.name: policy for policy in default_policies(seed=seed)}
    unknown = [name for name in policy_names if name not in available]
    if unknown:
        raise ValueError(f"unknown policies {unknown}; available: {sorted(available)}")
    return tuple(available[name] for name in policy_names)


def _ppo_hyperparameters(
    total_timesteps: int,
    *,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    learning_rate: float | None = None,
    gamma: float | None = None,
    ent_coef: float | None = None,
) -> dict[str, object]:
    resolved_n_steps = n_steps or min(128, max(16, total_timesteps // 4))
    resolved_batch_size = batch_size or min(64, resolved_n_steps)
    return {
        "n_steps": resolved_n_steps,
        "batch_size": resolved_batch_size,
        "n_epochs": n_epochs or 4,
        "gamma": 0.99 if gamma is None else gamma,
        "learning_rate": 3e-4 if learning_rate is None else learning_rate,
        "ent_coef": 0.0 if ent_coef is None else ent_coef,
        "device": "cpu",
    }


def _settings_ppo_hyperparameters(settings: PpoSmokeSettings) -> dict[str, object]:
    return _ppo_hyperparameters(
        settings.total_timesteps,
        n_steps=settings.ppo_n_steps,
        batch_size=settings.ppo_batch_size,
        n_epochs=settings.ppo_n_epochs,
        learning_rate=settings.ppo_learning_rate,
        gamma=settings.ppo_gamma,
        ent_coef=settings.ppo_ent_coef,
    )


def _runtime_estimate(total_timesteps: int, limit_rois: int) -> str:
    """Conservative user-facing estimate printed before training starts."""

    if total_timesteps <= 1000 or limit_rois <= 256:
        return "about 1 minute or less"
    return "about 2-8 minutes"


def _make_env(
    profiles: ActionProfileTable,
    settings: PpoSmokeSettings,
    *,
    slot_count: int,
    scenario_name: str | None = None,
) -> LeoSchedulingEnv:
    bundle = build_scenario(
        scenario_name or settings.scenario,
        base_config=settings.base_config,
        node_configs=settings.node_configs,
        slot_count=slot_count,
        bandwidth_hz=settings.bandwidth_hz,
    )
    return LeoSchedulingEnv(
        profiles,
        config=bundle.config,
        node_configs=bundle.node_configs,
        link_trace=bundle.link_trace,
        reward_quality_deficit_weight=settings.reward_quality_deficit_weight,
        reward_delay_excess_weight=settings.reward_delay_excess_weight,
        reward_virtual_queue_weight=settings.reward_virtual_queue_weight,
    )


class PpoDiagnosticsCallback(BaseCallback):
    """Write evaluation rows and checkpoints during one PPO smoke training run."""

    def __init__(
        self,
        profiles: ActionProfileTable,
        settings: PpoSmokeSettings,
        *,
        slot_count: int,
        scenario_name: str,
    ) -> None:
        super().__init__(verbose=0)
        self.profiles = profiles
        self.settings = settings
        self.slot_count = slot_count
        self.scenario_name = scenario_name
        self.training_curve_rows: list[dict[str, object]] = []
        self.checkpoint_paths: list[Path] = []
        self.best_model_path: Path | None = None
        self.best_timestep: int | None = None
        self.best_qoe_total: float | None = None
        self._next_eval_timestep = settings.eval_frequency if settings.eval_frequency > 0 else None
        self._next_checkpoint_timestep = (
            settings.checkpoint_frequency if settings.checkpoint_frequency > 0 else None
        )

    def _on_step(self) -> bool:
        timestep = int(self.num_timesteps)
        if self._next_eval_timestep is not None and timestep >= self._next_eval_timestep:
            self._record_evaluation(timestep)
            self._next_eval_timestep += self.settings.eval_frequency
        if self._next_checkpoint_timestep is not None and timestep >= self._next_checkpoint_timestep:
            self._save_checkpoint(timestep)
            self._next_checkpoint_timestep += self.settings.checkpoint_frequency
        return True

    def _on_training_end(self) -> None:
        if self.settings.eval_frequency > 0:
            timestep = int(self.num_timesteps)
            if not self.training_curve_rows or int(self.training_curve_rows[-1]["timestep"]) != timestep:
                self._record_evaluation(timestep)
        if self.training_curve_rows:
            pd.DataFrame(self.training_curve_rows).to_csv(
                self.settings.output_dir / "training_curve.csv",
                index=False,
            )

    def _record_evaluation(self, timestep: int) -> None:
        eval_env = _make_env(
            self.profiles,
            self.settings,
            slot_count=self.slot_count,
            scenario_name=self.scenario_name,
        )
        result = _evaluate_ppo(self.model, eval_env)
        metrics = dict(result.metrics)
        qoe_total = float(metrics.get("qoe_total", 0.0))
        success_rate = float(metrics.get("success_rate", 0.0))
        best_eligible = success_rate >= self.settings.best_checkpoint_min_success_rate
        is_best = best_eligible and (self.best_qoe_total is None or qoe_total > self.best_qoe_total)
        row = {
            "seed": self.settings.seed,
            "scenario": self.scenario_name,
            "timestep": timestep,
            "qoe_total": qoe_total,
            "training_reward_total": eval_env.training_reward_total,
            "success_rate": success_rate,
            "semantic_success_rate": metrics.get("semantic_success_rate", 0.0),
            "mean_delay_ms": metrics.get("mean_delay_ms", 0.0),
            "total_energy_j": metrics.get("total_energy_j", 0.0),
            "total_aosi_cost": metrics.get("total_aosi_cost", 0.0),
            "quality_virtual_queue": eval_env.quality_virtual_queue,
            "delay_virtual_queue_ms": eval_env.delay_virtual_queue_ms,
            "local_count": metrics.get("local_count", 0),
            "offload_count": metrics.get("offload_count", 0),
            "drop_count": metrics.get("drop_count", 0),
            "illegal_count": metrics.get("illegal_count", 0),
            "timeout_count": metrics.get("timeout_count", 0),
            "quality_violation_count": metrics.get("quality_violation_count", 0),
            "best_checkpoint_eligible": best_eligible,
            "selected_best_checkpoint": is_best,
        }
        if is_best:
            for existing in self.training_curve_rows:
                existing["selected_best_checkpoint"] = False
            self.best_qoe_total = qoe_total
            self.best_timestep = timestep
            self.best_model_path = self.settings.output_dir / "best_model.zip"
            self.model.save(self.best_model_path)
        self.training_curve_rows.append(row)
        pd.DataFrame(self.training_curve_rows).to_csv(
            self.settings.output_dir / "training_curve.csv",
            index=False,
        )

    def _save_checkpoint(self, timestep: int) -> None:
        checkpoint_dir = self.settings.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"step_{timestep}.zip"
        self.model.save(checkpoint_path)
        self.checkpoint_paths.append(checkpoint_path)


def _train_or_load_model(
    profiles: ActionProfileTable,
    settings: PpoSmokeSettings,
    *,
    slot_count: int,
    scenario_name: str,
) -> tuple[
    PPO,
    float,
    Mapping[str, object],
    Path,
    list[dict[str, object]],
    list[Path],
    dict[str, object],
]:
    hyperparameters = _settings_ppo_hyperparameters(settings)
    env = _make_env(profiles, settings, slot_count=slot_count, scenario_name=scenario_name)
    if settings.eval_only:
        model_path = settings.model_path or settings.output_dir / "model.zip"
        model = PPO.load(model_path, env=env, device="cpu")
        model_selection = {
            "mode": "eval_only",
            "select_best_checkpoint": settings.select_best_checkpoint,
            "best_checkpoint_min_success_rate": settings.best_checkpoint_min_success_rate,
            "selected_model": str(model_path.resolve()),
            "selected_timestep": None,
            "selected_qoe_total": None,
            "best_model": None,
            "best_timestep": None,
            "best_qoe_total": None,
            "final_model": None,
        }
        return model, 0.0, hyperparameters, model_path, [], [], model_selection

    model = PPO(
        "MlpPolicy",
        env,
        seed=settings.seed,
        verbose=0,
        **hyperparameters,
    )
    diagnostics = PpoDiagnosticsCallback(
        profiles,
        settings,
        slot_count=slot_count,
        scenario_name=scenario_name,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=settings.total_timesteps, callback=diagnostics)
    train_seconds = time.perf_counter() - started
    final_model_path = settings.output_dir / "final_model.zip"
    model.save(final_model_path)
    selected_model_path = settings.output_dir / "model.zip"
    selected_timestep = settings.total_timesteps
    selected_qoe_total: float | None = None
    selection_mode = "final"
    if settings.select_best_checkpoint and diagnostics.best_model_path is not None:
        model = PPO.load(diagnostics.best_model_path, env=env, device="cpu")
        selected_timestep = int(diagnostics.best_timestep or settings.total_timesteps)
        selected_qoe_total = diagnostics.best_qoe_total
        selection_mode = "best_checkpoint"
    elif settings.select_best_checkpoint:
        selection_mode = "final_no_eligible_best_checkpoint"
    model.save(selected_model_path)
    model_selection = {
        "mode": selection_mode,
        "select_best_checkpoint": settings.select_best_checkpoint,
        "best_checkpoint_min_success_rate": settings.best_checkpoint_min_success_rate,
        "selected_model": str(selected_model_path.resolve()),
        "selected_timestep": selected_timestep,
        "selected_qoe_total": selected_qoe_total,
        "best_model": (
            None if diagnostics.best_model_path is None else str(diagnostics.best_model_path.resolve())
        ),
        "best_timestep": diagnostics.best_timestep,
        "best_qoe_total": diagnostics.best_qoe_total,
        "final_model": str(final_model_path.resolve()),
    }
    return (
        model,
        train_seconds,
        hyperparameters,
        selected_model_path,
        diagnostics.training_curve_rows,
        diagnostics.checkpoint_paths,
        model_selection,
    )


def _evaluate_ppo(model: PPO, env: LeoSchedulingEnv) -> PolicyRunResult:
    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
    return env.as_policy_run_result(policy_name="Proposed-RL")


def _write_outputs(
    settings: PpoSmokeSettings,
    *,
    train_seconds: float,
    model_path: Path,
    ppo_result: PolicyRunResult,
    baseline_results: Sequence[PolicyRunResult],
    hyperparameters: Mapping[str, object],
    scenario_name: str,
    training_curve_rows: Sequence[Mapping[str, object]],
    checkpoint_paths: Sequence[Path],
    model_selection: Mapping[str, object],
) -> dict[str, object]:
    all_results = [ppo_result, *baseline_results]
    metrics_rows = [result.metrics for result in all_results]
    pd.DataFrame([decision.as_row() for decision in ppo_result.decisions]).to_csv(
        settings.output_dir / "eval_decisions.csv",
        index=False,
    )
    (settings.output_dir / "eval_metrics.json").write_text(
        json.dumps(ppo_result.metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(metrics_rows).to_csv(settings.output_dir / "comparison_metrics.csv", index=False)
    _write_comparison_report(settings.output_dir / "comparison_report.md", metrics_rows)
    training_curve_path = settings.output_dir / "training_curve.csv"
    if training_curve_rows and not training_curve_path.is_file():
        pd.DataFrame(training_curve_rows).to_csv(training_curve_path, index=False)
    training_figures = _write_training_diagnostic_figures(settings.output_dir, training_curve_rows)

    summary = {
        "purpose": "PPO smoke closure for the LEO semantic scheduling environment; not a formal paper result.",
        "config_file": str(settings.config_path.resolve()) if settings.config_path else None,
        "profile_csv": str(settings.profile_csv.resolve()),
        "quality_proxy": str(settings.quality_proxy.resolve()),
        "output_dir": str(settings.output_dir.resolve()),
        "configuration": {
            "limit_rois": settings.limit_rois,
            "total_timesteps": settings.total_timesteps,
            "scenario": scenario_name,
            "eval_only": settings.eval_only,
            "seed": settings.seed,
            "rois_per_slot": settings.base_config.rois_per_slot,
            "deadline_ms": settings.base_config.deadline_ms,
            "quality_threshold": settings.base_config.quality_threshold,
            "bandwidth_hz": settings.bandwidth_hz,
            "nodes": [asdict(node) for node in settings.node_configs],
            "ppo_hyperparameters": dict(hyperparameters),
            "eval_frequency": settings.eval_frequency,
            "checkpoint_frequency": settings.checkpoint_frequency,
            "select_best_checkpoint": settings.select_best_checkpoint,
            "best_checkpoint_min_success_rate": settings.best_checkpoint_min_success_rate,
            "reward_shaping": {
                "quality_deficit_weight": settings.reward_quality_deficit_weight,
                "delay_excess_weight": settings.reward_delay_excess_weight,
                "virtual_queue_weight": settings.reward_virtual_queue_weight,
            },
        },
        "runtime_seconds": train_seconds,
        "training_reward_total": ppo_result.metrics.get("training_reward_total", 0.0),
        "training_curve_rows": len(training_curve_rows),
        "checkpoint_count": len(checkpoint_paths),
        "model_selection": dict(model_selection),
        "policy_metrics": metrics_rows,
        "outputs": {
            "model": str(model_path.resolve()),
            "eval_decisions": str((settings.output_dir / "eval_decisions.csv").resolve()),
            "eval_metrics": str((settings.output_dir / "eval_metrics.json").resolve()),
            "comparison_metrics": str((settings.output_dir / "comparison_metrics.csv").resolve()),
            "comparison_report": str((settings.output_dir / "comparison_report.md").resolve()),
            "summary": str((settings.output_dir / "summary.json").resolve()),
            "training_curve": str(training_curve_path.resolve()) if training_curve_path.is_file() else None,
            "checkpoints": [str(path.resolve()) for path in checkpoint_paths],
            "training_figures": {key: str(path.resolve()) for key, path in training_figures.items()},
        },
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _write_training_diagnostic_figures(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    if not rows:
        return {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping training diagnostic figures because matplotlib is unavailable: {exc}")
        return {}

    frame = pd.DataFrame(rows).sort_values("timestep")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "training_curve_qoe": figure_dir / "training_curve_qoe.png",
        "training_curve_actions": figure_dir / "training_curve_actions.png",
        "training_curve_violations": figure_dir / "training_curve_violations.png",
        "training_curve_queues": figure_dir / "training_curve_queues.png",
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frame["timestep"], frame["qoe_total"], marker="o", label="canonical QoE")
    if "selected_best_checkpoint" in frame:
        selected = frame[frame["selected_best_checkpoint"].astype(bool)]
    else:
        selected = frame.iloc[0:0]
    if not selected.empty:
        ax.scatter(selected["timestep"], selected["qoe_total"], marker="*", s=140, label="selected best")
    ax.set_xlabel("PPO timesteps")
    ax.set_ylabel("Canonical QoE")
    ax.set_title("Evaluation QoE During Training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures["training_curve_qoe"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for column, label in (
        ("local_count", "local"),
        ("offload_count", "offload"),
        ("drop_count", "drop"),
        ("illegal_count", "illegal"),
    ):
        if column in frame:
            ax.plot(frame["timestep"], frame[column], marker="o", label=label)
    ax.set_xlabel("PPO timesteps")
    ax.set_ylabel("Count per evaluation")
    ax.set_title("Action Mix During Training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures["training_curve_actions"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for column, label in (
        ("illegal_count", "illegal"),
        ("timeout_count", "timeout"),
        ("quality_violation_count", "quality violation"),
    ):
        if column in frame:
            ax.plot(frame["timestep"], frame[column], marker="o", label=label)
    ax.set_xlabel("PPO timesteps")
    ax.set_ylabel("Count per evaluation")
    ax.set_title("Constraint Violations During Training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures["training_curve_violations"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame["timestep"], frame["quality_virtual_queue"], marker="o", color="#4c78a8")
    axes[0].set_xlabel("PPO timesteps")
    axes[0].set_ylabel("Quality queue")
    axes[0].set_title("Quality Virtual Queue")
    axes[1].plot(frame["timestep"], frame["delay_virtual_queue_ms"], marker="o", color="#f58518")
    axes[1].set_xlabel("PPO timesteps")
    axes[1].set_ylabel("Delay queue ms")
    axes[1].set_title("Delay Virtual Queue")
    fig.tight_layout()
    fig.savefig(figures["training_curve_queues"], dpi=180)
    plt.close(fig)
    return figures


def _write_all_scenarios_outputs(
    output_dir: Path,
    summaries: Sequence[dict[str, object]],
) -> dict[str, object]:
    flattened_rows: list[dict[str, object]] = []
    for summary in summaries:
        scenario_name = str(summary["configuration"]["scenario"])
        for row in summary["policy_metrics"]:
            item = dict(row)
            item["scenario"] = scenario_name
            flattened_rows.append(item)

    aggregate_views = _build_multi_scenario_aggregate(flattened_rows)
    ranked_rows = aggregate_views["ranked_policy_metrics"]
    scenario_summaries = aggregate_views["scenario_summaries"]
    policy_aggregate = aggregate_views["policy_aggregate"]

    pd.DataFrame(ranked_rows).to_csv(output_dir / "comparison_metrics_all.csv", index=False)
    pd.DataFrame(scenario_summaries).to_csv(output_dir / "scenario_winners.csv", index=False)
    pd.DataFrame(policy_aggregate).to_csv(output_dir / "policy_aggregate.csv", index=False)
    report_path = output_dir / "comparison_report.md"
    lines = [
        "# PPO Smoke Multi-Scenario Comparison",
        "",
        "All rows use the canonical slot QoE metric; PPO training reward is reported separately in each scenario summary.",
        "",
        "## Scenario Winners",
        "",
        _markdown_table_with_columns(scenario_summaries, SCENARIO_SUMMARY_COLUMNS),
        "",
        "## Policy Averages",
        "",
        _markdown_table_with_columns(policy_aggregate, POLICY_AGGREGATE_COLUMNS),
        "",
        "## Full Policy Metrics",
        "",
        _markdown_table(ranked_rows, include_scenario=True),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    aggregate = {
        "purpose": "PPO smoke multi-scenario summary; not a formal paper result.",
        "output_dir": str(output_dir.resolve()),
        "scenarios": list(summaries),
        "policy_metrics": ranked_rows,
        "scenario_summaries": scenario_summaries,
        "policy_aggregate": policy_aggregate,
        "outputs": {
            "comparison_metrics_all": str((output_dir / "comparison_metrics_all.csv").resolve()),
            "scenario_winners": str((output_dir / "scenario_winners.csv").resolve()),
            "policy_aggregate": str((output_dir / "policy_aggregate.csv").resolve()),
            "comparison_report": str(report_path.resolve()),
            "summary_all": str((output_dir / "summary_all.json").resolve()),
        },
    }
    (output_dir / "summary_all.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return aggregate


def _build_multi_scenario_aggregate(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Build report-only rankings from canonical QoE without changing formulas."""

    ranked_rows: list[dict[str, object]] = []
    scenario_summaries: list[dict[str, object]] = []
    rows_by_scenario: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        rows_by_scenario.setdefault(str(row["scenario"]), []).append(row)

    for scenario_name in rows_by_scenario:
        scenario_rows = sorted(
            rows_by_scenario[scenario_name],
            key=lambda row: (-float(row["qoe_total"]), str(row["policy"])),
        )
        best_row = scenario_rows[0]
        best_qoe = float(best_row["qoe_total"])
        proposed_row: dict[str, object] | None = None
        for rank, row in enumerate(scenario_rows, start=1):
            ranked_row = dict(row)
            ranked_row["scenario_rank"] = rank
            ranked_row["is_best_policy"] = rank == 1
            ranked_row["qoe_gap_to_best"] = float(row["qoe_total"]) - best_qoe
            ranked_rows.append(ranked_row)
            if ranked_row["policy"] == "Proposed-RL":
                proposed_row = ranked_row

        scenario_summaries.append(
            {
                "scenario": scenario_name,
                "best_policy": best_row["policy"],
                "best_qoe": best_qoe,
                "proposed_rl_qoe": None if proposed_row is None else float(proposed_row["qoe_total"]),
                "proposed_rl_rank": None if proposed_row is None else int(proposed_row["scenario_rank"]),
                "proposed_rl_gap_to_best_qoe": (
                    None if proposed_row is None else float(proposed_row["qoe_gap_to_best"])
                ),
            }
        )

    policy_aggregate = _build_policy_aggregate(ranked_rows)
    return {
        "ranked_policy_metrics": ranked_rows,
        "scenario_summaries": scenario_summaries,
        "policy_aggregate": policy_aggregate,
    }


def _build_policy_aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows_by_policy: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        rows_by_policy.setdefault(str(row["policy"]), []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for policy_name, policy_rows in rows_by_policy.items():
        aggregate_rows.append(
            {
                "policy": policy_name,
                "scenario_count": len(policy_rows),
                "mean_rank": _mean_metric(policy_rows, "scenario_rank"),
                "best_scenario_count": sum(1 for row in policy_rows if bool(row["is_best_policy"])),
                "mean_qoe": _mean_metric(policy_rows, "qoe_total"),
                "mean_qoe_gap_to_best": _mean_metric(policy_rows, "qoe_gap_to_best"),
                "mean_success_rate": _mean_metric(policy_rows, "success_rate"),
                "mean_semantic_success_rate": _mean_metric(policy_rows, "semantic_success_rate"),
                "mean_delay_ms": _mean_metric(policy_rows, "mean_delay_ms"),
                "mean_total_energy_j": _mean_metric(policy_rows, "total_energy_j"),
                "mean_total_aosi_cost": _mean_metric(policy_rows, "total_aosi_cost"),
            }
        )
    return sorted(aggregate_rows, key=lambda row: (-float(row["mean_qoe"]), str(row["policy"])))


def _mean_metric(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _write_comparison_report(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        "# PPO Smoke Scheduling Comparison",
        "",
        "This smoke report compares Proposed-RL with deterministic baselines on the same ROI stream and synthetic link trace.",
        "",
        _markdown_table(rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(rows: Sequence[Mapping[str, object]], *, include_scenario: bool = False) -> str:
    columns = (("scenario", "Scenario"),) + REPORT_COLUMNS if include_scenario else REPORT_COLUMNS
    return _markdown_table_with_columns(rows, columns)


def _markdown_table_with_columns(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[tuple[str, str]],
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_report_value(key, row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_report_value(key: str, value: object) -> str:
    if value in {"", None}:
        return ""
    if key in {"policy", "scenario", "best_policy"}:
        return str(value)
    if key in {"scenario_count", "best_scenario_count", "proposed_rl_rank", "scenario_rank"}:
        return str(int(float(value)))
    if key == "total_compressed_bytes":
        return str(int(float(value)))
    return f"{float(value):.4f}"


def main() -> int:
    args = parse_args()
    settings = _resolve_settings(args)
    _validate_settings(settings)
    if not settings.eval_only:
        _prepare_output_dir(settings.output_dir, exist_ok=settings.exist_ok)
    else:
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    if settings.eval_only:
        print("PPO eval-only run: no training will be performed.")
    else:
        print(
            "Estimated PPO smoke runtime before training: "
            f"{_runtime_estimate(settings.total_timesteps, settings.limit_rois)} "
            f"for {settings.total_timesteps} timesteps and {settings.limit_rois} ROI."
        )
    profiles = ActionProfileTable.from_csv(
        settings.profile_csv,
        quality_proxy_path=settings.quality_proxy,
        limit_rois=settings.limit_rois,
    )
    slot_count = (len(profiles.roi_order) + settings.base_config.rois_per_slot - 1) // (
        settings.base_config.rois_per_slot
    )
    scenario_names = SCENARIO_NAMES if settings.all_scenarios else (settings.scenario,)
    summaries: list[dict[str, object]] = []
    for scenario_name in scenario_names:
        scenario_output_dir = settings.output_dir / scenario_name if settings.all_scenarios else settings.output_dir
        scenario_output_dir.mkdir(parents=True, exist_ok=True)
        scenario_settings = replace(settings, output_dir=scenario_output_dir, scenario=scenario_name)
        (
            model,
            train_seconds,
            hyperparameters,
            model_path,
            training_curve_rows,
            checkpoint_paths,
            model_selection,
        ) = _train_or_load_model(
            profiles,
            scenario_settings,
            slot_count=slot_count,
            scenario_name=scenario_name,
        )
        eval_env = _make_env(
            profiles,
            scenario_settings,
            slot_count=slot_count,
            scenario_name=scenario_name,
        )
        ppo_result = _evaluate_ppo(model, eval_env)
        ppo_result.metrics["training_reward_total"] = eval_env.training_reward_total
        bundle = build_scenario(
            scenario_name,
            base_config=scenario_settings.base_config,
            node_configs=scenario_settings.node_configs,
            slot_count=slot_count,
            bandwidth_hz=scenario_settings.bandwidth_hz,
        )
        baseline_results = run_baseline_suite(
            profiles,
            config=bundle.config,
            node_configs=bundle.node_configs,
            policies=_select_policies(scenario_settings.policy_names, seed=scenario_settings.seed),
            link_trace=bundle.link_trace,
        )
        summaries.append(
            _write_outputs(
                scenario_settings,
                train_seconds=train_seconds,
                model_path=model_path,
                ppo_result=ppo_result,
                baseline_results=baseline_results,
                hyperparameters=hyperparameters,
                scenario_name=scenario_name,
                training_curve_rows=training_curve_rows,
                checkpoint_paths=checkpoint_paths,
                model_selection=model_selection,
            )
        )

    if settings.all_scenarios:
        summary = _write_all_scenarios_outputs(settings.output_dir, summaries)
    else:
        summary = summaries[0]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
