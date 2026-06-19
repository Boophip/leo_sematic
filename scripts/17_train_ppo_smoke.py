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


def _ppo_hyperparameters(total_timesteps: int) -> dict[str, object]:
    n_steps = min(128, max(16, total_timesteps // 4))
    batch_size = min(64, n_steps)
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 4,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "device": "cpu",
    }


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
    )


def _train_or_load_model(
    profiles: ActionProfileTable,
    settings: PpoSmokeSettings,
    *,
    slot_count: int,
    scenario_name: str,
) -> tuple[PPO, float, Mapping[str, object], Path]:
    hyperparameters = _ppo_hyperparameters(settings.total_timesteps)
    env = _make_env(profiles, settings, slot_count=slot_count, scenario_name=scenario_name)
    if settings.eval_only:
        model_path = settings.model_path or settings.output_dir / "model.zip"
        model = PPO.load(model_path, env=env, device="cpu")
        return model, 0.0, hyperparameters, model_path

    model = PPO(
        "MlpPolicy",
        env,
        seed=settings.seed,
        verbose=0,
        **hyperparameters,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=settings.total_timesteps)
    train_seconds = time.perf_counter() - started
    model.save(settings.output_dir / "model")
    return model, train_seconds, hyperparameters, settings.output_dir / "model.zip"


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
        },
        "runtime_seconds": train_seconds,
        "training_reward_total": ppo_result.metrics.get("training_reward_total", 0.0),
        "policy_metrics": metrics_rows,
        "outputs": {
            "model": str(model_path.resolve()),
            "eval_decisions": str((settings.output_dir / "eval_decisions.csv").resolve()),
            "eval_metrics": str((settings.output_dir / "eval_metrics.json").resolve()),
            "comparison_metrics": str((settings.output_dir / "comparison_metrics.csv").resolve()),
            "comparison_report": str((settings.output_dir / "comparison_report.md").resolve()),
            "summary": str((settings.output_dir / "summary.json").resolve()),
        },
    }
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


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

    pd.DataFrame(flattened_rows).to_csv(output_dir / "comparison_metrics_all.csv", index=False)
    report_path = output_dir / "comparison_report.md"
    lines = [
        "# PPO Smoke Multi-Scenario Comparison",
        "",
        "All rows use the canonical slot QoE metric; PPO training reward is reported separately in each scenario summary.",
        "",
        _markdown_table(flattened_rows, include_scenario=True),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    aggregate = {
        "purpose": "PPO smoke multi-scenario summary; not a formal paper result.",
        "output_dir": str(output_dir.resolve()),
        "scenarios": list(summaries),
        "policy_metrics": flattened_rows,
        "outputs": {
            "comparison_metrics_all": str((output_dir / "comparison_metrics_all.csv").resolve()),
            "comparison_report": str(report_path.resolve()),
            "summary_all": str((output_dir / "summary_all.json").resolve()),
        },
    }
    (output_dir / "summary_all.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return aggregate


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
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_report_value(key, row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_report_value(key: str, value: object) -> str:
    if key in {"policy", "scenario"}:
        return str(value)
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
        model, train_seconds, hyperparameters, model_path = _train_or_load_model(
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
