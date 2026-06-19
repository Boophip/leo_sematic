"""Run configurable stage-three deterministic satellite scheduling scenarios."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.episode import (  # noqa: E402
    DEFAULT_LINK_BANDWIDTH_HZ,
    SCENARIO_NAMES,
    ActionProfileTable,
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
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "simulation" / "deterministic"
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
class ResolvedSettings:
    """YAML plus CLI configuration resolved into one executable run contract."""

    config_path: Path | None
    config_loaded: bool
    profile_csv: Path
    quality_proxy: Path
    output_dir: Path
    limit_rois: int | None
    seed: int
    use_oracle_quality: bool
    scenario: str
    all_scenarios: bool
    exist_ok: bool
    bandwidth_hz: float
    base_config: SimulationConfig
    node_configs: tuple[SatelliteNodeConfig, ...]
    policy_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    """Collect config, scenario, and deterministic simulation overrides."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profile-csv", type=Path)
    parser.add_argument("--quality-proxy", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-rois", type=int)
    parser.add_argument("--rois-per-slot", type=int)
    parser.add_argument("--deadline-ms", type=float)
    parser.add_argument("--quality-threshold", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-oracle-quality", action="store_true")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _load_yaml_config(path: Path) -> tuple[dict[str, object], bool]:
    """Load YAML if present; the default path may be absent for legacy runs."""

    resolved = _resolve_path(path)
    if not resolved.is_file():
        if resolved == DEFAULT_CONFIG_PATH:
            return {}, False
        raise FileNotFoundError(f"config file not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a mapping: {resolved}")
    return dict(payload), True


def _resolve_settings(args: argparse.Namespace) -> ResolvedSettings:
    payload, config_loaded = _load_yaml_config(args.config)
    paths = _section(payload, "paths")
    run = _section(payload, "run")
    link = _section(payload, "link")
    reward = _section(payload, "reward")
    aosi = _section(payload, "aosi")

    profile_csv = _resolve_path(args.profile_csv or paths.get("profile_csv") or DEFAULT_PROFILE_CSV)
    quality_proxy = _resolve_path(args.quality_proxy or paths.get("quality_proxy") or DEFAULT_QUALITY_PROXY)
    output_dir = _resolve_path(args.output_dir or paths.get("output_dir") or DEFAULT_OUTPUT_DIR)
    limit_rois = _optional_int(args.limit_rois if args.limit_rois is not None else run.get("limit_rois"))
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
    bandwidth_hz = float(link.get("bandwidth_hz", DEFAULT_LINK_BANDWIDTH_HZ))

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

    return ResolvedSettings(
        config_path=_resolve_path(args.config) if config_loaded else None,
        config_loaded=config_loaded,
        profile_csv=profile_csv,
        quality_proxy=quality_proxy,
        output_dir=output_dir,
        limit_rois=limit_rois,
        seed=seed,
        use_oracle_quality=bool(args.use_oracle_quality or run.get("use_oracle_quality", False)),
        scenario=scenario,
        all_scenarios=bool(args.all_scenarios),
        exist_ok=bool(args.exist_ok),
        bandwidth_hz=bandwidth_hz,
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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


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


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    """Keep output directories reproducible unless the caller opts into overwrite."""

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


def _validate_inputs(settings: ResolvedSettings) -> None:
    if settings.limit_rois is not None and settings.limit_rois <= 0:
        raise ValueError("--limit-rois must be positive")
    if settings.base_config.rois_per_slot <= 0:
        raise ValueError("rois_per_slot must be positive")
    if not settings.profile_csv.is_file():
        raise FileNotFoundError(f"profile CSV not found: {settings.profile_csv}")
    if not settings.use_oracle_quality and not settings.quality_proxy.is_file():
        raise FileNotFoundError(
            f"quality proxy not found: {settings.quality_proxy}; "
            "use --use-oracle-quality for diagnostics"
        )


def _run_scenario(
    *,
    scenario_name: str,
    output_dir: Path,
    profiles: ActionProfileTable,
    settings: ResolvedSettings,
) -> dict[str, object]:
    slot_count = (len(profiles.roi_order) + settings.base_config.rois_per_slot - 1) // (
        settings.base_config.rois_per_slot
    )
    bundle = build_scenario(
        scenario_name,
        base_config=settings.base_config,
        node_configs=settings.node_configs,
        slot_count=slot_count,
        bandwidth_hz=settings.bandwidth_hz,
    )
    policies = _select_policies(settings.policy_names, seed=settings.seed)
    results = run_baseline_suite(
        profiles,
        config=bundle.config,
        node_configs=bundle.node_configs,
        policies=policies,
        link_trace=bundle.link_trace,
    )
    summary = _build_summary(
        scenario_name=scenario_name,
        output_dir=output_dir,
        profiles=profiles,
        settings=settings,
        config=bundle.config,
        nodes=bundle.node_configs,
        policies=policies,
        bandwidth_hz=bundle.bandwidth_hz,
        snr_offset_db=bundle.snr_offset_db,
        slot_count=slot_count,
        results=results,
    )
    _write_scenario_outputs(output_dir, results, summary)
    return summary


def _build_summary(
    *,
    scenario_name: str,
    output_dir: Path,
    profiles: ActionProfileTable,
    settings: ResolvedSettings,
    config: SimulationConfig,
    nodes: Sequence[SatelliteNodeConfig],
    policies: Sequence[SchedulingPolicy],
    bandwidth_hz: float,
    snr_offset_db: float,
    slot_count: int,
    results: Sequence[object],
) -> dict[str, object]:
    metrics_rows = [result.metrics for result in results]
    return {
        "purpose": "Stage-three deterministic satellite simulation; not an RL training run.",
        "scenario": scenario_name,
        "config_file": str(settings.config_path.resolve()) if settings.config_path else None,
        "profile_csv": str(settings.profile_csv.resolve()),
        "quality_source": (
            "oracle_task_quality"
            if settings.use_oracle_quality
            else str(settings.quality_proxy.resolve())
        ),
        "output_dir": str(output_dir.resolve()),
        "configuration": {
            "limit_rois": settings.limit_rois,
            "roi_count": len(profiles.roi_order),
            "slot_count": slot_count,
            "rois_per_slot": config.rois_per_slot,
            "deadline_ms": config.deadline_ms,
            "quality_threshold": config.quality_threshold,
            "seed": settings.seed,
            "bandwidth_hz": bandwidth_hz,
            "snr_offset_db": snr_offset_db,
            "nodes": [asdict(node) for node in nodes],
            "policies": [policy.name for policy in policies],
        },
        "policy_metrics": metrics_rows,
    }


def _write_scenario_outputs(
    output_dir: Path,
    results: Sequence[object],
    summary: dict[str, object],
) -> None:
    metrics_rows = [result.metrics for result in results]
    decision_rows = [
        decision.as_row()
        for result in results
        for decision in result.decisions
    ]
    slot_rows = [
        asdict(slot)
        for result in results
        for slot in result.slots
    ]
    pd.DataFrame(metrics_rows).to_csv(output_dir / "policy_metrics.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(output_dir / "decisions.csv", index=False)
    pd.DataFrame(slot_rows).to_csv(output_dir / "slot_metrics.csv", index=False)

    summary["outputs"] = {
        "policy_metrics": str((output_dir / "policy_metrics.csv").resolve()),
        "decisions": str((output_dir / "decisions.csv").resolve()),
        "slot_metrics": str((output_dir / "slot_metrics.csv").resolve()),
        "comparison_report": str((output_dir / "comparison_report.md").resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_comparison_report(
        output_dir / "comparison_report.md",
        title=f"Scenario: {summary['scenario']}",
        rows=metrics_rows,
    )


def _write_comparison_report(path: Path, *, title: str, rows: Sequence[Mapping[str, object]]) -> None:
    lines = [
        f"# Deterministic Simulation Comparison - {title}",
        "",
        "This report compares deterministic baselines on the same ROI stream, link trace, and metrics.",
        "",
        _markdown_table(rows),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_all_scenarios_outputs(
    output_dir: Path,
    summaries: Sequence[dict[str, object]],
) -> dict[str, object]:
    flattened_rows: list[dict[str, object]] = []
    for summary in summaries:
        for row in summary["policy_metrics"]:
            row_with_scenario = dict(row)
            row_with_scenario["scenario"] = summary["scenario"]
            flattened_rows.append(row_with_scenario)

    pd.DataFrame(flattened_rows).to_csv(output_dir / "policy_metrics_all.csv", index=False)
    report_path = output_dir / "comparison_report.md"
    lines = [
        "# Stage-Three Deterministic Simulation Comparison",
        "",
        "All scenarios use the same profiling table and policy list; only the configured stress preset changes.",
        "",
        _markdown_table(flattened_rows, include_scenario=True),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    aggregate = {
        "purpose": "Cross-scenario stage-three deterministic simulation summary.",
        "output_dir": str(output_dir.resolve()),
        "scenarios": list(summaries),
        "policy_metrics": flattened_rows,
        "outputs": {
            "policy_metrics_all": str((output_dir / "policy_metrics_all.csv").resolve()),
            "comparison_report": str(report_path.resolve()),
            "summary_all": str((output_dir / "summary_all.json").resolve()),
        },
    }
    (output_dir / "summary_all.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return aggregate


def _markdown_table(
    rows: Sequence[Mapping[str, object]],
    *,
    include_scenario: bool = False,
) -> str:
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
    _validate_inputs(settings)
    _prepare_output_dir(settings.output_dir, exist_ok=settings.exist_ok)

    profiles = ActionProfileTable.from_csv(
        settings.profile_csv,
        quality_proxy_path=None if settings.use_oracle_quality else settings.quality_proxy,
        limit_rois=settings.limit_rois,
        use_oracle_quality=settings.use_oracle_quality,
    )

    if settings.all_scenarios:
        summaries = []
        for scenario_name in SCENARIO_NAMES:
            scenario_dir = settings.output_dir / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            summaries.append(
                _run_scenario(
                    scenario_name=scenario_name,
                    output_dir=scenario_dir,
                    profiles=profiles,
                    settings=settings,
                )
            )
        aggregate = _write_all_scenarios_outputs(settings.output_dir, summaries)
        print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    else:
        summary = _run_scenario(
            scenario_name=settings.scenario,
            output_dir=settings.output_dir,
            profiles=profiles,
            settings=settings,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
