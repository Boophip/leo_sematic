"""Run multi-seed PPO scheduling experiments and aggregate formal reports."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.episode import SCENARIO_NAMES  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "rl" / "ppo_formal"
PPO_SCRIPT = ROOT / "scripts" / "17_train_ppo_smoke.py"
POLICY_FORMAL_COLUMNS = (
    ("policy", "Policy"),
    ("sample_count", "Samples"),
    ("seed_count", "Seeds"),
    ("scenario_count", "Scenarios"),
    ("mean_rank", "Mean Rank"),
    ("std_rank", "Std Rank"),
    ("best_count", "Best Count"),
    ("mean_qoe", "Mean QoE"),
    ("std_qoe", "Std QoE"),
    ("mean_qoe_gap_to_best", "Mean Gap"),
    ("mean_success_rate", "Mean Success"),
    ("std_success_rate", "Std Success"),
    ("mean_delay_ms", "Mean Delay ms"),
    ("mean_total_energy_j", "Mean Energy J"),
    ("mean_total_aosi_cost", "Mean AoSI Cost"),
)
SCENARIO_FORMAL_COLUMNS = (
    ("scenario", "Scenario"),
    ("best_policy", "Best Policy"),
    ("best_mean_qoe", "Best Mean QoE"),
    ("proposed_rl_mean_qoe", "PPO Mean QoE"),
    ("proposed_rl_mean_rank", "PPO Mean Rank"),
    ("proposed_rl_mean_gap_to_best", "PPO Mean Gap"),
)


@dataclass(frozen=True)
class FormalExperimentSettings:
    """Resolved contract for the formal multi-seed PPO experiment runner."""

    seeds: tuple[int, ...]
    output_dir: Path
    limit_rois: int
    total_timesteps: int
    python_executable: Path
    config: Path | None
    profile_csv: Path | None
    quality_proxy: Path | None
    rois_per_slot: int | None
    deadline_ms: float | None
    quality_threshold: float | None
    exist_ok: bool
    dry_run: bool
    skip_plots: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-rois", type=int, default=512)
    parser.add_argument("--total-timesteps", type=int, default=5000)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile-csv", type=Path)
    parser.add_argument("--quality-proxy", type=Path)
    parser.add_argument("--rois-per-slot", type=int)
    parser.add_argument("--deadline-ms", type=float)
    parser.add_argument("--quality-threshold", type=float)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def _resolve_settings(args: argparse.Namespace) -> FormalExperimentSettings:
    return FormalExperimentSettings(
        seeds=tuple(dict.fromkeys(int(seed) for seed in args.seeds)),
        output_dir=_resolve_path(args.output_dir),
        limit_rois=int(args.limit_rois),
        total_timesteps=int(args.total_timesteps),
        python_executable=_resolve_path(args.python),
        config=_resolve_path(args.config) if args.config is not None else None,
        profile_csv=_resolve_path(args.profile_csv) if args.profile_csv is not None else None,
        quality_proxy=_resolve_path(args.quality_proxy) if args.quality_proxy is not None else None,
        rois_per_slot=args.rois_per_slot,
        deadline_ms=args.deadline_ms,
        quality_threshold=args.quality_threshold,
        exist_ok=bool(args.exist_ok),
        dry_run=bool(args.dry_run),
        skip_plots=bool(args.skip_plots),
    )


def _resolve_path(value: object) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _validate_settings(settings: FormalExperimentSettings) -> None:
    if not settings.seeds:
        raise ValueError("--seeds must contain at least one seed")
    if settings.limit_rois <= 0:
        raise ValueError("--limit-rois must be positive")
    if settings.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if not settings.python_executable.is_file():
        raise FileNotFoundError(f"python executable not found: {settings.python_executable}")
    if not PPO_SCRIPT.is_file():
        raise FileNotFoundError(f"PPO runner not found: {PPO_SCRIPT}")
    for path in (settings.config, settings.profile_csv, settings.quality_proxy):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"input file not found: {path}")


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _runtime_estimate(settings: FormalExperimentSettings) -> str:
    scenario_runs = len(settings.seeds) * len(SCENARIO_NAMES)
    if settings.total_timesteps <= 1000 or settings.limit_rois <= 256:
        low = max(1, scenario_runs // 3)
        high = max(2, scenario_runs)
        return f"about {low}-{high} minutes"
    low = scenario_runs * 2
    high = scenario_runs * 8
    return f"about {low}-{high} minutes"


def _seed_output_dir(output_dir: Path, seed: int) -> Path:
    return output_dir / f"seed_{seed}"


def _build_seed_command(settings: FormalExperimentSettings, seed: int) -> list[str]:
    seed_dir = _seed_output_dir(settings.output_dir, seed)
    command = [
        str(settings.python_executable),
        str(PPO_SCRIPT),
        "--output-dir",
        str(seed_dir),
        "--limit-rois",
        str(settings.limit_rois),
        "--total-timesteps",
        str(settings.total_timesteps),
        "--seed",
        str(seed),
        "--all-scenarios",
        "--exist-ok",
    ]
    optional_args: list[tuple[str, object | None]] = [
        ("--config", settings.config),
        ("--profile-csv", settings.profile_csv),
        ("--quality-proxy", settings.quality_proxy),
        ("--rois-per-slot", settings.rois_per_slot),
        ("--deadline-ms", settings.deadline_ms),
        ("--quality-threshold", settings.quality_threshold),
    ]
    for flag, value in optional_args:
        if value is not None:
            command.extend([flag, str(value)])
    return command


def _run_seed(settings: FormalExperimentSettings, seed: int) -> dict[str, object]:
    command = _build_seed_command(settings, seed)
    print(f"Running PPO formal seed {seed}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    summary_path = _seed_output_dir(settings.output_dir, seed) / "summary_all.json"
    return {
        "seed": seed,
        "summary_path": str(summary_path.resolve()),
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
    }


def _build_formal_aggregate(seed_results: Sequence[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    policy_rows: list[dict[str, object]] = []
    scenario_winner_rows: list[dict[str, object]] = []
    for seed_result in seed_results:
        seed = int(seed_result["seed"])
        summary = seed_result["summary"]
        for row in summary["policy_metrics"]:
            item = dict(row)
            item["seed"] = seed
            policy_rows.append(item)
        for row in summary["scenario_summaries"]:
            item = dict(row)
            item["seed"] = seed
            scenario_winner_rows.append(item)

    scenario_policy_aggregate = _aggregate_by_keys(policy_rows, ("scenario", "policy"))
    policy_formal_aggregate = _aggregate_by_keys(policy_rows, ("policy",))
    scenario_formal_summary = _build_scenario_formal_summary(scenario_policy_aggregate)
    return {
        "policy_seed_scenario_metrics": policy_rows,
        "scenario_seed_winners": scenario_winner_rows,
        "scenario_policy_aggregate": scenario_policy_aggregate,
        "policy_formal_aggregate": policy_formal_aggregate,
        "scenario_formal_summary": scenario_formal_summary,
    }


def _aggregate_by_keys(
    rows: Sequence[Mapping[str, object]],
    keys: Sequence[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for group_key, group_rows in grouped.items():
        item = {key: value for key, value in zip(keys, group_key)}
        item.update(
            {
                "sample_count": len(group_rows),
                "seed_count": len({int(row["seed"]) for row in group_rows}),
                "scenario_count": len({str(row["scenario"]) for row in group_rows}),
                "mean_rank": _mean_metric(group_rows, "scenario_rank"),
                "std_rank": _std_metric(group_rows, "scenario_rank"),
                "best_count": sum(1 for row in group_rows if bool(row["is_best_policy"])),
                "mean_qoe": _mean_metric(group_rows, "qoe_total"),
                "std_qoe": _std_metric(group_rows, "qoe_total"),
                "mean_qoe_gap_to_best": _mean_metric(group_rows, "qoe_gap_to_best"),
                "mean_success_rate": _mean_metric(group_rows, "success_rate"),
                "std_success_rate": _std_metric(group_rows, "success_rate"),
                "mean_semantic_success_rate": _mean_metric(group_rows, "semantic_success_rate"),
                "mean_delay_ms": _mean_metric(group_rows, "mean_delay_ms"),
                "mean_total_energy_j": _mean_metric(group_rows, "total_energy_j"),
                "mean_total_aosi_cost": _mean_metric(group_rows, "total_aosi_cost"),
            }
        )
        aggregate_rows.append(item)

    sort_keys = ("scenario", "policy") if "scenario" in keys else ("policy",)
    return sorted(
        aggregate_rows,
        key=lambda row: tuple(str(row[key]) for key in sort_keys[:-1]) + (-float(row["mean_qoe"]), str(row["policy"])),
    )


def _build_scenario_formal_summary(
    scenario_policy_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows_by_scenario: dict[str, list[Mapping[str, object]]] = {}
    for row in scenario_policy_rows:
        rows_by_scenario.setdefault(str(row["scenario"]), []).append(row)

    summaries: list[dict[str, object]] = []
    for scenario, rows in rows_by_scenario.items():
        ordered = sorted(rows, key=lambda row: (-float(row["mean_qoe"]), str(row["policy"])))
        best = ordered[0]
        proposed = next((row for row in ordered if row["policy"] == "Proposed-RL"), None)
        summaries.append(
            {
                "scenario": scenario,
                "best_policy": best["policy"],
                "best_mean_qoe": best["mean_qoe"],
                "proposed_rl_mean_qoe": None if proposed is None else proposed["mean_qoe"],
                "proposed_rl_mean_rank": None if proposed is None else proposed["mean_rank"],
                "proposed_rl_mean_gap_to_best": None if proposed is None else proposed["mean_qoe"] - best["mean_qoe"],
            }
        )
    return summaries


def _mean_metric(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _std_metric(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _write_formal_outputs(
    settings: FormalExperimentSettings,
    seed_results: Sequence[Mapping[str, object]],
    aggregate: Mapping[str, list[dict[str, object]]],
) -> dict[str, object]:
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "policy_seed_scenario_metrics": output_dir / "policy_seed_scenario_metrics.csv",
        "scenario_seed_winners": output_dir / "scenario_seed_winners.csv",
        "scenario_policy_aggregate": output_dir / "scenario_policy_aggregate.csv",
        "policy_formal_aggregate": output_dir / "policy_formal_aggregate.csv",
        "scenario_formal_summary": output_dir / "scenario_formal_summary.csv",
        "comparison_report": output_dir / "comparison_report.md",
        "summary": output_dir / "summary.json",
    }
    for key, path in outputs.items():
        if path.suffix == ".csv":
            pd.DataFrame(aggregate[key]).to_csv(path, index=False)

    figures = {}
    if not settings.skip_plots:
        figures = _write_figures(output_dir, aggregate)

    report = _write_comparison_report(
        outputs["comparison_report"],
        settings=settings,
        aggregate=aggregate,
        figures=figures,
    )
    summary = {
        "purpose": "Formal multi-seed PPO scheduling experiment over deterministic stress scenarios.",
        "note": "Uses reproducible synthetic link traces; STK/ns-3 is not connected in this run.",
        "configuration": {
            "seeds": list(settings.seeds),
            "limit_rois": settings.limit_rois,
            "total_timesteps": settings.total_timesteps,
            "scenarios": list(SCENARIO_NAMES),
        },
        "seed_runs": [
            {
                "seed": int(result["seed"]),
                "summary_path": result["summary_path"],
            }
            for result in seed_results
        ],
        "policy_formal_aggregate": aggregate["policy_formal_aggregate"],
        "scenario_formal_summary": aggregate["scenario_formal_summary"],
        "outputs": {
            **{key: str(path.resolve()) for key, path in outputs.items()},
            **{key: str(path.resolve()) for key, path in figures.items()},
        },
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(report)
    return summary


def _write_figures(
    output_dir: Path,
    aggregate: Mapping[str, list[dict[str, object]]],
) -> dict[str, Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping figures because matplotlib is unavailable: {exc}")
        return {}

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_qoe_by_policy": figure_dir / "qoe_by_policy.png",
        "figure_qoe_by_scenario_policy": figure_dir / "qoe_by_scenario_policy.png",
        "figure_qoe_cdf_by_policy": figure_dir / "qoe_cdf_by_policy.png",
    }

    policy_rows = aggregate["policy_formal_aggregate"]
    policies = [str(row["policy"]) for row in policy_rows]
    means = [float(row["mean_qoe"]) for row in policy_rows]
    stds = [float(row["std_qoe"]) for row in policy_rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(policies, means, yerr=stds, capsize=4, color="#3f7cac")
    ax.set_ylabel("Canonical QoE")
    ax.set_title("Mean QoE by Policy")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figures["figure_qoe_by_policy"], dpi=180)
    plt.close(fig)

    scenario_policy = pd.DataFrame(aggregate["scenario_policy_aggregate"])
    pivot = scenario_policy.pivot(index="scenario", columns="policy", values="mean_qoe")
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title("Scenario x Policy Mean QoE")
    fig.colorbar(im, ax=ax, label="Canonical QoE")
    fig.tight_layout()
    fig.savefig(figures["figure_qoe_by_scenario_policy"], dpi=180)
    plt.close(fig)

    metrics = pd.DataFrame(aggregate["policy_seed_scenario_metrics"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for policy, group in metrics.groupby("policy"):
        values = sorted(float(value) for value in group["qoe_total"])
        y_values = [(index + 1) / len(values) for index in range(len(values))]
        ax.step(values, y_values, where="post", label=str(policy))
    ax.set_xlabel("Canonical QoE")
    ax.set_ylabel("CDF")
    ax.set_title("QoE CDF by Policy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures["figure_qoe_cdf_by_policy"], dpi=180)
    plt.close(fig)
    return figures


def _write_comparison_report(
    path: Path,
    *,
    settings: FormalExperimentSettings,
    aggregate: Mapping[str, list[dict[str, object]]],
    figures: Mapping[str, Path],
) -> str:
    lines = [
        "# Formal PPO Scheduling Experiment",
        "",
        "This report aggregates multi-seed PPO scheduling runs over the deterministic stress scenarios.",
        "It uses the canonical slot QoE metric shared with deterministic baselines.",
        "",
        "## Configuration",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in settings.seeds)}",
        f"- ROI limit: {settings.limit_rois}",
        f"- PPO timesteps per scenario: {settings.total_timesteps}",
        f"- Scenarios: {', '.join(SCENARIO_NAMES)}",
        "",
        "## Policy Aggregate",
        "",
        _markdown_table(aggregate["policy_formal_aggregate"], POLICY_FORMAL_COLUMNS),
        "",
        "## Scenario Summary",
        "",
        _markdown_table(aggregate["scenario_formal_summary"], SCENARIO_FORMAL_COLUMNS),
        "",
    ]
    if figures:
        lines.extend(
            [
                "## Figures",
                "",
                *[f"- {label}: `{path}`" for label, path in figures.items()],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def _markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format_value(key, row.get(key, "")) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_value(key: str, value: object) -> str:
    if value in {"", None}:
        return ""
    if key in {"policy", "scenario", "best_policy"}:
        return str(value)
    if key in {"sample_count", "seed_count", "scenario_count", "best_count"}:
        return str(int(float(value)))
    return f"{float(value):.4f}"


def main() -> int:
    args = parse_args()
    settings = _resolve_settings(args)
    _validate_settings(settings)
    print(
        "Estimated formal PPO experiment runtime before training: "
        f"{_runtime_estimate(settings)} for {len(settings.seeds)} seeds, "
        f"{len(SCENARIO_NAMES)} scenarios, {settings.total_timesteps} timesteps/scenario.",
        flush=True,
    )
    commands = [_build_seed_command(settings, seed) for seed in settings.seeds]
    if settings.dry_run:
        print(json.dumps({"dry_run": True, "commands": commands}, indent=2, ensure_ascii=False))
        return 0

    _prepare_output_dir(settings.output_dir, exist_ok=settings.exist_ok)
    seed_results = [_run_seed(settings, seed) for seed in settings.seeds]
    aggregate = _build_formal_aggregate(seed_results)
    summary = _write_formal_outputs(settings, seed_results, aggregate)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
