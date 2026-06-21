"""Run multi-seed PPO scheduling experiments and aggregate formal reports."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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
    if settings.select_best_checkpoint:
        command.append("--select-best-checkpoint")
    optional_args: list[tuple[str, object | None]] = [
        ("--config", settings.config),
        ("--profile-csv", settings.profile_csv),
        ("--quality-proxy", settings.quality_proxy),
        ("--rois-per-slot", settings.rois_per_slot),
        ("--deadline-ms", settings.deadline_ms),
        ("--quality-threshold", settings.quality_threshold),
        ("--ppo-n-steps", settings.ppo_n_steps),
        ("--ppo-batch-size", settings.ppo_batch_size),
        ("--ppo-n-epochs", settings.ppo_n_epochs),
        ("--ppo-learning-rate", settings.ppo_learning_rate),
        ("--ppo-gamma", settings.ppo_gamma),
        ("--ppo-ent-coef", settings.ppo_ent_coef),
        ("--eval-frequency", settings.eval_frequency if settings.eval_frequency > 0 else None),
        (
            "--checkpoint-frequency",
            settings.checkpoint_frequency if settings.checkpoint_frequency > 0 else None,
        ),
        (
            "--best-checkpoint-min-success-rate",
            settings.best_checkpoint_min_success_rate
            if settings.select_best_checkpoint or settings.best_checkpoint_min_success_rate > 0
            else None,
        ),
        ("--reward-quality-deficit-weight", settings.reward_quality_deficit_weight),
        ("--reward-delay-excess-weight", settings.reward_delay_excess_weight),
        ("--reward-virtual-queue-weight", settings.reward_virtual_queue_weight),
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
    training_curve_rows: list[dict[str, object]] = []
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
        training_curve_rows.extend(_read_seed_training_curves(seed, summary))

    scenario_policy_aggregate = _aggregate_by_keys(policy_rows, ("scenario", "policy"))
    policy_formal_aggregate = _aggregate_by_keys(policy_rows, ("policy",))
    scenario_formal_summary = _build_scenario_formal_summary(scenario_policy_aggregate)
    return {
        "policy_seed_scenario_metrics": policy_rows,
        "scenario_seed_winners": scenario_winner_rows,
        "scenario_policy_aggregate": scenario_policy_aggregate,
        "policy_formal_aggregate": policy_formal_aggregate,
        "scenario_formal_summary": scenario_formal_summary,
        "training_curve": training_curve_rows,
    }


def _read_seed_training_curves(seed: int, summary: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario_summary in summary.get("scenarios", []):
        outputs = scenario_summary.get("outputs", {})
        curve_path = outputs.get("training_curve")
        if not curve_path:
            continue
        path = Path(str(curve_path))
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            item = dict(row)
            item.setdefault("seed", seed)
            item.setdefault("scenario", scenario_summary["configuration"]["scenario"])
            rows.append(item)
    return rows


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
        "training_curve_all": output_dir / "training_curve_all.csv",
        "comparison_report": output_dir / "comparison_report.md",
        "summary": output_dir / "summary.json",
    }
    for key, path in outputs.items():
        if path.suffix == ".csv":
            aggregate_key = "training_curve" if key == "training_curve_all" else key
            pd.DataFrame(aggregate[aggregate_key]).to_csv(path, index=False)

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
            "eval_frequency": settings.eval_frequency,
            "checkpoint_frequency": settings.checkpoint_frequency,
            "select_best_checkpoint": settings.select_best_checkpoint,
            "best_checkpoint_min_success_rate": settings.best_checkpoint_min_success_rate,
            "ppo_overrides": {
                "n_steps": settings.ppo_n_steps,
                "batch_size": settings.ppo_batch_size,
                "n_epochs": settings.ppo_n_epochs,
                "learning_rate": settings.ppo_learning_rate,
                "gamma": settings.ppo_gamma,
                "ent_coef": settings.ppo_ent_coef,
            },
            "reward_shaping": {
                "quality_deficit_weight": settings.reward_quality_deficit_weight,
                "delay_excess_weight": settings.reward_delay_excess_weight,
                "virtual_queue_weight": settings.reward_virtual_queue_weight,
            },
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
        "training_curve_rows": len(aggregate["training_curve"]),
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
    if aggregate["training_curve"]:
        figures["figure_training_curve_qoe"] = figure_dir / "training_curve_qoe.png"
        figures["figure_training_curve_actions"] = figure_dir / "training_curve_actions.png"
        figures["figure_training_curve_violations"] = figure_dir / "training_curve_violations.png"
        figures["figure_training_curve_queues"] = figure_dir / "training_curve_queues.png"

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
    if aggregate["training_curve"]:
        curve = pd.DataFrame(aggregate["training_curve"])
        grouped = (
            curve.groupby(["scenario", "timestep"], as_index=False)["qoe_total"]
            .mean()
            .sort_values(["scenario", "timestep"])
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for scenario, group in grouped.groupby("scenario"):
            ax.plot(group["timestep"], group["qoe_total"], marker="o", label=str(scenario))
        ax.set_xlabel("PPO timesteps")
        ax.set_ylabel("Mean canonical QoE")
        ax.set_title("Training Evaluation Curve")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures["figure_training_curve_qoe"], dpi=180)
        plt.close(fig)

        action_columns = ("local_count", "offload_count", "drop_count", "illegal_count")
        action_grouped = (
            curve.groupby("timestep", as_index=False)[list(action_columns)]
            .mean()
            .sort_values("timestep")
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for column in action_columns:
            ax.plot(action_grouped["timestep"], action_grouped[column], marker="o", label=column)
        ax.set_xlabel("PPO timesteps")
        ax.set_ylabel("Mean count per evaluation")
        ax.set_title("Aggregate Action Mix During Training")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures["figure_training_curve_actions"], dpi=180)
        plt.close(fig)

        violation_columns = ("illegal_count", "timeout_count", "quality_violation_count")
        violation_grouped = (
            curve.groupby(["scenario", "timestep"], as_index=False)[list(violation_columns)]
            .mean()
            .sort_values(["scenario", "timestep"])
        )
        fig, axes = plt.subplots(1, len(violation_columns), figsize=(15, 4), sharex=True)
        for ax, column in zip(axes, violation_columns):
            for scenario, group in violation_grouped.groupby("scenario"):
                ax.plot(group["timestep"], group[column], marker="o", label=str(scenario))
            ax.set_title(column)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Mean count")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures["figure_training_curve_violations"], dpi=180)
        plt.close(fig)

        queue_columns = ("quality_virtual_queue", "delay_virtual_queue_ms")
        queue_grouped = (
            curve.groupby(["scenario", "timestep"], as_index=False)[list(queue_columns)]
            .mean()
            .sort_values(["scenario", "timestep"])
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        for scenario, group in queue_grouped.groupby("scenario"):
            axes[0].plot(group["timestep"], group["quality_virtual_queue"], marker="o", label=str(scenario))
            axes[1].plot(group["timestep"], group["delay_virtual_queue_ms"], marker="o", label=str(scenario))
        axes[0].set_title("Quality Virtual Queue")
        axes[0].set_xlabel("Timesteps")
        axes[0].set_ylabel("Mean queue")
        axes[1].set_title("Delay Virtual Queue")
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Mean queue ms")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures["figure_training_curve_queues"], dpi=180)
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
