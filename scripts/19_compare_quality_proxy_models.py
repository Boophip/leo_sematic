"""Compare task-quality proxy model families on one profiling table."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.proxy.quality_proxy import (  # noqa: E402
    MODEL_TYPE_HIST_GBDT,
    MODEL_TYPE_MLP,
    MODEL_TYPES,
    train_quality_proxy_smoke,
    train_quality_proxy_strict,
)


DEFAULT_PROFILE_CSV = (
    ROOT
    / "data"
    / "profiling"
    / "dota_v1_lite_300_100_100"
    / "test_full"
    / "roi_profile_smoke.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "proxy" / "strict_image_features"
COMPARISON_FIELDNAMES = (
    "model_type",
    "model_path",
    "model_size_bytes",
    "val_mae",
    "val_mse",
    "val_r2",
    "val_high_value_mae",
    "val_threshold_f1",
    "val_top1_agreement",
    "val_mean_regret",
    "test_mae",
    "test_mse",
    "test_r2",
    "test_high_value_mae",
    "test_threshold_f1",
    "test_top1_agreement",
    "test_mean_regret",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-csv", type=Path, default=DEFAULT_PROFILE_CSV)
    parser.add_argument("--train-profile-csv", type=Path)
    parser.add_argument("--val-profile-csv", type=Path)
    parser.add_argument("--test-profile-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=MODEL_TYPES,
        default=[MODEL_TYPE_MLP, MODEL_TYPE_HIST_GBDT],
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--hidden-layer-sizes", type=int, nargs="+", default=[128, 64, 32])
    parser.add_argument("--quality-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-primary-model-mb",
        type=float,
        default=50.0,
        help=(
            "Only models at or below this size are eligible for primary selection. "
            "Use a non-positive value to disable the lightweight cap."
        ),
    )
    parser.add_argument("--no-image-features", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def choose_primary_model(
    rows: Iterable[Mapping[str, object]],
    *,
    max_primary_model_mb: float | None = 50.0,
) -> Mapping[str, object]:
    """Pick the best lightweight model using validation metrics when present."""

    candidates = list(rows)
    if not candidates:
        raise ValueError("at least one model comparison row is required")
    if max_primary_model_mb is not None and max_primary_model_mb > 0:
        max_bytes = max_primary_model_mb * 1024 * 1024
        lightweight_candidates = [
            row
            for row in candidates
            if float(row.get("model_size_bytes", float("inf"))) <= max_bytes
        ]
        if lightweight_candidates:
            candidates = lightweight_candidates
    def selection_key(row: Mapping[str, object]) -> tuple[float, float, float, float]:
        prefix = "val" if "val_high_value_mae" in row else "test"
        return (
            float(row[f"{prefix}_high_value_mae"]),
            -float(row[f"{prefix}_r2"]),
            -float(row[f"{prefix}_threshold_f1"]),
            float(row.get("model_size_bytes", 0.0)),
        )

    return sorted(candidates, key=selection_key)[0]


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _comparison_row(model_type: str, summary: Mapping[str, object]) -> dict[str, object]:
    all_metrics = summary["metrics"]  # type: ignore[index]
    val_metrics = all_metrics.get("val", all_metrics["test"])  # type: ignore[union-attr,index]
    test_metrics = all_metrics["test"]  # type: ignore[index]
    val_high_value = val_metrics["high_value"]  # type: ignore[index]
    val_threshold = val_metrics["threshold_metrics"]  # type: ignore[index]
    val_ranking = val_metrics["action_ranking"]  # type: ignore[index]
    test_high_value = test_metrics["high_value"]  # type: ignore[index]
    test_threshold = test_metrics["threshold_metrics"]  # type: ignore[index]
    test_ranking = test_metrics["action_ranking"]  # type: ignore[index]
    outputs = summary["model_path"]
    model_path = Path(str(outputs))
    return {
        "model_type": model_type,
        "model_path": outputs,
        "model_size_bytes": summary.get("model_size_bytes", model_path.stat().st_size),
        "val_mae": val_metrics["mae"],  # type: ignore[index]
        "val_mse": val_metrics["mse"],  # type: ignore[index]
        "val_r2": val_metrics["r2"],  # type: ignore[index]
        "val_high_value_mae": val_high_value["mae"],
        "val_threshold_f1": val_threshold["f1"],
        "val_top1_agreement": val_ranking["top1_agreement"],
        "val_mean_regret": val_ranking["mean_regret"],
        "test_mae": test_metrics["mae"],  # type: ignore[index]
        "test_mse": test_metrics["mse"],  # type: ignore[index]
        "test_r2": test_metrics["r2"],  # type: ignore[index]
        "test_high_value_mae": test_high_value["mae"],
        "test_threshold_f1": test_threshold["f1"],
        "test_top1_agreement": test_ranking["top1_agreement"],
        "test_mean_regret": test_ranking["mean_regret"],
    }


def _write_comparison_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    _prepare_output_dir(args.output_dir, exist_ok=args.exist_ok)
    print("Quality proxy model comparison estimate: about 1-5 minutes per model; not an RL run.")
    strict_inputs = (args.train_profile_csv, args.val_profile_csv, args.test_profile_csv)
    strict_mode = any(path is not None for path in strict_inputs)
    if strict_mode and not all(path is not None for path in strict_inputs):
        raise ValueError(
            "--train-profile-csv, --val-profile-csv, and --test-profile-csv must be provided together"
        )

    summaries: dict[str, object] = {}
    comparison_rows: list[dict[str, object]] = []
    for model_type in args.model_types:
        model_output_dir = args.output_dir / model_type
        if strict_mode:
            summary = train_quality_proxy_strict(
                args.train_profile_csv,
                args.val_profile_csv,
                args.test_profile_csv,
                model_output_dir,
                seed=args.seed,
                max_iter=args.max_iter,
                hidden_layer_sizes=args.hidden_layer_sizes,
                model_type=model_type,
                include_image_features=not args.no_image_features,
                quality_threshold=args.quality_threshold,
            )
        else:
            summary = train_quality_proxy_smoke(
                args.profile_csv,
                model_output_dir,
                test_size=args.test_size,
                seed=args.seed,
                max_iter=args.max_iter,
                hidden_layer_sizes=args.hidden_layer_sizes,
                model_type=model_type,
                include_image_features=not args.no_image_features,
                quality_threshold=args.quality_threshold,
            )
        summaries[model_type] = summary
        comparison_rows.append(_comparison_row(model_type, summary))

    max_primary_model_mb = args.max_primary_model_mb
    if max_primary_model_mb is not None and max_primary_model_mb <= 0:
        max_primary_model_mb = None
    primary = choose_primary_model(
        comparison_rows,
        max_primary_model_mb=max_primary_model_mb,
    )
    primary_model_path = Path(str(primary["model_path"]))
    copied_primary_path = args.output_dir / "quality_proxy.joblib"
    shutil.copy2(primary_model_path, copied_primary_path)

    comparison_csv = args.output_dir / "model_comparison.csv"
    _write_comparison_csv(comparison_csv, comparison_rows)
    summary_payload = {
        "purpose": "Task-quality proxy model-family comparison.",
        "profile_csv": None if strict_mode else str(args.profile_csv.resolve()),
        "profile_csvs": (
            None
            if not strict_mode
            else {
                "train": str(args.train_profile_csv.resolve()),
                "val": str(args.val_profile_csv.resolve()),
                "test": str(args.test_profile_csv.resolve()),
            }
        ),
        "configuration": {
            "model_types": list(args.model_types),
            "split_mode": "explicit_train_val_test" if strict_mode else "group_shuffle_or_smoke",
            "test_size": args.test_size,
            "seed": args.seed,
            "max_iter": args.max_iter,
            "hidden_layer_sizes": list(args.hidden_layer_sizes),
            "quality_threshold": args.quality_threshold,
            "include_image_features": not args.no_image_features,
            "max_primary_model_mb": max_primary_model_mb,
        },
        "primary_model": {
            **dict(primary),
            "copied_model_path": str(copied_primary_path.resolve()),
        },
        "comparison_rows": comparison_rows,
        "model_summaries": summaries,
        "outputs": {
            "comparison_csv": str(comparison_csv.resolve()),
            "summary": str((args.output_dir / "summary.json").resolve()),
            "primary_quality_proxy": str(copied_primary_path.resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
