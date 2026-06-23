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

from src.proxy.quality_proxy import MODEL_TYPES, train_quality_proxy_smoke  # noqa: E402


DEFAULT_PROFILE_CSV = (
    ROOT
    / "data"
    / "profiling"
    / "dota_v1_lite_300_100_100"
    / "test_full"
    / "roi_profile_smoke.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "proxy" / "model_comparison"
COMPARISON_FIELDNAMES = (
    "model_type",
    "model_path",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-types", nargs="+", choices=MODEL_TYPES, default=list(MODEL_TYPES))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--hidden-layer-sizes", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--quality-threshold", type=float, default=0.5)
    parser.add_argument("--no-image-features", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def choose_primary_model(rows: Iterable[Mapping[str, object]]) -> Mapping[str, object]:
    """Pick the strongest model using high-value MAE, then R2, then F1."""

    candidates = list(rows)
    if not candidates:
        raise ValueError("at least one model comparison row is required")
    return sorted(
        candidates,
        key=lambda row: (
            float(row["test_high_value_mae"]),
            -float(row["test_r2"]),
            -float(row["test_threshold_f1"]),
        ),
    )[0]


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _comparison_row(model_type: str, summary: Mapping[str, object]) -> dict[str, object]:
    metrics = summary["metrics"]["test"]  # type: ignore[index]
    high_value = metrics["high_value"]  # type: ignore[index]
    threshold = metrics["threshold_metrics"]  # type: ignore[index]
    ranking = metrics["action_ranking"]  # type: ignore[index]
    outputs = summary["model_path"]
    return {
        "model_type": model_type,
        "model_path": outputs,
        "test_mae": metrics["mae"],  # type: ignore[index]
        "test_mse": metrics["mse"],  # type: ignore[index]
        "test_r2": metrics["r2"],  # type: ignore[index]
        "test_high_value_mae": high_value["mae"],
        "test_threshold_f1": threshold["f1"],
        "test_top1_agreement": ranking["top1_agreement"],
        "test_mean_regret": ranking["mean_regret"],
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

    summaries: dict[str, object] = {}
    comparison_rows: list[dict[str, object]] = []
    for model_type in args.model_types:
        model_output_dir = args.output_dir / model_type
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

    primary = choose_primary_model(comparison_rows)
    primary_model_path = Path(str(primary["model_path"]))
    copied_primary_path = args.output_dir / "quality_proxy.joblib"
    shutil.copy2(primary_model_path, copied_primary_path)

    comparison_csv = args.output_dir / "model_comparison.csv"
    _write_comparison_csv(comparison_csv, comparison_rows)
    summary_payload = {
        "purpose": "Task-quality proxy model-family comparison.",
        "profile_csv": str(args.profile_csv.resolve()),
        "configuration": {
            "model_types": list(args.model_types),
            "test_size": args.test_size,
            "seed": args.seed,
            "max_iter": args.max_iter,
            "hidden_layer_sizes": list(args.hidden_layer_sizes),
            "quality_threshold": args.quality_threshold,
            "include_image_features": not args.no_image_features,
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
