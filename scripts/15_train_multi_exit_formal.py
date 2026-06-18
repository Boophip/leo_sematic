"""Train the formal multi-exit baseline on GT ROI crops."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multi_exit.training import train_multi_exit_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-metadata",
        type=Path,
        default=(
            ROOT
            / "data"
            / "roi_crops_gt"
            / "dota_v1_lite_300_100_100"
            / "train"
            / "roi_metadata_labeled.jsonl"
        ),
    )
    parser.add_argument(
        "--val-metadata",
        type=Path,
        default=(
            ROOT
            / "data"
            / "roi_crops_gt"
            / "dota_v1_lite_300_100_100"
            / "val"
            / "roi_metadata_labeled.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "multi_exit" / "formal",
    )
    parser.add_argument("--limit-train-rois", type=int)
    parser.add_argument("--limit-val-rois", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _estimate_runtime(args: argparse.Namespace) -> str:
    if args.limit_train_rois is not None and args.limit_train_rois <= 512:
        return "about 1-5 minutes"
    if args.limit_train_rois is not None and args.limit_train_rois <= 3000:
        return "about 5-20 minutes"
    return "about 30-90 minutes depending on GPU and ROI count"


def main() -> int:
    args = parse_args()
    _prepare_output_dir(args.output_dir, exist_ok=args.exist_ok)
    print(f"Multi-exit training estimate: {_estimate_runtime(args)}; this is not an RL run.")
    summary = train_multi_exit_model(
        args.train_metadata,
        args.val_metadata,
        args.output_dir,
        limit_train_rois=args.limit_train_rois,
        limit_val_rois=args.limit_val_rois,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_size=args.input_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
