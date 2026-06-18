"""Run a tiny multi-exit smoke training pass."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multi_exit.training import train_multi_exit_smoke  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(
            ROOT
            / "data"
            / "roi_metadata"
            / "dota_demo_1024_o200"
            / "test"
            / "roi_metadata_labeled.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "multi_exit" / "smoke",
    )
    parser.add_argument("--limit-rois", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    _prepare_output_dir(args.output_dir, exist_ok=args.exist_ok)
    print("Tiny multi-exit smoke training estimate: about 5-15 minutes; not a formal run.")
    summary = train_multi_exit_smoke(
        args.metadata,
        args.output_dir,
        limit_rois=args.limit_rois,
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
