"""Build GT ROI crop datasets for formal multi-exit training."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.gt_roi_crops import build_gt_roi_crop_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT / "data" / "splits" / "dota_v1_lite_300_100_100",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "roi_crops_gt" / "dota_v1_lite_300_100_100",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--include-difficult", action="store_true")
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--limit-objects", type=int)
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
    if args.grid_size <= 0:
        raise ValueError("--grid-size must be positive")
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    summaries = {}
    for split in splits:
        output_dir = args.output_root / split
        _prepare_output_dir(output_dir, exist_ok=args.exist_ok)
        summaries[split] = build_gt_roi_crop_dataset(
            args.split_dir / f"{split}.csv",
            output_dir,
            split=split,
            grid_rows=args.grid_size,
            grid_cols=args.grid_size,
            include_difficult=args.include_difficult,
            limit_images=args.limit_images,
            limit_objects=args.limit_objects,
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
