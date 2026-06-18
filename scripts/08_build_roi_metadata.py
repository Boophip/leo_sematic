"""Build ROI metadata, crops, and AoSI grid summaries from original-image NMS."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.roi_metadata import build_roi_metadata_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "detector"
            / "yolo11n_obb_full_test_original"
            / "nms_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=ROOT / "data" / "DOTA-demo" / "test" / "images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "roi_metadata" / "dota_demo_1024_o200" / "test",
    )
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument(
        "--limit-images",
        type=int,
        help="Build only the first N original images for smoke testing.",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Write metadata and grid summaries without perspective-corrected crop PNGs.",
    )
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

    _prepare_output_dir(args.output_dir, exist_ok=args.exist_ok)
    summary = build_roi_metadata_dataset(
        predictions_path=args.predictions,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        grid_rows=args.grid_size,
        grid_cols=args.grid_size,
        limit_images=args.limit_images,
        make_crops=not args.no_crops,
    )
    summary["configuration"]["split"] = args.split
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
