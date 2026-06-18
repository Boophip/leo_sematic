"""Label ROI metadata by matching detections to original DOTA ground truths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.roi_labeling import build_labeled_roi_metadata_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT
        / "data"
        / "roi_metadata"
        / "dota_demo_1024_o200"
        / "test"
        / "roi_metadata.jsonl",
    )
    parser.add_argument(
        "--gt-label-dir",
        type=Path,
        default=ROOT / "data" / "DOTA-demo" / "test" / "labelTxt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "roi_metadata" / "dota_demo_1024_o200" / "test",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--limit-images",
        type=int,
        help="Label only the first N original images for smoke testing.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing labeled metadata outputs.",
    )
    return parser.parse_args()


def _check_outputs(output_dir: Path, *, exist_ok: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        output_dir / "roi_metadata_labeled.jsonl",
        output_dir / "roi_metadata_labeled.csv",
        output_dir / "roi_label_summary.json",
    )
    existing = [path for path in targets if path.exists()]
    if existing and not exist_ok:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output files already exist; pass --exist-ok to overwrite: {paths}")


def main() -> int:
    args = parse_args()
    _check_outputs(args.output_dir, exist_ok=args.exist_ok)
    summary = build_labeled_roi_metadata_dataset(
        metadata_path=args.metadata,
        label_dir=args.gt_label_dir,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        limit_images=args.limit_images,
    )
    summary["configuration"]["split"] = args.split
    summary_path = args.output_dir / "roi_label_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
