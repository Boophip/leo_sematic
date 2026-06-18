"""Create a class-aware DOTA-v1.0 lite split from original large images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dota_lite_split import (  # noqa: E402
    build_summary,
    discover_records,
    stratified_split,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "splits" / "dota_v1_lite_300_100_100",
    )
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-class-images",
        type=int,
        default=5,
        help="Hard minimum number of images containing each class in every split.",
    )
    parser.add_argument("--refinement-steps", type=int, default=30_000)
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Preview a split from all labels even when image files are missing.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Allow images with no annotated DOTA-v1.0 objects to enter the split.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Copy selected images and labels after writing manifests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, report = discover_records(
        args.dataset_root,
        labels_only=args.labels_only,
        include_empty=args.include_empty,
    )
    print(
        f"labels={report.label_count} images={report.image_count} "
        f"matched={report.matched_count} usable={report.usable_record_count} "
        f"missing_images={len(report.missing_image_ids)} "
        f"empty_labels={len(report.empty_label_ids)}"
    )
    if not args.labels_only and len(report.missing_image_ids) > 0:
        print(
            "Note: unmatched labels are excluded, so this split represents only the "
            "currently available image pool.",
            file=sys.stderr,
        )

    splits = stratified_split(
        records,
        {"train": args.train_size, "val": args.val_size, "test": args.test_size},
        seed=args.seed,
        min_class_images=args.min_class_images,
        refinement_steps=args.refinement_steps,
    )
    summary = build_summary(splits)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.dry_run:
        if args.labels_only and args.materialize:
            raise ValueError("--labels-only and --materialize cannot be used together")
        write_outputs(
            splits,
            args.output_dir,
            materialize=args.materialize,
            metadata={
                "dataset_root": str(args.dataset_root.resolve()),
                "split_sizes": {
                    "train": args.train_size,
                    "val": args.val_size,
                    "test": args.test_size,
                },
                "seed": args.seed,
                "min_class_images": args.min_class_images,
                "refinement_steps": args.refinement_steps,
                "labels_only": args.labels_only,
                "include_empty": args.include_empty,
                "materialized": args.materialize,
                "source_inventory": {
                    "labels": report.label_count,
                    "images": report.image_count,
                    "matched": report.matched_count,
                    "usable_records": report.usable_record_count,
                    "missing_images": len(report.missing_image_ids),
                    "images_without_labels": len(report.image_without_label_ids),
                    "empty_labels": len(report.empty_label_ids),
                },
            },
        )
        print(f"Wrote manifests and summary to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
