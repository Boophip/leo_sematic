"""Tile already-separated DOTA train/val/test images with overlap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dota_tiling import TileConfig, process_dota_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--min-visible-ratio", type=float, default=0.7)
    parser.add_argument("--padding-value", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TileConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visible_ratio=args.min_visible_ratio,
        padding_value=args.padding_value,
    )
    summaries = {}
    for split in args.splits:
        summaries[split] = process_dota_split(
            args.input_root / split,
            args.output_root / split,
            config,
            max_images=args.max_images,
            dry_run=args.dry_run,
        )
        print(json.dumps({split: summaries[split]}, indent=2, ensure_ascii=False))

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "summary.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
