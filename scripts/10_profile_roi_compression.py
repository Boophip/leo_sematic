"""Smoke-generate ROI compression and multi-exit profiling rows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector.dota_original_eval import ULTRALYTICS_DOTA_V1_CLASSES  # noqa: E402
from src.multi_exit.inference import predict_image_exits, true_class_id_from_record  # noqa: E402
from src.multi_exit.model import TinyMultiExitCNN  # noqa: E402
from src.profiling.profile_generation import (  # noqa: E402
    ExitProfile,
    load_roi_metadata,
    profile_roi_records,
    write_profile_csv,
)


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
        "--checkpoint",
        type=Path,
        help="Optional tiny multi-exit checkpoint. If omitted, use an untrained smoke model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "profiling" / "dota_demo_1024_o200" / "test_smoke",
    )
    parser.add_argument(
        "--limit-rois",
        type=int,
        help="Optional ROI limit for smoke runs. Omit for full profiling.",
    )
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument("--include-local", action="store_true")
    parser.add_argument(
        "--compression-levels",
        nargs="+",
        default=["0", "1", "2", "3"],
        help="Compression levels such as 0 1 2 3 or beta_0 beta_1.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_model(checkpoint_path: Path | None, device: torch.device) -> tuple[TinyMultiExitCNN, str]:
    model = TinyMultiExitCNN(num_classes=len(ULTRALYTICS_DOTA_V1_CLASSES)).to(device)
    if checkpoint_path is None:
        return model, "untrained_smoke_model"
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    return model, str(checkpoint_path.resolve())


def main() -> int:
    args = parse_args()
    if args.limit_rois is not None and args.limit_rois <= 0:
        raise ValueError("--limit-rois must be positive")

    _prepare_output_dir(args.output_dir, exist_ok=args.exist_ok)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, model_source = _load_model(args.checkpoint, device)
    records = load_roi_metadata(args.metadata, limit_rois=args.limit_rois)
    levels: list[str | int] = ["local"] if args.include_local else []
    levels.extend(args.compression_levels)

    def predictor(image, record):
        predictions = predict_image_exits(
            model,
            image,
            input_size=args.input_size,
            device=device,
            true_class_id=true_class_id_from_record(record),
        )
        return [
            ExitProfile(
                exit_level=prediction.exit_level,
                inference_ms=prediction.inference_ms,
                task_quality=prediction.task_quality,
                predicted_class_id=prediction.predicted_class_id,
                exit_confidence=prediction.exit_confidence,
            )
            for prediction in predictions
        ]

    rows = profile_roi_records(records, predictor, compression_levels=levels)
    csv_path = args.output_dir / "roi_profile_smoke.csv"
    write_profile_csv(csv_path, rows)
    summary = {
        "purpose": "Smoke ROI compression/profile rows; not a formal experiment.",
        "metadata": str(args.metadata.resolve()),
        "model_source": model_source,
        "output_csv": str(csv_path.resolve()),
        "configuration": {
            "limit_rois": args.limit_rois,
            "input_size": args.input_size,
            "compression_levels": levels,
            "device": str(device),
        },
        "counts": {
            "roi_count": len(records),
            "profile_row_count": len(rows),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
