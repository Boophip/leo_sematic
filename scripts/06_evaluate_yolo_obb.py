"""Evaluate a trained YOLO-OBB checkpoint on an independent tiled split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector.yolo_obb_reporting import (  # noqa: E402
    environment_summary,
    metrics_summary,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "detector" / "yolo11n_obb_full_test",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    model = YOLO(str(args.weights.resolve()))
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        max_det=args.max_det,
        plots=True,
        project=str(args.output_dir.parent.resolve()),
        name=args.output_dir.name,
        exist_ok=True,
    )
    report = {
        "purpose": f"Independent tiled {args.split} evaluation.",
        "environment": environment_summary(ultralytics, torch),
        "configuration": {
            "data": str(args.data.resolve()),
            "weights": str(args.weights.resolve()),
            "split": args.split,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "device": args.device,
            "max_det": args.max_det,
        },
        "model": {
            "parameters": int(sum(parameter.numel() for parameter in model.model.parameters())),
            "flops_gflops": float(get_flops(model.model, imgsz=args.imgsz)),
        },
        "evaluation": metrics_summary(metrics),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "evaluation_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
