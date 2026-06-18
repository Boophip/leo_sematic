"""Train YOLO11n-OBB on the complete tiled DOTA-lite train/val split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector.yolo_obb_reporting import (  # noqa: E402
    best_training_row,
    environment_summary,
    metrics_summary,
    read_training_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "detector" / "yolo11n_obb_full",
    )
    parser.add_argument("--model", default=str(ROOT / "yolo11n-obb.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-det", type=int, default=2000)
    parser.add_argument("--cache", choices=["false", "ram", "disk"], default="false")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    if args.resume:
        checkpoint = args.output_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True)
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {args.output_dir}")
        cache: bool | str = False if args.cache == "false" else args.cache
        model = YOLO(args.model)
        model.train(
            data=str(args.data.resolve()),
            imgsz=args.imgsz,
            batch=args.batch,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            optimizer="AdamW",
            lr0=0.001,
            amp=True,
            cache=cache,
            mosaic=1.0,
            close_mosaic=10,
            mixup=0.0,
            copy_paste=0.0,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.5,
            max_det=args.max_det,
            project=str(args.output_dir.parent.resolve()),
            name=args.output_dir.name,
            exist_ok=False,
            plots=True,
            verbose=True,
        )

    best_path = args.output_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_path))
    val_metrics = best_model.val(
        data=str(args.data.resolve()),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        max_det=args.max_det,
        plots=True,
        project=str(args.output_dir.parent.resolve()),
        name=f"{args.output_dir.name}_val",
        exist_ok=True,
    )
    rows = read_training_rows(args.output_dir / "results.csv")
    report = {
        "purpose": "Formal tiled validation metrics; original-image restoration/NMS is a later stage.",
        "environment": environment_summary(ultralytics, torch),
        "configuration": {
            "model": args.model,
            "data": str(args.data.resolve()),
            "epochs_requested": args.epochs,
            "epochs_completed": len(rows),
            "patience": args.patience,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "device": args.device,
            "seed": args.seed,
            "max_det": args.max_det,
            "cache": args.cache,
        },
        "model": {
            "parameters": int(sum(parameter.numel() for parameter in best_model.model.parameters())),
            "flops_gflops": float(get_flops(best_model.model, imgsz=args.imgsz)),
            "weights": str(best_path.resolve()),
        },
        "best_training_row": best_training_row(rows),
        "validation": metrics_summary(val_metrics),
    }
    write_json(args.output_dir / "formal_train_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
