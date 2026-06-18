"""Run a deliberate YOLO11n-OBB overfit test on the same smoke train/val set."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "detector" / "yolo11n_obb_overfit32",
    )
    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--map50-threshold", type=float, default=0.90)
    return parser.parse_args()


def _results_rows(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}, {}

    def parse(row: dict[str, str]) -> dict[str, float]:
        return {
            key.strip(): float(value)
            for key, value in row.items()
            if value not in (None, "")
        }

    return parse(rows[0]), parse(rows[-1])


def _metric(results: dict[str, float], suffix: str) -> float | None:
    for key, value in results.items():
        if key.endswith(suffix):
            return float(value)
    return None


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO

    data_payload = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    data_root = Path(data_payload["path"])
    train_list = data_root / data_payload["train"]
    train_images = [
        line.strip()
        for line in train_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        optimizer="AdamW",
        lr0=0.001,
        amp=True,
        cache="ram",
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        translate=0.0,
        scale=0.0,
        degrees=0.0,
        fliplr=0.0,
        flipud=0.0,
        patience=0,
        project=str(args.output_dir.parent.resolve()),
        name=args.output_dir.name,
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    best_path = args.output_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(args.data.resolve()),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=True,
        project=str(args.output_dir.parent.resolve()),
        name=f"{args.output_dir.name}_validation",
        exist_ok=True,
    )
    metric_results = {
        key: float(value)
        for key, value in metrics.results_dict.items()
        if isinstance(value, (int, float))
    }
    first_row, final_row = _results_rows(args.output_dir / "results.csv")
    loss_keys = (
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "train/angle_loss",
    )
    loss_change = {
        key: final_row[key] - first_row[key]
        for key in loss_keys
        if key in first_row and key in final_row
    }
    map50 = _metric(metric_results, "mAP50(B)")
    if map50 is None:
        map50 = _metric(metric_results, "mAP50-95(B)")

    manifest_path = data_root / f"{args.data.stem}_manifest.json"
    smoke_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    report = {
        "purpose": "Intentional same-set YOLO-OBB overfit test; not a formal metric.",
        "accepted": map50 is not None and map50 >= args.map50_threshold,
        "map50_threshold": args.map50_threshold,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "configuration": {
            "model": args.model,
            "data": str(args.data.resolve()),
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "device": args.device,
            "seed": args.seed,
        },
        "data": {
            "images": len(train_images),
            "covered_classes": smoke_manifest.get("covered_classes", []),
            "class_image_presence": smoke_manifest.get("class_image_presence", {}),
            "class_object_counts": smoke_manifest.get("class_object_counts", {}),
        },
        "first_training_row": first_row,
        "final_training_row": final_row,
        "training_loss_change_final_minus_first": loss_change,
        "validation_metrics": metric_results,
    }
    (args.output_dir / "overfit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
