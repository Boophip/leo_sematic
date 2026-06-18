"""Restore tiled YOLO-OBB predictions and evaluate at original-image level."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector.dota_original_eval import (  # noqa: E402
    TilePrediction,
    ULTRALYTICS_DOTA_V1_CLASSES,
    evaluate_predictions,
    global_rotated_nms,
    load_ground_truths,
    load_tile_mappings,
    restore_predictions,
    write_dota_task1,
    write_json,
    write_per_image_summary,
    write_predictions_jsonl,
)
from src.detector.yolo_obb_reporting import environment_summary  # noqa: E402


SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "outputs" / "detector" / "yolo11n_obb_full" / "weights" / "best.pt",
    )
    parser.add_argument(
        "--yolo-root",
        type=Path,
        default=ROOT / "data" / "yolo_obb" / "dota_demo_1024_o200",
    )
    parser.add_argument(
        "--tiles-root",
        type=Path,
        default=ROOT / "data" / "tiles" / "dota_demo_1024_o200",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=ROOT / "data" / "DOTA-demo",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--tile-iou", type=float, default=0.7)
    parser.add_argument("--nms-iou", type=float, default=0.1)
    parser.add_argument("--eval-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--predict-chunk-size", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=2000)
    parser.add_argument(
        "--limit-images",
        type=int,
        help="Evaluate only the first N original images for smoke testing.",
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _default_output_dir(split: str) -> Path:
    return ROOT / "outputs" / "detector" / f"yolo11n_obb_full_{split}_original"


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _find_tile_image(image_dir: Path, tile_id: str) -> Path:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = image_dir / f"{tile_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing image for tile {tile_id!r} under {image_dir}")


def _selected_image_ids(
    source_ids: set[str],
    *,
    limit_images: int | None,
) -> list[str]:
    if limit_images is not None:
        if limit_images <= 0:
            raise ValueError("--limit-images must be positive")
        return sorted(source_ids)[:limit_images]
    return sorted(source_ids)


def _collect_tile_paths(
    image_dir: Path,
    mappings,
    selected_source_ids: set[str],
) -> list[Path]:
    paths: list[Path] = []
    for tile_id, mapping in mappings.items():
        if mapping.source_image_id in selected_source_ids:
            paths.append(_find_tile_image(image_dir, tile_id))
    if not paths:
        raise ValueError("no tile images were selected")
    return paths


def _prediction_class_name(class_id: int) -> str:
    if not 0 <= class_id < len(ULTRALYTICS_DOTA_V1_CLASSES):
        raise ValueError(f"model returned class id out of range: {class_id}")
    return ULTRALYTICS_DOTA_V1_CLASSES[class_id]


def _collect_predictions(model, tile_paths: list[Path], args: argparse.Namespace) -> list[TilePrediction]:
    if args.predict_chunk_size <= 0:
        raise ValueError("--predict-chunk-size must be positive")
    predictions: list[TilePrediction] = []
    for chunk_start in range(0, len(tile_paths), args.predict_chunk_size):
        chunk_paths = tile_paths[chunk_start : chunk_start + args.predict_chunk_size]
        results = model.predict(
            source=[str(path) for path in chunk_paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.tile_iou,
            max_det=args.max_det,
            device=args.device,
            batch=args.batch,
            stream=True,
            verbose=True,
        )
        for result_index, result in enumerate(results):
            tile_id = chunk_paths[result_index].stem
            obb = result.obb
            if obb is None or len(obb) == 0:
                continue
            polygons = obb.xyxyxyxy.detach().cpu().tolist()
            confidences = obb.conf.detach().cpu().tolist()
            class_ids = [int(value) for value in obb.cls.detach().cpu().tolist()]
            for polygon, confidence, class_id in zip(polygons, confidences, class_ids):
                points = tuple(
                    (float(point[0]), float(point[1]))
                    for point in polygon
                )
                predictions.append(
                    TilePrediction(
                        tile_id=tile_id,
                        class_id=class_id,
                        class_name=_prediction_class_name(class_id),
                        confidence=float(confidence),
                        polygon=points,  # type: ignore[arg-type]
                    )
                )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return predictions


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or _default_output_dir(args.split)
    _prepare_output_dir(output_dir, exist_ok=args.exist_ok)

    mapping_path = args.tiles_root / args.split / "mapping.jsonl"
    image_dir = args.yolo_root / "images" / args.split
    gt_label_dir = args.gt_root / args.split / "labelTxt"
    mappings = load_tile_mappings(mapping_path)
    selected_image_ids = _selected_image_ids(
        {mapping.source_image_id for mapping in mappings.values()},
        limit_images=args.limit_images,
    )
    tile_paths = _collect_tile_paths(image_dir, mappings, set(selected_image_ids))

    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))
    raw_predictions = _collect_predictions(model, tile_paths, args)
    restored_predictions = restore_predictions(raw_predictions, mappings)
    nms_predictions = global_rotated_nms(
        restored_predictions,
        iou_threshold=args.nms_iou,
    )
    ground_truths = load_ground_truths(
        gt_label_dir,
        image_ids=selected_image_ids,
    )
    evaluation = evaluate_predictions(
        nms_predictions,
        ground_truths,
        iou_threshold=args.eval_iou,
    )

    write_predictions_jsonl(output_dir / "raw_tile_predictions.jsonl", raw_predictions)
    write_predictions_jsonl(output_dir / "restored_predictions.jsonl", restored_predictions)
    write_predictions_jsonl(output_dir / "nms_predictions.jsonl", nms_predictions)
    write_dota_task1(output_dir / "dota_task1", nms_predictions)
    write_per_image_summary(
        output_dir / "per_image_summary.csv",
        selected_image_ids,
        restored_predictions=restored_predictions,
        nms_predictions=nms_predictions,
        ground_truths=ground_truths,
    )

    counts = dict(evaluation["counts"])
    counts.update(
        {
            "selected_original_images": len(selected_image_ids),
            "selected_tile_images": len(tile_paths),
            "tile_predictions": len(raw_predictions),
            "restored_predictions": len(restored_predictions),
            "nms_predictions": len(nms_predictions),
        }
    )
    report = {
        "note": (
            "Original-image local DOTA-style AP50 evaluation after coordinate "
            "restoration and global rotated NMS; this is not the Ultralytics "
            "tile-level metric."
        ),
        "environment": environment_summary(ultralytics, torch),
        "config": {
            "weights": str(args.weights.resolve()),
            "split": args.split,
            "conf": args.conf,
            "tile_iou": args.tile_iou,
            "nms_iou": args.nms_iou,
            "eval_iou": args.eval_iou,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "predict_chunk_size": args.predict_chunk_size,
            "device": args.device,
            "max_det": args.max_det,
            "limit_images": args.limit_images,
            "mapping_path": str(mapping_path.resolve()),
            "image_dir": str(image_dir.resolve()),
            "ground_truth_dir": str(gt_label_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
        },
        "aggregate": evaluation["aggregate"],
        "per_class": evaluation["per_class"],
        "counts": counts,
    }
    write_json(output_dir / "original_eval_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
