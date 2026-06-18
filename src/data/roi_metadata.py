"""Build ROI metadata, semantic values, crops, and AoSI grid summaries."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image

from src.data.dota_tiling import polygon_area
from src.detector.dota_original_eval import (
    RestoredPrediction,
    ULTRALYTICS_DOTA_V1_CLASSES,
)


Point = tuple[float, float]
Polygon = tuple[Point, Point, Point, Point]

ROI_FIELDNAMES = (
    "roi_id",
    "image_id",
    "source_image_path",
    "crop_path",
    "predicted_class",
    "class_id",
    "confidence",
    "global_obb",
    "center_x",
    "center_y",
    "area_pixels",
    "area_ratio",
    "grid_row",
    "grid_col",
    "grid_id",
    "semantic_value",
    "class_priority",
    "difficult",
    "source_tile_id",
)

SEMANTIC_WEIGHTS = {
    "class": 0.50,
    "confidence": 0.35,
    "area": 0.15,
}

TASK_CLASS_PRIORITIES = {
    "plane": 1.00,
    "ship": 1.00,
    "harbor": 1.00,
    "large-vehicle": 0.90,
    "small-vehicle": 0.90,
    "helicopter": 0.90,
    "bridge": 0.75,
    "storage-tank": 0.75,
    "roundabout": 0.60,
    "baseball-diamond": 0.60,
    "ground-track-field": 0.60,
    "tennis-court": 0.45,
    "basketball-court": 0.45,
    "soccer-ball-field": 0.45,
    "swimming-pool": 0.45,
}

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageInfo:
    image_id: str
    path: Path
    width: int
    height: int


@dataclass(frozen=True)
class RoiRecord:
    roi_id: str
    image_id: str
    source_image_path: str
    crop_path: str
    predicted_class: str
    class_id: int
    confidence: float
    global_obb: Polygon
    center_x: float
    center_y: float
    area_pixels: float
    area_ratio: float
    grid_row: int
    grid_col: int
    grid_id: int
    semantic_value: float
    class_priority: float
    difficult: int
    source_tile_id: str


def _polygon_from_json(points: Sequence[Sequence[float]]) -> Polygon:
    if len(points) != 4:
        raise ValueError(f"expected 4 polygon points, got {len(points)}")
    polygon = tuple((float(point[0]), float(point[1])) for point in points)
    if polygon_area(polygon) <= 0:
        raise ValueError(f"degenerate polygon: {points!r}")
    return polygon  # type: ignore[return-value]


def load_nms_predictions(path: Path) -> list[RestoredPrediction]:
    predictions: list[RestoredPrediction] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        class_id = int(payload["class_id"])
        if not 0 <= class_id < len(ULTRALYTICS_DOTA_V1_CLASSES):
            raise ValueError(f"{path}:{line_number}: class_id out of range: {class_id}")
        predictions.append(
            RestoredPrediction(
                tile_id=payload["tile_id"],
                source_image_id=payload["source_image_id"],
                class_id=class_id,
                class_name=payload["class_name"],
                confidence=float(payload["confidence"]),
                polygon=_polygon_from_json(payload["polygon"]),
            )
        )
    return predictions


def validate_class_priorities(
    class_priorities: Mapping[str, float] = TASK_CLASS_PRIORITIES,
    class_names: Sequence[str] = ULTRALYTICS_DOTA_V1_CLASSES,
) -> None:
    missing = sorted(set(class_names) - set(class_priorities))
    extra = sorted(set(class_priorities) - set(class_names))
    if missing or extra:
        raise ValueError(f"class priority mismatch: missing={missing}, extra={extra}")
    for class_name, value in class_priorities.items():
        if not 0 <= value <= 1:
            raise ValueError(f"class priority for {class_name!r} must be in [0, 1]")


def clip01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def semantic_value(
    *,
    class_priority: float,
    confidence: float,
    area_ratio: float,
    weights: Mapping[str, float] = SEMANTIC_WEIGHTS,
) -> float:
    return clip01(
        weights["class"] * class_priority
        + weights["confidence"] * confidence
        + weights["area"] * area_ratio
    )


def polygon_center(polygon: Sequence[Point]) -> Point:
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


def grid_assignment(
    center_x: float,
    center_y: float,
    image_width: int,
    image_height: int,
    *,
    grid_rows: int = 8,
    grid_cols: int = 8,
) -> tuple[int, int, int]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid dimensions must be positive")
    col = min(max(int(center_x / (image_width / grid_cols)), 0), grid_cols - 1)
    row = min(max(int(center_y / (image_height / grid_rows)), 0), grid_rows - 1)
    return row, col, row * grid_cols + col


def aosi_grid_value(values: Iterable[float]) -> float:
    product = 1.0
    for value in values:
        product *= 1.0 - clip01(value)
    return 1.0 - product


def find_image_path(image_dir: Path, image_id: str) -> Path:
    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing source image for {image_id!r} under {image_dir}")


def load_image_infos(image_dir: Path, image_ids: Iterable[str]) -> dict[str, ImageInfo]:
    infos: dict[str, ImageInfo] = {}
    for image_id in sorted(set(image_ids)):
        path = find_image_path(image_dir, image_id)
        with Image.open(path) as image:
            width, height = image.size
        infos[image_id] = ImageInfo(
            image_id=image_id,
            path=path.resolve(),
            width=width,
            height=height,
        )
    return infos


def _record_sort_key(record: RoiRecord) -> tuple[object, ...]:
    return (
        record.image_id,
        record.grid_id,
        record.class_id,
        -record.confidence,
        record.source_tile_id,
    )


def make_roi_records(
    predictions: Iterable[RestoredPrediction],
    image_infos: Mapping[str, ImageInfo],
    output_dir: Path,
    *,
    grid_rows: int = 8,
    grid_cols: int = 8,
    class_priorities: Mapping[str, float] = TASK_CLASS_PRIORITIES,
    weights: Mapping[str, float] = SEMANTIC_WEIGHTS,
    make_crops: bool = True,
) -> list[RoiRecord]:
    validate_class_priorities(class_priorities)
    drafts: list[RoiRecord] = []
    for prediction in predictions:
        if prediction.source_image_id not in image_infos:
            raise KeyError(f"missing source image info for {prediction.source_image_id!r}")
        info = image_infos[prediction.source_image_id]
        area_pixels = polygon_area(prediction.polygon)
        image_area = info.width * info.height
        area_ratio = area_pixels / image_area
        center_x, center_y = polygon_center(prediction.polygon)
        grid_row, grid_col, grid_id = grid_assignment(
            center_x,
            center_y,
            info.width,
            info.height,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        class_priority = class_priorities[prediction.class_name]
        drafts.append(
            RoiRecord(
                roi_id="",
                image_id=prediction.source_image_id,
                source_image_path=str(info.path),
                crop_path="",
                predicted_class=prediction.class_name,
                class_id=prediction.class_id,
                confidence=prediction.confidence,
                global_obb=prediction.polygon,
                center_x=center_x,
                center_y=center_y,
                area_pixels=area_pixels,
                area_ratio=area_ratio,
                grid_row=grid_row,
                grid_col=grid_col,
                grid_id=grid_id,
                semantic_value=semantic_value(
                    class_priority=class_priority,
                    confidence=prediction.confidence,
                    area_ratio=area_ratio,
                    weights=weights,
                ),
                class_priority=class_priority,
                difficult=0,
                source_tile_id=prediction.tile_id,
            )
        )

    counters: defaultdict[str, int] = defaultdict(int)
    records: list[RoiRecord] = []
    crop_dir = output_dir / "crops"
    for draft in sorted(drafts, key=_record_sort_key):
        roi_index = counters[draft.image_id]
        counters[draft.image_id] += 1
        roi_id = f"{draft.image_id}__roi{roi_index:06d}"
        crop_path = str((crop_dir / f"{roi_id}.png").resolve()) if make_crops else ""
        records.append(replace(draft, roi_id=roi_id, crop_path=crop_path))
    return records


def _edge_length(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _crop_size(polygon: Sequence[Point]) -> tuple[int, int]:
    width = max(_edge_length(polygon[0], polygon[1]), _edge_length(polygon[2], polygon[3]))
    height = max(_edge_length(polygon[1], polygon[2]), _edge_length(polygon[3], polygon[0]))
    return max(int(round(width)), 1), max(int(round(height)), 1)


def _warp_crop_array(image_array, polygon: Sequence[Point]):
    import cv2
    import numpy as np

    width, height = _crop_size(polygon)
    source = np.asarray(polygon, dtype=np.float32)
    destination = np.asarray(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image_array, transform, (width, height))


def write_rotated_crop(image_path: Path, polygon: Sequence[Point], output_path: Path) -> None:
    image_array = _load_image_array(image_path)
    crop = _warp_crop_array(image_array, polygon)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop).save(output_path)


def _load_image_array(image_path: Path):
    import numpy as np

    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def write_roi_crops(records: Iterable[RoiRecord]) -> int:
    records_by_image: defaultdict[str, list[RoiRecord]] = defaultdict(list)
    for record in records:
        if record.crop_path:
            records_by_image[record.source_image_path].append(record)

    written = 0
    for source_image_path, image_records in records_by_image.items():
        image_array = _load_image_array(Path(source_image_path))
        for record in image_records:
            crop = _warp_crop_array(image_array, record.global_obb)
            output_path = Path(record.crop_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop).save(output_path)
            written += 1
    return written


def _record_payload(record: RoiRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["global_obb"] = [[x, y] for x, y in record.global_obb]
    return payload


def write_metadata_jsonl(path: Path, records: Iterable[RoiRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_record_payload(record), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def write_metadata_csv(path: Path, records: Iterable[RoiRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROI_FIELDNAMES)
        writer.writeheader()
        for record in records:
            payload = _record_payload(record)
            payload["global_obb"] = json.dumps(payload["global_obb"])
            writer.writerow(payload)


def build_grid_summary(
    records: Iterable[RoiRecord],
    image_infos: Mapping[str, ImageInfo],
    *,
    grid_rows: int = 8,
    grid_cols: int = 8,
) -> dict[str, object]:
    records_by_image_grid: defaultdict[tuple[str, int], list[RoiRecord]] = defaultdict(list)
    for record in records:
        records_by_image_grid[(record.image_id, record.grid_id)].append(record)

    images: dict[str, object] = {}
    all_values: list[float] = []
    non_empty_cells = 0
    for image_id, info in sorted(image_infos.items()):
        cells = []
        for grid_id in range(grid_rows * grid_cols):
            grid_records = records_by_image_grid[(image_id, grid_id)]
            values = [record.semantic_value for record in grid_records]
            confidences = [record.confidence for record in grid_records]
            value = aosi_grid_value(values)
            if grid_records:
                non_empty_cells += 1
            all_values.append(value)
            cells.append(
                {
                    "image_id": image_id,
                    "grid_id": grid_id,
                    "grid_row": grid_id // grid_cols,
                    "grid_col": grid_id % grid_cols,
                    "roi_count": len(grid_records),
                    "grid_value": value,
                    "max_semantic_value": max(values) if values else 0.0,
                    "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                }
            )
        images[image_id] = {
            "source_width": info.width,
            "source_height": info.height,
            "grid_cells": cells,
        }

    return {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "image_count": len(image_infos),
        "cell_count": len(image_infos) * grid_rows * grid_cols,
        "non_empty_cell_count": non_empty_cells,
        "mean_grid_value": sum(all_values) / len(all_values) if all_values else 0.0,
        "images": images,
    }


def build_roi_metadata_dataset(
    predictions_path: Path,
    image_dir: Path,
    output_dir: Path,
    *,
    grid_rows: int = 8,
    grid_cols: int = 8,
    limit_images: int | None = None,
    make_crops: bool = True,
) -> dict[str, object]:
    predictions = load_nms_predictions(predictions_path)
    selected_image_ids = sorted({prediction.source_image_id for prediction in predictions})
    if limit_images is not None:
        if limit_images <= 0:
            raise ValueError("limit_images must be positive")
        selected_image_ids = selected_image_ids[:limit_images]
        selected = set(selected_image_ids)
        predictions = [
            prediction
            for prediction in predictions
            if prediction.source_image_id in selected
        ]

    image_infos = load_image_infos(image_dir, selected_image_ids)
    records = make_roi_records(
        predictions,
        image_infos,
        output_dir,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        make_crops=make_crops,
    )
    crop_count = write_roi_crops(records) if make_crops else 0
    write_metadata_jsonl(output_dir / "roi_metadata.jsonl", records)
    write_metadata_csv(output_dir / "roi_metadata.csv", records)
    grid_summary = build_grid_summary(
        records,
        image_infos,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    (output_dir / "grid_summary.json").write_text(
        json.dumps(grid_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    class_counts = defaultdict(int)
    for record in records:
        class_counts[record.predicted_class] += 1
    summary = {
        "purpose": "ROI metadata for downstream semantic scheduling inputs.",
        "inputs": {
            "predictions_path": str(predictions_path.resolve()),
            "image_dir": str(image_dir.resolve()),
        },
        "outputs": {
            "output_dir": str(output_dir.resolve()),
            "metadata_jsonl": str((output_dir / "roi_metadata.jsonl").resolve()),
            "metadata_csv": str((output_dir / "roi_metadata.csv").resolve()),
            "grid_summary": str((output_dir / "grid_summary.json").resolve()),
            "crop_dir": str((output_dir / "crops").resolve()) if make_crops else "",
        },
        "configuration": {
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "semantic_weights": dict(SEMANTIC_WEIGHTS),
            "class_priorities": dict(TASK_CLASS_PRIORITIES),
            "limit_images": limit_images,
            "make_crops": make_crops,
        },
        "counts": {
            "image_count": len(image_infos),
            "roi_count": len(records),
            "crop_count": crop_count,
            "class_counts": {
                class_name: class_counts[class_name]
                for class_name in ULTRALYTICS_DOTA_V1_CLASSES
            },
            "grid_cell_count": grid_summary["cell_count"],
            "non_empty_grid_cell_count": grid_summary["non_empty_cell_count"],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary

