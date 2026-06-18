"""Build supervised GT ROI crop datasets for multi-exit training."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image

from src.data.dota_tiling import parse_dota_annotations, polygon_area
from src.data.roi_metadata import (
    TASK_CLASS_PRIORITIES,
    grid_assignment,
    polygon_center,
    semantic_value,
)
from src.detector.dota_original_eval import CLASS_NAME_TO_ID, ULTRALYTICS_DOTA_V1_CLASSES


GT_ROI_FIELDNAMES = (
    "roi_id",
    "image_id",
    "source_image_path",
    "source_label_path",
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
    "gt_index",
    "split",
    "quality_label",
)


@dataclass(frozen=True)
class SplitManifestRecord:
    image_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class GtRoiRecord:
    roi_id: str
    image_id: str
    source_image_path: str
    source_label_path: str
    crop_path: str
    predicted_class: str
    class_id: int
    confidence: float
    global_obb: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
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
    gt_index: int
    split: str
    quality_label: str = "true_positive"


def load_split_manifest(path: Path, *, limit_images: int | None = None) -> list[SplitManifestRecord]:
    records: list[SplitManifestRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SplitManifestRecord(
                    image_id=row["image_id"],
                    image_path=Path(row["image_path"]),
                    label_path=Path(row["label_path"]),
                )
            )
            if limit_images is not None and len(records) >= limit_images:
                break
    if not records:
        raise ValueError(f"no split manifest records found in {path}")
    return records


def _load_image_array(image_path: Path):
    import numpy as np

    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB")), image.size


def _warp_crop_array(image_array, polygon):
    import cv2
    import numpy as np

    def edge_length(a, b) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    width = max(edge_length(polygon[0], polygon[1]), edge_length(polygon[2], polygon[3]))
    height = max(edge_length(polygon[1], polygon[2]), edge_length(polygon[3], polygon[0]))
    width_i = max(int(round(width)), 1)
    height_i = max(int(round(height)), 1)
    source = np.asarray(polygon, dtype=np.float32)
    destination = np.asarray(
        [
            [0.0, 0.0],
            [float(width_i - 1), 0.0],
            [float(width_i - 1), float(height_i - 1)],
            [0.0, float(height_i - 1)],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image_array, transform, (width_i, height_i))


def make_gt_roi_records(
    manifest_records: Iterable[SplitManifestRecord],
    output_dir: Path,
    *,
    split: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    include_difficult: bool = False,
    limit_objects: int | None = None,
) -> list[GtRoiRecord]:
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid dimensions must be positive")
    records: list[GtRoiRecord] = []
    crop_dir = output_dir / "crops"
    for manifest_record in manifest_records:
        with Image.open(manifest_record.image_path) as image:
            image_width, image_height = image.size
        _, objects = parse_dota_annotations(manifest_record.label_path)
        for gt_index, obj in enumerate(objects):
            if not include_difficult and obj.difficult != 0:
                continue
            if obj.class_name not in CLASS_NAME_TO_ID:
                raise ValueError(f"unknown DOTA class: {obj.class_name!r}")
            area_pixels = polygon_area(obj.polygon)
            if area_pixels <= 0:
                continue
            center_x, center_y = polygon_center(obj.polygon)
            grid_row, grid_col, grid_id = grid_assignment(
                center_x,
                center_y,
                image_width,
                image_height,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
            )
            class_priority = TASK_CLASS_PRIORITIES[obj.class_name]
            roi_id = f"{manifest_record.image_id}__gt{gt_index:06d}"
            crop_path = (crop_dir / f"{roi_id}.png").resolve()
            records.append(
                GtRoiRecord(
                    roi_id=roi_id,
                    image_id=manifest_record.image_id,
                    source_image_path=str(manifest_record.image_path.resolve()),
                    source_label_path=str(manifest_record.label_path.resolve()),
                    crop_path=str(crop_path),
                    predicted_class=obj.class_name,
                    class_id=CLASS_NAME_TO_ID[obj.class_name],
                    confidence=1.0,
                    global_obb=obj.polygon,
                    center_x=center_x,
                    center_y=center_y,
                    area_pixels=area_pixels,
                    area_ratio=area_pixels / (image_width * image_height),
                    grid_row=grid_row,
                    grid_col=grid_col,
                    grid_id=grid_id,
                    semantic_value=semantic_value(
                        class_priority=class_priority,
                        confidence=1.0,
                        area_ratio=area_pixels / (image_width * image_height),
                    ),
                    class_priority=class_priority,
                    difficult=obj.difficult,
                    gt_index=gt_index,
                    split=split,
                )
            )
            if limit_objects is not None and len(records) >= limit_objects:
                return records
    return records


def write_gt_roi_crops(records: Iterable[GtRoiRecord]) -> int:
    records_by_image: dict[str, list[GtRoiRecord]] = {}
    for record in records:
        records_by_image.setdefault(record.source_image_path, []).append(record)

    written = 0
    for image_path, image_records in records_by_image.items():
        image_array, _ = _load_image_array(Path(image_path))
        for record in image_records:
            crop = _warp_crop_array(image_array, record.global_obb)
            output_path = Path(record.crop_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop).save(output_path)
            written += 1
    return written


def _json_ready_record(record: GtRoiRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["global_obb"] = [[float(x), float(y)] for x, y in record.global_obb]
    return payload


def write_records_jsonl(path: Path, records: Iterable[GtRoiRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_ready_record(record), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_records_csv(path: Path, records: Iterable[GtRoiRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GT_ROI_FIELDNAMES)
        writer.writeheader()
        for record in records:
            payload = _json_ready_record(record)
            payload["global_obb"] = json.dumps(payload["global_obb"])
            writer.writerow(payload)


def summarize_records(records: list[GtRoiRecord]) -> dict[str, object]:
    class_counts = Counter(record.predicted_class for record in records)
    image_ids = {record.image_id for record in records}
    return {
        "image_count": len(image_ids),
        "roi_count": len(records),
        "class_counts": {
            class_name: class_counts[class_name]
            for class_name in ULTRALYTICS_DOTA_V1_CLASSES
        },
    }


def build_gt_roi_crop_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    split: str,
    grid_rows: int = 8,
    grid_cols: int = 8,
    include_difficult: bool = False,
    limit_images: int | None = None,
    limit_objects: int | None = None,
) -> dict[str, object]:
    if limit_images is not None and limit_images <= 0:
        raise ValueError("limit_images must be positive")
    if limit_objects is not None and limit_objects <= 0:
        raise ValueError("limit_objects must be positive")
    manifest_records = load_split_manifest(manifest_path, limit_images=limit_images)
    records = make_gt_roi_records(
        manifest_records,
        output_dir,
        split=split,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        include_difficult=include_difficult,
        limit_objects=limit_objects,
    )
    crop_count = write_gt_roi_crops(records)
    jsonl_path = output_dir / "roi_metadata_labeled.jsonl"
    csv_path = output_dir / "roi_metadata_labeled.csv"
    summary_path = output_dir / "summary.json"
    write_records_jsonl(jsonl_path, records)
    write_records_csv(csv_path, records)
    summary = {
        "purpose": "GT ROI crops for supervised multi-exit task model training.",
        "note": "These crops are supervision data, not detector-produced scheduling inputs.",
        "inputs": {
            "manifest_path": str(manifest_path.resolve()),
        },
        "outputs": {
            "output_dir": str(output_dir.resolve()),
            "metadata_jsonl": str(jsonl_path.resolve()),
            "metadata_csv": str(csv_path.resolve()),
            "crop_dir": str((output_dir / "crops").resolve()),
            "summary": str(summary_path.resolve()),
        },
        "configuration": {
            "split": split,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "include_difficult": include_difficult,
            "limit_images": limit_images,
            "limit_objects": limit_objects,
        },
        "counts": {
            **summarize_records(records),
            "crop_count": crop_count,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
