"""Match ROI metadata against original DOTA ground truth labels."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.data.dota_tiling import parse_dota_annotations, polygon_area
from src.data.roi_metadata import ROI_FIELDNAMES
from src.detector.dota_original_eval import (
    CLASS_NAME_TO_ID,
    ULTRALYTICS_DOTA_V1_CLASSES,
    rotated_iou,
)


Point = tuple[float, float]
Polygon = tuple[Point, Point, Point, Point]

ROI_LABEL_FIELDNAMES = (
    *ROI_FIELDNAMES,
    "matched_gt_class",
    "matched_gt_class_id",
    "matched_gt_iou",
    "matched_gt_difficult",
    "matched_gt_index",
    "matched_same_class_iou",
    "is_class_correct",
    "is_true_positive",
    "is_false_positive",
    "is_ignored",
    "quality_label",
)


@dataclass(frozen=True)
class RoiCandidate:
    payload: dict[str, object]
    roi_id: str
    image_id: str
    class_id: int
    predicted_class: str
    confidence: float
    polygon: Polygon


@dataclass(frozen=True)
class IndexedGroundTruth:
    image_id: str
    gt_index: int
    class_id: int
    class_name: str
    polygon: Polygon
    difficult: int


def _polygon_from_json(points: Sequence[Sequence[float]]) -> Polygon:
    if len(points) != 4:
        raise ValueError(f"expected 4 polygon points, got {len(points)}")
    polygon = tuple((float(point[0]), float(point[1])) for point in points)
    if polygon_area(polygon) <= 0:
        raise ValueError(f"degenerate polygon: {points!r}")
    return polygon  # type: ignore[return-value]


def load_roi_metadata(path: Path) -> list[RoiCandidate]:
    rois: list[RoiCandidate] = []
    seen_roi_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        missing = set(ROI_FIELDNAMES) - set(payload)
        if missing:
            raise ValueError(f"{path}:{line_number}: missing ROI fields {sorted(missing)}")
        roi_id = str(payload["roi_id"])
        if roi_id in seen_roi_ids:
            raise ValueError(f"{path}:{line_number}: duplicate roi_id {roi_id!r}")
        seen_roi_ids.add(roi_id)

        class_id = int(payload["class_id"])
        if not 0 <= class_id < len(ULTRALYTICS_DOTA_V1_CLASSES):
            raise ValueError(f"{path}:{line_number}: class_id out of range: {class_id}")
        predicted_class = str(payload["predicted_class"])
        expected_class = ULTRALYTICS_DOTA_V1_CLASSES[class_id]
        if predicted_class != expected_class:
            raise ValueError(
                f"{path}:{line_number}: predicted_class={predicted_class!r} "
                f"does not match class_id={class_id} ({expected_class!r})"
            )

        rois.append(
            RoiCandidate(
                payload=payload,
                roi_id=roi_id,
                image_id=str(payload["image_id"]),
                class_id=class_id,
                predicted_class=predicted_class,
                confidence=float(payload["confidence"]),
                polygon=_polygon_from_json(payload["global_obb"]),
            )
        )
    return rois


def load_indexed_ground_truths(
    label_dir: Path,
    *,
    image_ids: Iterable[str] | None = None,
) -> list[IndexedGroundTruth]:
    selected_ids = set(image_ids) if image_ids is not None else None
    ground_truths: list[IndexedGroundTruth] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_id = label_path.stem
        if selected_ids is not None and image_id not in selected_ids:
            continue
        _, objects = parse_dota_annotations(label_path)
        for gt_index, obj in enumerate(objects):
            if obj.class_name not in CLASS_NAME_TO_ID:
                raise ValueError(f"{label_path}: unknown class {obj.class_name!r}")
            ground_truths.append(
                IndexedGroundTruth(
                    image_id=image_id,
                    gt_index=gt_index,
                    class_id=CLASS_NAME_TO_ID[obj.class_name],
                    class_name=obj.class_name,
                    polygon=obj.polygon,
                    difficult=obj.difficult,
                )
            )
    return ground_truths


def _best_match(
    roi: RoiCandidate,
    ground_truths: Sequence[IndexedGroundTruth],
) -> tuple[IndexedGroundTruth | None, float]:
    best_gt: IndexedGroundTruth | None = None
    best_iou = 0.0
    for ground_truth in ground_truths:
        iou = rotated_iou(roi.polygon, ground_truth.polygon)
        if iou > best_iou:
            best_gt = ground_truth
            best_iou = iou
    return best_gt, best_iou


def _gt_key(ground_truth: IndexedGroundTruth) -> tuple[str, int]:
    return ground_truth.image_id, ground_truth.gt_index


def _base_label_payload(
    roi: RoiCandidate,
    best_gt: IndexedGroundTruth | None,
    best_iou: float,
    same_class_iou: float,
    *,
    iou_threshold: float,
) -> dict[str, object]:
    matched_gt = best_gt if best_gt is not None and best_iou > 0 else None
    return {
        "matched_gt_class": matched_gt.class_name if matched_gt else "",
        "matched_gt_class_id": matched_gt.class_id if matched_gt else -1,
        "matched_gt_iou": best_iou if matched_gt else 0.0,
        "matched_gt_difficult": matched_gt.difficult if matched_gt else -1,
        "matched_gt_index": matched_gt.gt_index if matched_gt else -1,
        "matched_same_class_iou": same_class_iou,
        "is_class_correct": int(
            best_gt is not None
            and best_iou >= iou_threshold
            and best_gt.class_id == roi.class_id
        ),
        "is_true_positive": 0,
        "is_false_positive": 1,
        "is_ignored": 0,
        "quality_label": "background_fp",
    }


def _row_from_label(roi: RoiCandidate, label_payload: Mapping[str, object]) -> dict[str, object]:
    row = {field: roi.payload[field] for field in ROI_FIELDNAMES}
    row.update(label_payload)
    return row


def label_roi_records(
    rois: Iterable[RoiCandidate],
    ground_truths: Iterable[IndexedGroundTruth],
    *,
    iou_threshold: float = 0.5,
) -> list[dict[str, object]]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")

    roi_list = list(rois)
    ground_truth_list = list(ground_truths)
    gts_by_image: dict[str, list[IndexedGroundTruth]] = defaultdict(list)
    valid_by_image_class: dict[tuple[str, int], list[IndexedGroundTruth]] = defaultdict(list)
    ignored_by_image_class: dict[tuple[str, int], list[IndexedGroundTruth]] = defaultdict(list)
    for ground_truth in ground_truth_list:
        gts_by_image[ground_truth.image_id].append(ground_truth)
        key = (ground_truth.image_id, ground_truth.class_id)
        if ground_truth.difficult == 0:
            valid_by_image_class[key].append(ground_truth)
        else:
            ignored_by_image_class[key].append(ground_truth)

    matched_valid_gts: set[tuple[str, int]] = set()
    label_by_roi_id: dict[str, dict[str, object]] = {}
    for roi in sorted(
        roi_list,
        key=lambda item: (item.class_id, -item.confidence, item.image_id, item.roi_id),
    ):
        best_gt, best_iou = _best_match(roi, gts_by_image.get(roi.image_id, []))
        same_class_valid_gt, same_class_valid_iou = _best_match(
            roi,
            valid_by_image_class.get((roi.image_id, roi.class_id), []),
        )
        same_class_ignored_gt, same_class_ignored_iou = _best_match(
            roi,
            ignored_by_image_class.get((roi.image_id, roi.class_id), []),
        )
        same_class_iou = max(same_class_valid_iou, same_class_ignored_iou)
        label_payload = _base_label_payload(
            roi,
            best_gt,
            best_iou,
            same_class_iou,
            iou_threshold=iou_threshold,
        )

        if same_class_valid_gt is not None and same_class_valid_iou >= iou_threshold:
            key = _gt_key(same_class_valid_gt)
            if key not in matched_valid_gts:
                matched_valid_gts.add(key)
                label_payload.update(
                    {
                        "is_true_positive": 1,
                        "is_false_positive": 0,
                        "quality_label": "true_positive",
                    }
                )
            else:
                label_payload["quality_label"] = "duplicate_detection"
        elif same_class_ignored_gt is not None and same_class_ignored_iou >= iou_threshold:
            label_payload.update(
                {
                    "is_false_positive": 0,
                    "is_ignored": 1,
                    "quality_label": "ignored_difficult",
                }
            )
        elif best_gt is not None and best_iou >= iou_threshold and best_gt.class_id != roi.class_id:
            label_payload["quality_label"] = "class_mismatch"

        label_by_roi_id[roi.roi_id] = label_payload

    return [
        _row_from_label(roi, label_by_roi_id[roi.roi_id])
        for roi in sorted(roi_list, key=lambda item: item.roi_id)
    ]


def _json_ready_row(row: Mapping[str, object]) -> dict[str, object]:
    payload = dict(row)
    payload["global_obb"] = [
        [float(point[0]), float(point[1])]
        for point in payload["global_obb"]  # type: ignore[index]
    ]
    return payload


def write_labeled_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_json_ready_row(row), ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_labeled_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROI_LABEL_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            payload = _json_ready_row(row)
            payload["global_obb"] = json.dumps(payload["global_obb"])
            writer.writerow(payload)


def summarize_labeled_rows(
    rows: Sequence[Mapping[str, object]],
    ground_truths: Sequence[IndexedGroundTruth],
    *,
    iou_threshold: float,
) -> dict[str, object]:
    quality_counts = Counter(str(row["quality_label"]) for row in rows)
    predicted_counts = Counter(str(row["predicted_class"]) for row in rows)
    gt_counts = Counter(gt.class_name for gt in ground_truths if gt.difficult == 0)
    ignored_gt_counts = Counter(gt.class_name for gt in ground_truths if gt.difficult != 0)

    per_class: dict[str, dict[str, object]] = {}
    for class_name in ULTRALYTICS_DOTA_V1_CLASSES:
        class_rows = [row for row in rows if row["predicted_class"] == class_name]
        tp = sum(int(row["is_true_positive"]) for row in class_rows)
        fp = sum(int(row["is_false_positive"]) for row in class_rows)
        ignored = sum(int(row["is_ignored"]) for row in class_rows)
        per_class[class_name] = {
            "predictions": predicted_counts[class_name],
            "true_positives": tp,
            "false_positives": fp,
            "ignored_predictions": ignored,
            "ground_truth": gt_counts[class_name],
            "ignored_ground_truth": ignored_gt_counts[class_name],
            "precision": tp / max(tp + fp, 1),
            "recall": tp / gt_counts[class_name] if gt_counts[class_name] else None,
        }

    return {
        "iou_threshold": iou_threshold,
        "counts": {
            "roi_count": len(rows),
            "ground_truth": sum(gt_counts.values()),
            "ignored_ground_truth": sum(ignored_gt_counts.values()),
            "true_positive": quality_counts["true_positive"],
            "false_positive": sum(
                quality_counts[label]
                for label in ("duplicate_detection", "class_mismatch", "background_fp")
            ),
            "ignored_prediction": quality_counts["ignored_difficult"],
            "duplicate_detection": quality_counts["duplicate_detection"],
            "class_mismatch": quality_counts["class_mismatch"],
            "background_fp": quality_counts["background_fp"],
            "matched_any_gt": sum(float(row["matched_gt_iou"]) >= iou_threshold for row in rows),
            "class_correct": sum(int(row["is_class_correct"]) for row in rows),
        },
        "quality_label_counts": dict(sorted(quality_counts.items())),
        "per_class": per_class,
    }


def build_labeled_roi_metadata_dataset(
    metadata_path: Path,
    label_dir: Path,
    output_dir: Path,
    *,
    iou_threshold: float = 0.5,
    limit_images: int | None = None,
) -> dict[str, object]:
    rois = load_roi_metadata(metadata_path)
    selected_image_ids = sorted({roi.image_id for roi in rois})
    if limit_images is not None:
        if limit_images <= 0:
            raise ValueError("limit_images must be positive")
        selected_image_ids = selected_image_ids[:limit_images]
        selected = set(selected_image_ids)
        rois = [roi for roi in rois if roi.image_id in selected]

    ground_truths = load_indexed_ground_truths(label_dir, image_ids=selected_image_ids)
    rows = label_roi_records(rois, ground_truths, iou_threshold=iou_threshold)
    jsonl_path = output_dir / "roi_metadata_labeled.jsonl"
    csv_path = output_dir / "roi_metadata_labeled.csv"
    summary_path = output_dir / "roi_label_summary.json"
    write_labeled_jsonl(jsonl_path, rows)
    write_labeled_csv(csv_path, rows)

    summary = {
        "purpose": "ROI metadata labels from original DOTA ground truths for downstream task-quality modeling.",
        "inputs": {
            "metadata_path": str(metadata_path.resolve()),
            "label_dir": str(label_dir.resolve()),
        },
        "outputs": {
            "metadata_labeled_jsonl": str(jsonl_path.resolve()),
            "metadata_labeled_csv": str(csv_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
        "configuration": {
            "iou_threshold": iou_threshold,
            "limit_images": limit_images,
        },
        **summarize_labeled_rows(rows, ground_truths, iou_threshold=iou_threshold),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
