"""Original-image DOTA-style evaluation for tiled YOLO-OBB predictions."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.data.dota_tiling import parse_dota_annotations, polygon_area


Point = tuple[float, float]
Polygon = tuple[Point, Point, Point, Point]

# Keep this local to avoid importing src.data.dota_yolo_obb, which currently
# depends on PyYAML at import time.
ULTRALYTICS_DOTA_V1_CLASSES = (
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
)
CLASS_NAME_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(ULTRALYTICS_DOTA_V1_CLASSES)
}


@dataclass(frozen=True)
class TileInfo:
    tile_id: str
    source_image_id: str
    source_image_path: str
    source_label_path: str
    offset_x: int
    offset_y: int
    source_width: int
    source_height: int
    crop_width: int
    crop_height: int
    tile_width: int
    tile_height: int


@dataclass(frozen=True)
class TilePrediction:
    tile_id: str
    class_id: int
    class_name: str
    confidence: float
    polygon: Polygon


@dataclass(frozen=True)
class RestoredPrediction:
    tile_id: str
    source_image_id: str
    class_id: int
    class_name: str
    confidence: float
    polygon: Polygon


@dataclass(frozen=True)
class GroundTruth:
    image_id: str
    class_id: int
    class_name: str
    polygon: Polygon
    difficult: int


def _polygon_from_points(points: Sequence[Sequence[float]]) -> Polygon:
    if len(points) != 4:
        raise ValueError(f"expected 4 polygon points, got {len(points)}")
    polygon = tuple((float(point[0]), float(point[1])) for point in points)
    if polygon_area(polygon) <= 0:
        raise ValueError(f"degenerate polygon: {points!r}")
    return polygon  # type: ignore[return-value]


def load_tile_mappings(path: Path) -> dict[str, TileInfo]:
    mappings: dict[str, TileInfo] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        info = TileInfo(
            tile_id=payload["tile_id"],
            source_image_id=payload["source_image_id"],
            source_image_path=payload["source_image_path"],
            source_label_path=payload["source_label_path"],
            offset_x=int(payload["offset_x"]),
            offset_y=int(payload["offset_y"]),
            source_width=int(payload["source_width"]),
            source_height=int(payload["source_height"]),
            crop_width=int(payload["crop_width"]),
            crop_height=int(payload["crop_height"]),
            tile_width=int(payload["tile_width"]),
            tile_height=int(payload["tile_height"]),
        )
        if info.tile_id in mappings:
            raise ValueError(f"{path}:{line_number}: duplicate tile_id {info.tile_id!r}")
        mappings[info.tile_id] = info
    return mappings


def polygon_center(polygon: Sequence[Point]) -> Point:
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )


def restore_tile_prediction(
    prediction: TilePrediction,
    mapping: TileInfo,
    *,
    drop_padding_center: bool = True,
) -> RestoredPrediction | None:
    center_x, center_y = polygon_center(prediction.polygon)
    if drop_padding_center and (
        center_x < 0
        or center_y < 0
        or center_x > mapping.crop_width
        or center_y > mapping.crop_height
    ):
        return None

    restored = tuple(
        (
            min(max(x + mapping.offset_x, 0.0), float(mapping.source_width)),
            min(max(y + mapping.offset_y, 0.0), float(mapping.source_height)),
        )
        for x, y in prediction.polygon
    )
    if polygon_area(restored) <= 0:
        return None
    return RestoredPrediction(
        tile_id=prediction.tile_id,
        source_image_id=mapping.source_image_id,
        class_id=prediction.class_id,
        class_name=prediction.class_name,
        confidence=prediction.confidence,
        polygon=restored,  # type: ignore[arg-type]
    )


def restore_predictions(
    predictions: Iterable[TilePrediction],
    mappings: Mapping[str, TileInfo],
) -> list[RestoredPrediction]:
    restored: list[RestoredPrediction] = []
    for prediction in predictions:
        if prediction.tile_id not in mappings:
            raise KeyError(f"missing tile mapping for prediction {prediction.tile_id!r}")
        restored_prediction = restore_tile_prediction(prediction, mappings[prediction.tile_id])
        if restored_prediction is not None:
            restored.append(restored_prediction)
    return restored


def _signed_polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    return (
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def _line_intersection(a: Point, b: Point, c: Point, d: Point) -> Point:
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return b
    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return px, py


def convex_polygon_intersection(subject: Sequence[Point], clip: Sequence[Point]) -> list[Point]:
    """Clip a convex subject polygon by another convex polygon."""
    output = list(subject)
    if len(output) < 3 or len(clip) < 3:
        return []
    orientation = 1.0 if _signed_polygon_area(clip) >= 0 else -1.0

    def inside(point: Point, edge_start: Point, edge_end: Point) -> bool:
        cross = (
            (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
        )
        return cross * orientation >= -1e-9

    for index, edge_start in enumerate(clip):
        edge_end = clip[(index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = inside(previous, edge_start, edge_end)
        for current in input_points:
            current_inside = inside(current, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous = current
            previous_inside = current_inside
    return output


def rotated_iou(polygon_a: Sequence[Point], polygon_b: Sequence[Point]) -> float:
    area_a = polygon_area(polygon_a)
    area_b = polygon_area(polygon_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection = convex_polygon_intersection(polygon_a, polygon_b)
    intersection_area = polygon_area(intersection)
    if intersection_area <= 0:
        return 0.0
    union = area_a + area_b - intersection_area
    return intersection_area / union if union > 0 else 0.0


def global_rotated_nms(
    predictions: Iterable[RestoredPrediction],
    *,
    iou_threshold: float = 0.1,
) -> list[RestoredPrediction]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    grouped: dict[tuple[str, int], list[RestoredPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[(prediction.source_image_id, prediction.class_id)].append(prediction)

    kept: list[RestoredPrediction] = []
    for group in grouped.values():
        candidates = sorted(group, key=lambda item: item.confidence, reverse=True)
        while candidates:
            chosen = candidates.pop(0)
            kept.append(chosen)
            candidates = [
                candidate
                for candidate in candidates
                if rotated_iou(chosen.polygon, candidate.polygon) <= iou_threshold
            ]
    return sorted(
        kept,
        key=lambda item: (item.source_image_id, item.class_id, -item.confidence, item.tile_id),
    )


def load_ground_truths(
    label_dir: Path,
    *,
    image_ids: Iterable[str] | None = None,
) -> list[GroundTruth]:
    selected_ids = set(image_ids) if image_ids is not None else None
    ground_truths: list[GroundTruth] = []
    for label_path in sorted(label_dir.glob("*.txt")):
        image_id = label_path.stem
        if selected_ids is not None and image_id not in selected_ids:
            continue
        _, objects = parse_dota_annotations(label_path)
        for obj in objects:
            if obj.class_name not in CLASS_NAME_TO_ID:
                raise ValueError(f"{label_path}: unknown class {obj.class_name!r}")
            ground_truths.append(
                GroundTruth(
                    image_id=image_id,
                    class_id=CLASS_NAME_TO_ID[obj.class_name],
                    class_name=obj.class_name,
                    polygon=obj.polygon,
                    difficult=obj.difficult,
                )
            )
    return ground_truths


def voc_ap(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def _match_class_predictions(
    predictions: list[RestoredPrediction],
    ground_truths: list[GroundTruth],
    *,
    iou_threshold: float,
) -> dict[str, object]:
    gt_by_image: dict[str, list[GroundTruth]] = defaultdict(list)
    ignored_by_image: dict[str, list[GroundTruth]] = defaultdict(list)
    for ground_truth in ground_truths:
        if ground_truth.difficult == 0:
            gt_by_image[ground_truth.image_id].append(ground_truth)
        else:
            ignored_by_image[ground_truth.image_id].append(ground_truth)

    matched: set[tuple[str, int]] = set()
    tp_values: list[int] = []
    fp_values: list[int] = []
    ignored_predictions = 0

    for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
        valid_gts = gt_by_image.get(prediction.source_image_id, [])
        best_iou = 0.0
        best_index = -1
        for index, ground_truth in enumerate(valid_gts):
            iou = rotated_iou(prediction.polygon, ground_truth.polygon)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_iou >= iou_threshold:
            match_key = (prediction.source_image_id, best_index)
            if match_key not in matched:
                matched.add(match_key)
                tp_values.append(1)
                fp_values.append(0)
            else:
                tp_values.append(0)
                fp_values.append(1)
            continue

        ignored_iou = max(
            (
                rotated_iou(prediction.polygon, ignored_gt.polygon)
                for ignored_gt in ignored_by_image.get(prediction.source_image_id, [])
            ),
            default=0.0,
        )
        if ignored_iou >= iou_threshold:
            ignored_predictions += 1
            continue
        tp_values.append(0)
        fp_values.append(1)

    npos = sum(len(items) for items in gt_by_image.values())
    tp_cumulative: list[int] = []
    fp_cumulative: list[int] = []
    tp_total = 0
    fp_total = 0
    for tp, fp in zip(tp_values, fp_values):
        tp_total += tp
        fp_total += fp
        tp_cumulative.append(tp_total)
        fp_cumulative.append(fp_total)

    recalls = [tp / npos if npos else 0.0 for tp in tp_cumulative]
    precisions = [
        tp / max(tp + fp, 1)
        for tp, fp in zip(tp_cumulative, fp_cumulative)
    ]
    return {
        "AP50": voc_ap(recalls, precisions) if npos else None,
        "precision": tp_total / max(tp_total + fp_total, 1),
        "recall": tp_total / npos if npos else None,
        "true_positives": tp_total,
        "false_positives": fp_total,
        "ignored_predictions": ignored_predictions,
        "ground_truth": npos,
        "ignored_ground_truth": sum(len(items) for items in ignored_by_image.values()),
        "predictions": len(predictions),
    }


def evaluate_predictions(
    predictions: Iterable[RestoredPrediction],
    ground_truths: Iterable[GroundTruth],
    *,
    iou_threshold: float = 0.5,
    class_names: Sequence[str] = ULTRALYTICS_DOTA_V1_CLASSES,
) -> dict[str, object]:
    predictions_by_class: dict[int, list[RestoredPrediction]] = defaultdict(list)
    ground_truths_by_class: dict[int, list[GroundTruth]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_class[prediction.class_id].append(prediction)
    for ground_truth in ground_truths:
        ground_truths_by_class[ground_truth.class_id].append(ground_truth)

    per_class: dict[str, dict[str, object]] = {}
    ap_values: list[float] = []
    total_tp = 0
    total_fp = 0
    total_gt = 0
    total_ignored_gt = 0
    for class_id, class_name in enumerate(class_names):
        metrics = _match_class_predictions(
            predictions_by_class[class_id],
            ground_truths_by_class[class_id],
            iou_threshold=iou_threshold,
        )
        per_class[class_name] = metrics
        if metrics["AP50"] is not None:
            ap_values.append(float(metrics["AP50"]))
        total_tp += int(metrics["true_positives"])
        total_fp += int(metrics["false_positives"])
        total_gt += int(metrics["ground_truth"])
        total_ignored_gt += int(metrics["ignored_ground_truth"])

    return {
        "aggregate": {
            "precision": total_tp / max(total_tp + total_fp, 1),
            "recall": total_tp / total_gt if total_gt else None,
            "mAP50": sum(ap_values) / len(ap_values) if ap_values else None,
            "true_positives": total_tp,
            "false_positives": total_fp,
        },
        "per_class": per_class,
        "counts": {
            "ground_truth_objects": total_gt,
            "ignored_difficult_gt": total_ignored_gt,
        },
    }


def _prediction_to_json(prediction: TilePrediction | RestoredPrediction) -> dict[str, object]:
    payload = asdict(prediction)
    payload["polygon"] = [[x, y] for x, y in prediction.polygon]
    return payload


def write_predictions_jsonl(
    path: Path,
    predictions: Iterable[TilePrediction | RestoredPrediction],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_prediction_to_json(prediction), ensure_ascii=False) + "\n"
            for prediction in predictions
        ),
        encoding="utf-8",
    )


def write_dota_task1(
    output_dir: Path,
    predictions: Iterable[RestoredPrediction],
    *,
    class_names: Sequence[str] = ULTRALYTICS_DOTA_V1_CLASSES,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_class: dict[str, list[RestoredPrediction]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_class[prediction.class_name].append(prediction)

    for class_name in class_names:
        rows = sorted(
            predictions_by_class[class_name],
            key=lambda item: (-item.confidence, item.source_image_id, item.tile_id),
        )
        lines = []
        for prediction in rows:
            coordinates = " ".join(
                f"{value:.2f}"
                for point in prediction.polygon
                for value in point
            )
            lines.append(
                f"{prediction.source_image_id} {prediction.confidence:.6f} {coordinates}\n"
            )
        (output_dir / f"Task1_{class_name}.txt").write_text(
            "".join(lines),
            encoding="utf-8",
        )


def write_per_image_summary(
    path: Path,
    image_ids: Iterable[str],
    *,
    restored_predictions: Iterable[RestoredPrediction],
    nms_predictions: Iterable[RestoredPrediction],
    ground_truths: Iterable[GroundTruth],
) -> None:
    restored_counts = Counter(prediction.source_image_id for prediction in restored_predictions)
    nms_counts = Counter(prediction.source_image_id for prediction in nms_predictions)
    gt_counts = Counter(gt.image_id for gt in ground_truths if gt.difficult == 0)
    ignored_counts = Counter(gt.image_id for gt in ground_truths if gt.difficult != 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "restored_predictions",
                "nms_predictions",
                "ground_truth",
                "ignored_difficult_gt",
            ],
        )
        writer.writeheader()
        for image_id in sorted(set(image_ids)):
            writer.writerow(
                {
                    "image_id": image_id,
                    "restored_predictions": restored_counts[image_id],
                    "nms_predictions": nms_counts[image_id],
                    "ground_truth": gt_counts[image_id],
                    "ignored_difficult_gt": ignored_counts[image_id],
                }
            )


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
