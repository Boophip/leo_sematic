"""Prepare tiled DOTA annotations for Ultralytics YOLO-OBB training."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

import yaml
from PIL import Image

from src.data.dota_tiling import DotaObject, parse_dota_annotations, polygon_area


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
CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(ULTRALYTICS_DOTA_V1_CLASSES)
}
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class YoloTileRecord:
    tile_id: str
    image_path: str
    label_path: str
    source_image_id: str
    class_ids: tuple[int, ...]
    object_count: int


def format_yolo_obb_object(
    obj: DotaObject,
    image_width: int,
    image_height: int,
    *,
    excluded_difficulties: frozenset[int] = frozenset({2}),
) -> str | None:
    """Convert one DOTA quadrilateral to normalized Ultralytics YOLO-OBB."""
    if obj.difficult in excluded_difficulties:
        return None
    if obj.class_name not in CLASS_TO_ID:
        raise ValueError(f"unknown DOTA-v1.0 class {obj.class_name!r}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if polygon_area(obj.polygon) <= 0:
        raise ValueError(f"degenerate polygon for class {obj.class_name!r}")

    normalized: list[float] = []
    for x, y in obj.polygon:
        normalized.extend(
            (
                min(max(x / image_width, 0.0), 1.0),
                min(max(y / image_height, 0.0), 1.0),
            )
        )
    line = f"{CLASS_TO_ID[obj.class_name]} " + " ".join(
        f"{coordinate:.8f}" for coordinate in normalized
    )
    validate_yolo_obb_line(line)
    return line


def validate_yolo_obb_line(line: str) -> tuple[int, tuple[float, ...]]:
    """Validate and parse one normalized YOLO-OBB label line."""
    fields = line.split()
    if len(fields) != 9:
        raise ValueError(f"YOLO-OBB line must contain 9 fields, got {len(fields)}")
    try:
        class_id = int(fields[0])
        coordinates = tuple(float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(f"invalid YOLO-OBB values: {line!r}") from error
    if not 0 <= class_id < len(ULTRALYTICS_DOTA_V1_CLASSES):
        raise ValueError(f"class ID out of range: {class_id}")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coordinates):
        raise ValueError(f"normalized coordinate out of range: {line!r}")
    polygon = tuple(
        (coordinates[index], coordinates[index + 1])
        for index in range(0, len(coordinates), 2)
    )
    if polygon_area(polygon) <= 0:
        raise ValueError(f"degenerate normalized polygon: {line!r}")
    return class_id, coordinates


def validate_yolo_label_file(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            class_id, _ = validate_yolo_obb_line(line)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        counts[class_id] += 1
    return counts


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _find_images(path: Path) -> list[Path]:
    return sorted(
        image_path
        for image_path in path.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def _source_image_id(tile_id: str) -> str:
    marker = "__x"
    if marker not in tile_id:
        raise ValueError(f"tile ID does not contain an original-image mapping: {tile_id}")
    return tile_id.split(marker, 1)[0]


def convert_tiled_split(
    tiles_split_dir: Path,
    output_root: Path,
    split_name: str,
) -> tuple[list[YoloTileRecord], dict[str, object]]:
    """Convert one tiled DOTA split into the Ultralytics directory layout."""
    source_image_dir = tiles_split_dir / "images"
    source_label_dir = tiles_split_dir / "labelTxt"
    output_image_dir = output_root / "images" / split_name
    output_label_dir = output_root / "labels" / split_name
    output_image_dir.mkdir(parents=True, exist_ok=False)
    output_label_dir.mkdir(parents=True, exist_ok=False)

    records: list[YoloTileRecord] = []
    class_counts: Counter[int] = Counter()
    original_objects = 0
    excluded_difficult_2 = 0
    hardlinks = 0
    copies = 0
    empty_labels = 0
    for image_path in _find_images(source_image_dir):
        source_label_path = source_label_dir / f"{image_path.stem}.txt"
        if not source_label_path.is_file():
            raise FileNotFoundError(f"missing tiled DOTA label: {source_label_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        _, objects = parse_dota_annotations(source_label_path)
        original_objects += len(objects)
        excluded_difficult_2 += sum(obj.difficult == 2 for obj in objects)

        label_lines = [
            line
            for obj in objects
            if (line := format_yolo_obb_object(obj, width, height)) is not None
        ]
        output_label_path = output_label_dir / f"{image_path.stem}.txt"
        output_label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        per_tile_counts = validate_yolo_label_file(output_label_path)
        class_counts.update(per_tile_counts)
        empty_labels += int(not label_lines)

        output_image_path = output_image_dir / image_path.name
        materialization = _link_or_copy(image_path, output_image_path)
        hardlinks += int(materialization == "hardlink")
        copies += int(materialization == "copy")
        records.append(
            YoloTileRecord(
                tile_id=image_path.stem,
                image_path=str(output_image_path.resolve()),
                label_path=str(output_label_path.resolve()),
                source_image_id=_source_image_id(image_path.stem),
                class_ids=tuple(sorted(per_tile_counts)),
                object_count=sum(per_tile_counts.values()),
            )
        )

    summary: dict[str, object] = {
        "split": split_name,
        "images": len(records),
        "labels": len(records),
        "empty_labels": empty_labels,
        "original_tiled_objects": original_objects,
        "excluded_difficult_2": excluded_difficult_2,
        "kept_objects": sum(class_counts.values()),
        "hardlinked_images": hardlinks,
        "copied_images": copies,
        "class_object_counts": {
            ULTRALYTICS_DOTA_V1_CLASSES[class_id]: class_counts[class_id]
            for class_id in range(len(ULTRALYTICS_DOTA_V1_CLASSES))
        },
    }
    return records, summary


def select_smoke_records(
    records: Sequence[YoloTileRecord],
    size: int = 32,
    *,
    seed: int = 42,
) -> list[YoloTileRecord]:
    """Select deterministic non-empty tiles, prioritizing rare-class coverage."""
    candidates = [record for record in records if record.object_count > 0]
    if size <= 0:
        raise ValueError("smoke size must be positive")
    if len(candidates) < size:
        raise ValueError(
            f"requested {size} non-empty smoke tiles, but only {len(candidates)} exist"
        )

    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    presence: Counter[int] = Counter()
    for record in shuffled:
        presence.update(record.class_ids)
    target_density = median(record.object_count for record in shuffled)

    selected: list[YoloTileRecord] = []
    remaining = list(shuffled)
    covered: set[int] = set()
    while remaining and len(selected) < size:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                len(set(remaining[index].class_ids) - covered),
                sum(1.0 / presence[class_id] for class_id in remaining[index].class_ids),
                -abs(remaining[index].object_count - target_density),
                -remaining[index].object_count,
            ),
        )
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        covered.update(chosen.class_ids)
    return sorted(selected, key=lambda record: record.tile_id)


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_image_list(path: Path, records: Iterable[YoloTileRecord]) -> None:
    path.write_text(
        "".join(record.image_path.replace("\\", "/") + "\n" for record in records),
        encoding="utf-8",
    )


def _verify_source_isolation(records_by_split: dict[str, list[YoloTileRecord]]) -> None:
    source_ids = {
        split: {record.source_image_id for record in records}
        for split, records in records_by_split.items()
    }
    for index, split_a in enumerate(source_ids):
        for split_b in list(source_ids)[index + 1 :]:
            overlap = source_ids[split_a] & source_ids[split_b]
            if overlap:
                raise ValueError(
                    f"original-image leakage between {split_a} and {split_b}: "
                    f"{sorted(overlap)[:5]}"
                )


def prepare_yolo_obb_dataset(
    tiles_root: Path,
    output_root: Path,
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    smoke_size: int = 32,
    seed: int = 42,
) -> dict[str, object]:
    """Convert all tiled splits and create a deterministic overfit smoke dataset."""
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    records_by_split: dict[str, list[YoloTileRecord]] = {}
    split_summaries: dict[str, object] = {}
    for split in splits:
        records, summary = convert_tiled_split(tiles_root / split, output_root, split)
        records_by_split[split] = records
        split_summaries[split] = summary
    _verify_source_isolation(records_by_split)

    names = {
        class_id: class_name
        for class_id, class_name in enumerate(ULTRALYTICS_DOTA_V1_CLASSES)
    }
    dataset_yaml: dict[str, object] = {
        "path": str(output_root.resolve()),
        **{split: f"images/{split}" for split in splits},
        "names": names,
    }
    _write_yaml(output_root / "dataset.yaml", dataset_yaml)

    if "train" not in records_by_split:
        raise ValueError("train split is required to create the overfit smoke dataset")
    smoke_records = select_smoke_records(
        records_by_split["train"],
        size=smoke_size,
        seed=seed,
    )
    _write_image_list(output_root / f"overfit{smoke_size}_train.txt", smoke_records)
    _write_image_list(output_root / f"overfit{smoke_size}_val.txt", smoke_records)
    overfit_yaml = {
        "path": str(output_root.resolve()),
        "train": f"overfit{smoke_size}_train.txt",
        "val": f"overfit{smoke_size}_val.txt",
        "names": names,
    }
    _write_yaml(output_root / f"overfit{smoke_size}.yaml", overfit_yaml)

    smoke_class_presence: Counter[int] = Counter()
    smoke_class_objects: Counter[int] = Counter()
    for record in smoke_records:
        smoke_class_presence.update(record.class_ids)
        smoke_class_objects.update(validate_yolo_label_file(Path(record.label_path)))
    smoke_manifest = {
        "purpose": "Intentional train/val reuse for data-pipeline overfit testing only.",
        "seed": seed,
        "size": smoke_size,
        "covered_classes": [
            ULTRALYTICS_DOTA_V1_CLASSES[class_id]
            for class_id in sorted(smoke_class_presence)
        ],
        "class_image_presence": {
            ULTRALYTICS_DOTA_V1_CLASSES[class_id]: smoke_class_presence[class_id]
            for class_id in range(len(ULTRALYTICS_DOTA_V1_CLASSES))
        },
        "class_object_counts": {
            ULTRALYTICS_DOTA_V1_CLASSES[class_id]: smoke_class_objects[class_id]
            for class_id in range(len(ULTRALYTICS_DOTA_V1_CLASSES))
        },
        "records": [asdict(record) for record in smoke_records],
    }
    (output_root / f"overfit{smoke_size}_manifest.json").write_text(
        json.dumps(smoke_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "tiles_root": str(tiles_root.resolve()),
        "output_root": str(output_root.resolve()),
        "class_order": list(ULTRALYTICS_DOTA_V1_CLASSES),
        "splits": split_summaries,
        "smoke": {
            "size": smoke_size,
            "seed": seed,
            "covered_class_count": len(smoke_class_presence),
            "covered_classes": smoke_manifest["covered_classes"],
        },
    }
    (output_root / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
