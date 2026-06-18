"""Reusable DOTA original-image-level lite split generation."""

from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


DOTA_V1_CLASSES = (
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
)


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    label_path: Path
    image_path: Path | None
    class_counts: Counter[str]

    @property
    def classes(self) -> set[str]:
        return set(self.class_counts)


@dataclass(frozen=True)
class DiscoveryReport:
    label_count: int
    image_count: int
    matched_count: int
    usable_record_count: int
    missing_image_ids: tuple[str, ...]
    image_without_label_ids: tuple[str, ...]
    empty_label_ids: tuple[str, ...]


def parse_dota_label(path: Path) -> Counter[str]:
    """Parse DOTA quadrilateral annotations, ignoring the two metadata lines."""
    counts: Counter[str] = Counter()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        fields = line.split()
        if not fields or fields[0].startswith(("imagesource:", "gsd:")):
            continue
        if len(fields) < 10:
            raise ValueError(f"{path}:{line_number}: expected at least 10 fields")
        class_name = fields[8]
        if class_name not in DOTA_V1_CLASSES:
            raise ValueError(f"{path}:{line_number}: unknown DOTA-v1.0 class {class_name!r}")
        counts[class_name] += 1
    return counts


def discover_records(
    dataset_root: Path,
    *,
    labels_only: bool = False,
    include_empty: bool = False,
    label_dir: Path | None = None,
    image_dir: Path | None = None,
) -> tuple[list[ImageRecord], DiscoveryReport]:
    """Discover DOTA records and report image/label mismatches."""
    label_dir = label_dir or dataset_root / "labelTxt-v1.0" / "labelTxt"
    image_dir = image_dir or dataset_root / "images"
    label_paths = sorted(label_dir.glob("*.txt"))
    image_paths = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )

    images_by_id: dict[str, Path] = {}
    for path in image_paths:
        if path.stem in images_by_id:
            raise ValueError(f"duplicate image ID {path.stem!r}: {path}")
        images_by_id[path.stem] = path

    labels_by_id = {path.stem: path for path in label_paths}
    missing_image_ids = tuple(sorted(set(labels_by_id) - set(images_by_id)))
    image_without_label_ids = tuple(sorted(set(images_by_id) - set(labels_by_id)))
    selected_label_ids = sorted(labels_by_id if labels_only else set(labels_by_id) & set(images_by_id))

    records: list[ImageRecord] = []
    empty_label_ids: list[str] = []
    for image_id in selected_label_ids:
        class_counts = parse_dota_label(labels_by_id[image_id])
        if not class_counts:
            empty_label_ids.append(image_id)
            if not include_empty:
                continue
        records.append(
            ImageRecord(
                image_id=image_id,
                label_path=labels_by_id[image_id],
                image_path=images_by_id.get(image_id),
                class_counts=class_counts,
            )
        )
    report = DiscoveryReport(
        label_count=len(label_paths),
        image_count=len(image_paths),
        matched_count=len(set(labels_by_id) & set(images_by_id)),
        usable_record_count=len(records),
        missing_image_ids=missing_image_ids,
        image_without_label_ids=image_without_label_ids,
        empty_label_ids=tuple(empty_label_ids),
    )
    return records, report


def _totals(records: Iterable[ImageRecord]) -> tuple[Counter[str], Counter[str]]:
    presence: Counter[str] = Counter()
    objects: Counter[str] = Counter()
    for record in records:
        presence.update(record.classes)
        objects.update(record.class_counts)
    return presence, objects


def _minimum_targets(
    total_presence: Mapping[str, int],
    split_sizes: Mapping[str, int],
    active_splits: set[str],
    min_class_images: int,
) -> dict[str, dict[str, int]]:
    targets: dict[str, dict[str, int]] = {}
    for split_name, split_size in split_sizes.items():
        targets[split_name] = {}
        for class_name in DOTA_V1_CLASSES:
            if split_name not in active_splits or total_presence[class_name] == 0:
                targets[split_name][class_name] = 0
                continue
            targets[split_name][class_name] = min(min_class_images, split_size)

    for class_name in DOTA_V1_CLASSES:
        required = sum(targets[split][class_name] for split in active_splits)
        if required > total_presence[class_name]:
            raise ValueError(
                f"class {class_name!r} occurs in {total_presence[class_name]} images, "
                f"but minimum coverage requires {required}"
            )
    return targets


def stratified_split(
    records: list[ImageRecord],
    requested_sizes: Mapping[str, int],
    *,
    seed: int = 42,
    min_class_images: int = 5,
    refinement_steps: int = 30_000,
) -> dict[str, list[ImageRecord]]:
    """Create exact-size multi-label splits while preserving class distributions."""
    if not records:
        raise ValueError("no records were discovered")
    if any(size <= 0 for size in requested_sizes.values()):
        raise ValueError("all requested split sizes must be positive")
    if min_class_images < 0:
        raise ValueError("min_class_images must be non-negative")
    if refinement_steps < 0:
        raise ValueError("refinement_steps must be non-negative")
    selected_total = sum(requested_sizes.values())
    if selected_total > len(records):
        raise ValueError(
            f"requested {selected_total} images, but only {len(records)} usable records exist"
        )

    rng = random.Random(seed)
    split_sizes = dict(requested_sizes)
    unused_size = len(records) - selected_total
    if unused_size:
        split_sizes["_unused"] = unused_size
    split_names = list(split_sizes)
    active_splits = set(requested_sizes)

    total_presence, total_objects = _totals(records)
    min_targets = _minimum_targets(
        total_presence,
        split_sizes,
        active_splits,
        min_class_images,
    )
    presence_targets = {
        split: {
            class_name: total_presence[class_name] * size / len(records)
            for class_name in DOTA_V1_CLASSES
        }
        for split, size in split_sizes.items()
    }
    object_targets = {
        split: {
            class_name: total_objects[class_name] * size / len(records)
            for class_name in DOTA_V1_CLASSES
        }
        for split, size in split_sizes.items()
    }

    assignments: dict[str, list[ImageRecord]] = {split: [] for split in split_names}
    presence_stats = {split: Counter() for split in split_names}
    object_stats = {split: Counter() for split in split_names}

    def score() -> float:
        value = 0.0
        for split in split_names:
            for class_name in DOTA_V1_CLASSES:
                presence_target = presence_targets[split][class_name]
                object_target = object_targets[split][class_name]
                value += 3.0 * (
                    (presence_stats[split][class_name] - presence_target)
                    / max(presence_target, 1.0)
                ) ** 2
                value += 0.35 * (
                    (object_stats[split][class_name] - object_target)
                    / max(object_target, 1.0)
                ) ** 2
                deficit = max(
                    min_targets[split][class_name] - presence_stats[split][class_name],
                    0,
                )
                value += 250.0 * deficit * deficit
        return value

    def update_stats(split: str, record: ImageRecord, direction: int) -> None:
        for class_name in record.classes:
            presence_stats[split][class_name] += direction
        for class_name, count in record.class_counts.items():
            object_stats[split][class_name] += direction * count

    rarity = {
        class_name: 1.0 / max(total_presence[class_name], 1)
        for class_name in DOTA_V1_CLASSES
    }
    shuffled = list(records)
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda record: (
            sum(rarity[class_name] for class_name in record.classes),
            len(record.classes),
        ),
        reverse=True,
    )

    current_score = score()
    for record in shuffled:
        best_splits: list[str] = []
        best_score = math.inf
        for split in split_names:
            if len(assignments[split]) >= split_sizes[split]:
                continue
            assignments[split].append(record)
            update_stats(split, record, 1)
            candidate_score = score()
            update_stats(split, record, -1)
            assignments[split].pop()
            if candidate_score < best_score - 1e-12:
                best_score = candidate_score
                best_splits = [split]
            elif abs(candidate_score - best_score) <= 1e-12:
                best_splits.append(split)
        chosen = rng.choice(best_splits)
        assignments[chosen].append(record)
        update_stats(chosen, record, 1)
        current_score = best_score

    for _ in range(refinement_steps if len(split_names) > 1 else 0):
        split_a, split_b = rng.sample(split_names, 2)
        index_a = rng.randrange(len(assignments[split_a]))
        index_b = rng.randrange(len(assignments[split_b]))
        record_a = assignments[split_a][index_a]
        record_b = assignments[split_b][index_b]
        if record_a is record_b:
            continue

        update_stats(split_a, record_a, -1)
        update_stats(split_b, record_b, -1)
        update_stats(split_a, record_b, 1)
        update_stats(split_b, record_a, 1)
        candidate_score = score()
        if candidate_score <= current_score:
            assignments[split_a][index_a] = record_b
            assignments[split_b][index_b] = record_a
            current_score = candidate_score
        else:
            update_stats(split_a, record_b, -1)
            update_stats(split_b, record_a, -1)
            update_stats(split_a, record_a, 1)
            update_stats(split_b, record_b, 1)

    for split in active_splits:
        for class_name, minimum in min_targets[split].items():
            actual = presence_stats[split][class_name]
            if actual < minimum:
                raise RuntimeError(
                    f"could not satisfy {split}/{class_name} coverage: {actual} < {minimum}"
                )

    return {
        split: sorted(assignments[split], key=lambda record: record.image_id)
        for split in requested_sizes
    }


def build_summary(splits: Mapping[str, list[ImageRecord]]) -> dict[str, object]:
    summary: dict[str, object] = {"splits": {}}
    all_ids: list[str] = []
    for split_name, records in splits.items():
        presence, objects = _totals(records)
        all_ids.extend(record.image_id for record in records)
        summary["splits"][split_name] = {
            "images": len(records),
            "objects": sum(objects.values()),
            "class_image_presence": {
                class_name: presence[class_name] for class_name in DOTA_V1_CLASSES
            },
            "class_object_counts": {
                class_name: objects[class_name] for class_name in DOTA_V1_CLASSES
            },
        }
    summary["unique_selected_images"] = len(set(all_ids))
    summary["selected_images"] = len(all_ids)
    return summary


def write_outputs(
    splits: Mapping[str, list[ImageRecord]],
    output_dir: Path,
    *,
    materialize: bool = False,
    metadata: Mapping[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        manifest_path = output_dir / f"{split_name}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "image_path", "label_path", "classes", "objects"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "image_id": record.image_id,
                        "image_path": str(record.image_path or ""),
                        "label_path": str(record.label_path),
                        "classes": " ".join(sorted(record.classes)),
                        "objects": sum(record.class_counts.values()),
                    }
                )

        if materialize:
            split_root = output_dir / "materialized" / split_name
            image_output = split_root / "images"
            label_output = split_root / "labelTxt"
            image_output.mkdir(parents=True, exist_ok=True)
            label_output.mkdir(parents=True, exist_ok=True)
            for record in records:
                if record.image_path is None:
                    raise ValueError(f"cannot materialize missing image {record.image_id}")
                shutil.copy2(record.image_path, image_output / record.image_path.name)
                shutil.copy2(record.label_path, label_output / record.label_path.name)

    summary = build_summary(splits)
    if metadata:
        summary["metadata"] = dict(metadata)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
