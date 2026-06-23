"""Build ROI x compression x exit profiling rows."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from PIL import Image

from src.proxy.image_features import IMAGE_FEATURE_COLUMNS, extract_image_quality_features
from src.profiling.compression import DEFAULT_BETA_LEVELS, compress_image


@dataclass(frozen=True)
class ExitProfile:
    """Per-exit model result measured after a selected compression transform."""

    exit_level: int
    inference_ms: float
    task_quality: float
    predicted_class_id: int
    exit_confidence: float


@dataclass(frozen=True)
class RoiProfileRow:
    """Single ROI x exit x compression row consumed by proxy and simulation."""

    roi_id: str
    image_id: str
    predicted_class: str
    class_id: int
    detector_confidence: float
    area_ratio: float
    semantic_value: float
    grid_id: int
    class_priority: float
    quality_label: str
    exit_level: int
    compression_level: str
    compressed_bytes: int
    encode_ms: float
    decode_ms: float
    inference_ms: float
    task_quality: float
    encoded_format: str
    output_width: int
    output_height: int
    crop_width: float
    crop_height: float
    crop_aspect_ratio: float
    brightness_mean: float
    brightness_std: float
    rgb_mean_r: float
    rgb_mean_g: float
    rgb_mean_b: float
    rgb_std_r: float
    rgb_std_g: float
    rgb_std_b: float
    laplacian_var: float
    edge_density: float
    entropy: float
    predicted_class_id: int
    exit_confidence: float


PROFILE_FIELDNAMES = tuple(RoiProfileRow.__dataclass_fields__.keys())
IMAGE_PROFILE_FIELDNAMES = IMAGE_FEATURE_COLUMNS


Predictor = Callable[[Image.Image, Mapping[str, object]], Sequence[ExitProfile]]


def load_roi_metadata(path: Path, *, limit_rois: int | None = None) -> list[dict[str, object]]:
    """Load ROI records while preserving the upstream metadata schema."""

    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("crop_path"):
            raise ValueError(f"{path}:{line_number}: missing crop_path")
        records.append(record)
        if limit_rois is not None and len(records) >= limit_rois:
            break
    return records


def profile_roi_records(
    records: Iterable[Mapping[str, object]],
    predictor: Predictor,
    *,
    compression_levels: Sequence[str | int] = DEFAULT_BETA_LEVELS,
) -> list[RoiProfileRow]:
    """Generate deterministic profile rows while preserving the ROI schema."""

    rows: list[RoiProfileRow] = []
    for record in records:
        crop_path = Path(str(record["crop_path"]))
        with Image.open(crop_path) as image:
            source = image.convert("RGB")
        for level in compression_levels:
            # Compression is performed before model inference so each row
            # captures the action-dependent byte size, latency, and quality.
            compressed = compress_image(source, level)
            image_features = extract_image_quality_features(compressed.decoded_image)
            exit_profiles = predictor(compressed.decoded_image, record)
            for exit_profile in exit_profiles:
                rows.append(
                    RoiProfileRow(
                        roi_id=str(record.get("roi_id", "")),
                        image_id=str(record.get("image_id", "")),
                        predicted_class=str(record.get("predicted_class", "")),
                        class_id=int(record.get("class_id", -1)),
                        detector_confidence=float(record.get("confidence", 0.0)),
                        area_ratio=float(record.get("area_ratio", 0.0)),
                        semantic_value=float(record.get("semantic_value", 0.0)),
                        grid_id=int(record.get("grid_id", -1)),
                        class_priority=float(record.get("class_priority", 0.0)),
                        quality_label=str(record.get("quality_label", "")),
                        exit_level=exit_profile.exit_level,
                        compression_level=compressed.compression_level,
                        compressed_bytes=compressed.compressed_bytes,
                        encode_ms=compressed.encode_ms,
                        decode_ms=compressed.decode_ms,
                        inference_ms=exit_profile.inference_ms,
                        task_quality=exit_profile.task_quality,
                        encoded_format=compressed.encoded_format,
                        output_width=compressed.output_width,
                        output_height=compressed.output_height,
                        crop_width=image_features["crop_width"],
                        crop_height=image_features["crop_height"],
                        crop_aspect_ratio=image_features["crop_aspect_ratio"],
                        brightness_mean=image_features["brightness_mean"],
                        brightness_std=image_features["brightness_std"],
                        rgb_mean_r=image_features["rgb_mean_r"],
                        rgb_mean_g=image_features["rgb_mean_g"],
                        rgb_mean_b=image_features["rgb_mean_b"],
                        rgb_std_r=image_features["rgb_std_r"],
                        rgb_std_g=image_features["rgb_std_g"],
                        rgb_std_b=image_features["rgb_std_b"],
                        laplacian_var=image_features["laplacian_var"],
                        edge_density=image_features["edge_density"],
                        entropy=image_features["entropy"],
                        predicted_class_id=exit_profile.predicted_class_id,
                        exit_confidence=exit_profile.exit_confidence,
                    )
                )
    return rows


def write_profile_csv(path: Path, rows: Iterable[RoiProfileRow]) -> None:
    """Write rows with a stable column order for downstream scripts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
