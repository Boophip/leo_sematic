"""Reporting helpers for formal Ultralytics YOLO-OBB experiments."""

from __future__ import annotations

import csv
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping


def read_training_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            key.strip(): float(value)
            for key, value in row.items()
            if value not in (None, "")
        }
        for row in rows
    ]


def best_training_row(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return max(rows, key=lambda row: row.get("metrics/mAP50-95(B)", -math.inf))


def metrics_summary(metrics: Any) -> dict[str, object]:
    """Serialize aggregate, per-class, speed, and model metrics."""
    names: Mapping[int, str] = metrics.names
    per_class: dict[str, dict[str, float]] = {}
    box = metrics.box
    for class_index, class_name in names.items():
        per_class[class_name] = {
            "precision": float(box.p[class_index]),
            "recall": float(box.r[class_index]),
            "mAP50": float(box.ap50[class_index]),
            "mAP50-95": float(box.maps[class_index]),
        }
    return {
        "aggregate": {
            key: float(value)
            for key, value in metrics.results_dict.items()
            if isinstance(value, (int, float))
        },
        "per_class": per_class,
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }


def environment_summary(ultralytics: Any, torch: Any) -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
