"""ROI crop dataset for tiny multi-exit smoke training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_labeled_roi_records(
    metadata_path: Path,
    *,
    true_positive_only: bool = True,
    limit_records: int | None = None,
) -> list[dict[str, Any]]:
    """Load existing ROI metadata without changing its schema."""

    records: list[dict[str, Any]] = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if true_positive_only and record.get("quality_label") != "true_positive":
            continue
        if not record.get("crop_path"):
            continue
        if int(record.get("class_id", -1)) < 0:
            continue
        records.append(record)
        if limit_records is not None and len(records) >= limit_records:
            break
    return records


def image_to_tensor(image: Image.Image, *, input_size: int = 64) -> torch.Tensor:
    """Convert a ROI crop into the fixed model input tensor."""

    if input_size <= 0:
        raise ValueError("input_size must be positive")
    resized = image.convert("RGB").resize((input_size, input_size))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class RoiCropDataset(Dataset):
    """Dataset over true-positive ROI crops for the first multi-exit baseline."""

    def __init__(
        self,
        metadata_path: Path,
        *,
        input_size: int = 64,
        true_positive_only: bool = True,
        limit_records: int | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.input_size = input_size
        self.records = load_labeled_roi_records(
            metadata_path,
            true_positive_only=true_positive_only,
            limit_records=limit_records,
        )
        if not self.records:
            raise ValueError(f"no usable ROI crop records found in {metadata_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        crop_path = Path(str(record["crop_path"]))
        with Image.open(crop_path) as image:
            tensor = image_to_tensor(image, input_size=self.input_size)
        label = torch.tensor(int(record["class_id"]), dtype=torch.long)
        return tensor, label
