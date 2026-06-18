"""Inference helpers that expose per-exit confidence and quality values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from PIL import Image

from src.multi_exit.dataset import image_to_tensor
from src.multi_exit.model import TinyMultiExitCNN


@dataclass(frozen=True)
class MultiExitPrediction:
    exit_level: int
    inference_ms: float
    task_quality: float
    predicted_class_id: int
    exit_confidence: float


def true_class_id_from_record(record: Mapping[str, object]) -> int | None:
    """Use true-positive ROI labels as the first profiling supervision signal."""

    if record.get("quality_label") != "true_positive":
        return None
    class_id = int(record.get("class_id", -1))
    return class_id if class_id >= 0 else None


@torch.no_grad()
def predict_image_exits(
    model: TinyMultiExitCNN,
    image: Image.Image,
    *,
    input_size: int = 64,
    device: str | torch.device = "cpu",
    true_class_id: int | None = None,
) -> list[MultiExitPrediction]:
    model.eval()
    tensor = image_to_tensor(image, input_size=input_size).unsqueeze(0).to(device)
    timed_outputs = model.forward_with_exit_times(tensor)

    predictions: list[MultiExitPrediction] = []
    for output in timed_outputs:
        probabilities = F.softmax(output.logits, dim=1)[0]
        confidence, predicted_class_id = torch.max(probabilities, dim=0)
        if true_class_id is None or true_class_id >= probabilities.numel():
            task_quality = 0.0
        else:
            task_quality = float(probabilities[true_class_id].item())
        predictions.append(
            MultiExitPrediction(
                exit_level=output.exit_level,
                inference_ms=output.cumulative_ms,
                task_quality=task_quality,
                predicted_class_id=int(predicted_class_id.item()),
                exit_confidence=float(confidence.item()),
            )
        )
    return predictions
