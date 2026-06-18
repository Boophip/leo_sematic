"""Tiny smoke-training loop for the multi-exit task model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.detector.dota_original_eval import ULTRALYTICS_DOTA_V1_CLASSES
from src.multi_exit.dataset import RoiCropDataset
from src.multi_exit.model import TinyMultiExitCNN, multi_exit_cross_entropy


def set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_multi_exit_smoke(
    metadata_path: Path,
    output_dir: Path,
    *,
    limit_rois: int = 128,
    epochs: int = 1,
    batch_size: int = 16,
    input_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, object]:
    """Run a tiny overfit/smoke training pass; not a formal experiment."""

    if limit_rois <= 0:
        raise ValueError("limit_rois must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    set_training_seed(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = RoiCropDataset(
        metadata_path,
        input_size=input_size,
        true_positive_only=True,
        limit_records=limit_rois,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TinyMultiExitCNN(num_classes=len(ULTRALYTICS_DOTA_V1_CLASSES)).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, targets in loader:
            images = images.to(device_obj)
            targets = targets.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = multi_exit_cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            total_samples += images.size(0)
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": total_loss / max(total_samples, 1),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "tiny_multi_exit_smoke.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": len(ULTRALYTICS_DOTA_V1_CLASSES),
            "class_names": list(ULTRALYTICS_DOTA_V1_CLASSES),
            "input_size": input_size,
            "seed": seed,
        },
        checkpoint_path,
    )

    summary = {
        "purpose": "Tiny multi-exit smoke training; not a formal experiment.",
        "metadata_path": str(metadata_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "configuration": {
            "limit_rois": limit_rois,
            "epochs": epochs,
            "batch_size": batch_size,
            "input_size": input_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": str(device_obj),
        },
        "counts": {
            "training_roi_count": len(dataset),
            "class_count": len(ULTRALYTICS_DOTA_V1_CLASSES),
        },
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


@torch.no_grad()
def evaluate_multi_exit_model(
    model: TinyMultiExitCNN,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    exit_count = len(model.exit_heads)
    total_samples = 0
    loss_sums = [0.0] * exit_count
    correct_counts = [0] * exit_count
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits_by_exit = model(images)
        total_samples += images.size(0)
        for exit_index, logits in enumerate(logits_by_exit):
            loss = torch.nn.functional.cross_entropy(logits, targets)
            predictions = torch.argmax(logits, dim=1)
            loss_sums[exit_index] += float(loss.item()) * images.size(0)
            correct_counts[exit_index] += int((predictions == targets).sum().item())
    per_exit = []
    for exit_index in range(exit_count):
        per_exit.append(
            {
                "exit_level": exit_index + 1,
                "loss": loss_sums[exit_index] / max(total_samples, 1),
                "accuracy": correct_counts[exit_index] / max(total_samples, 1),
            }
        )
    return {
        "samples": total_samples,
        "per_exit": per_exit,
        "mean_accuracy": sum(item["accuracy"] for item in per_exit) / max(exit_count, 1),
    }


def train_multi_exit_model(
    train_metadata_path: Path,
    val_metadata_path: Path,
    output_dir: Path,
    *,
    limit_train_rois: int | None = None,
    limit_val_rois: int | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    input_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
    hidden_exit_weights: Sequence[float] | None = None,
) -> dict[str, object]:
    """Train the formal multi-exit baseline on split-isolated GT ROI crops."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    set_training_seed(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_dataset = RoiCropDataset(
        train_metadata_path,
        input_size=input_size,
        true_positive_only=True,
        limit_records=limit_train_rois,
    )
    val_dataset = RoiCropDataset(
        val_metadata_path,
        input_size=input_size,
        true_positive_only=True,
        limit_records=limit_val_rois,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = TinyMultiExitCNN(num_classes=len(ULTRALYTICS_DOTA_V1_CLASSES)).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_mean_accuracy = -1.0
    best_checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, object]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for images, targets in train_loader:
            images = images.to(device_obj)
            targets = targets.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = multi_exit_cross_entropy(
                logits,
                targets,
                exit_weights=hidden_exit_weights,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            total_samples += images.size(0)

        val_metrics = evaluate_multi_exit_model(model, val_loader, device=device_obj)
        mean_accuracy = float(val_metrics["mean_accuracy"])
        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(ULTRALYTICS_DOTA_V1_CLASSES),
                    "class_names": list(ULTRALYTICS_DOTA_V1_CLASSES),
                    "input_size": input_size,
                    "seed": seed,
                    "epoch": epoch + 1,
                },
                best_checkpoint_path,
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(total_samples, 1),
                "val": val_metrics,
            }
        )

    last_checkpoint_path = output_dir / "last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": len(ULTRALYTICS_DOTA_V1_CLASSES),
            "class_names": list(ULTRALYTICS_DOTA_V1_CLASSES),
            "input_size": input_size,
            "seed": seed,
            "epoch": epochs,
        },
        last_checkpoint_path,
    )

    summary = {
        "purpose": "Formal multi-exit baseline training on split-isolated GT ROI crops.",
        "note": "GT crops train the task model; detector ROI metadata remains the scheduling input.",
        "inputs": {
            "train_metadata_path": str(train_metadata_path.resolve()),
            "val_metadata_path": str(val_metadata_path.resolve()),
        },
        "outputs": {
            "output_dir": str(output_dir.resolve()),
            "best_checkpoint": str(best_checkpoint_path.resolve()),
            "last_checkpoint": str(last_checkpoint_path.resolve()),
        },
        "configuration": {
            "limit_train_rois": limit_train_rois,
            "limit_val_rois": limit_val_rois,
            "epochs": epochs,
            "batch_size": batch_size,
            "input_size": input_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": str(device_obj),
            "exit_weights": list(hidden_exit_weights) if hidden_exit_weights else None,
        },
        "counts": {
            "train_roi_count": len(train_dataset),
            "val_roi_count": len(val_dataset),
            "class_count": len(ULTRALYTICS_DOTA_V1_CLASSES),
        },
        "best": {
            "mean_val_accuracy": best_mean_accuracy,
        },
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
