"""Online-safe ROI image statistics for task-quality proxy features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_FEATURE_COLUMNS: tuple[str, ...] = (
    "crop_width",
    "crop_height",
    "crop_aspect_ratio",
    "brightness_mean",
    "brightness_std",
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "laplacian_var",
    "edge_density",
    "entropy",
)


def extract_image_quality_features(image: Image.Image) -> dict[str, float]:
    """Return cheap, non-label ROI quality features available before scheduling."""

    rgb = image.convert("RGB")
    width, height = rgb.size
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    )

    gradient_x = np.diff(gray, axis=1)
    gradient_y = np.diff(gray, axis=0)
    gradient_magnitude = np.zeros_like(gray)
    if gradient_x.size:
        gradient_magnitude[:, 1:] += np.abs(gradient_x)
    if gradient_y.size:
        gradient_magnitude[1:, :] += np.abs(gradient_y)

    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * gray
    )

    histogram, _ = np.histogram(gray, bins=256, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64) / max(float(histogram.sum()), 1.0)
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))

    rgb_mean = array.mean(axis=(0, 1))
    rgb_std = array.std(axis=(0, 1))
    edge_density = float(np.mean(gradient_magnitude > 0.10))

    return {
        "crop_width": float(width),
        "crop_height": float(height),
        "crop_aspect_ratio": float(width / max(height, 1)),
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "rgb_mean_r": float(rgb_mean[0]),
        "rgb_mean_g": float(rgb_mean[1]),
        "rgb_mean_b": float(rgb_mean[2]),
        "rgb_std_r": float(rgb_std[0]),
        "rgb_std_g": float(rgb_std[1]),
        "rgb_std_b": float(rgb_std[2]),
        "laplacian_var": float(np.var(laplacian)),
        "edge_density": edge_density,
        "entropy": entropy,
    }


def extract_image_quality_features_from_path(path: Path) -> dict[str, float]:
    """Load a ROI crop and compute online-safe image quality features."""

    with Image.open(path) as image:
        return extract_image_quality_features(image)
