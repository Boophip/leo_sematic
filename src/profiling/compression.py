"""Reproducible ROI compression policies for offload profiling."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image


CompressionKey = Literal["local", 0, 1, 2, 3]


@dataclass(frozen=True)
class CompressionSpec:
    """Concrete engineering setting for one compression action."""

    key: str
    encoded_format: str
    quality: int | None
    scale: float
    min_short_side: int | None
    communication_payload: bool


@dataclass(frozen=True)
class CompressionResult:
    compression_level: str
    encoded_format: str
    compressed_bytes: int
    encode_ms: float
    decode_ms: float
    output_width: int
    output_height: int
    decoded_image: Image.Image


LOCAL_SPEC = CompressionSpec(
    key="local",
    encoded_format="none",
    quality=None,
    scale=1.0,
    min_short_side=None,
    communication_payload=False,
)

COMPRESSION_SPECS: dict[int, CompressionSpec] = {
    0: CompressionSpec(
        key="beta_0",
        encoded_format="PNG",
        quality=None,
        scale=1.0,
        min_short_side=None,
        communication_payload=True,
    ),
    1: CompressionSpec(
        key="beta_1",
        encoded_format="JPEG",
        quality=90,
        scale=1.0,
        min_short_side=None,
        communication_payload=True,
    ),
    2: CompressionSpec(
        key="beta_2",
        encoded_format="JPEG",
        quality=70,
        scale=0.75,
        min_short_side=16,
        communication_payload=True,
    ),
    3: CompressionSpec(
        key="beta_3",
        encoded_format="JPEG",
        quality=45,
        scale=0.50,
        min_short_side=8,
        communication_payload=True,
    ),
}

DEFAULT_BETA_LEVELS: tuple[int, ...] = (0, 1, 2, 3)


def resolve_compression_spec(level: str | int) -> CompressionSpec:
    """Return the concrete compression spec for a user-facing level."""

    if isinstance(level, str):
        normalized = level.strip().lower().replace("-", "_")
        if normalized == "local":
            return LOCAL_SPEC
        if normalized.startswith("beta_"):
            normalized = normalized.removeprefix("beta_")
        elif normalized.startswith("beta"):
            normalized = normalized.removeprefix("beta")
        try:
            level = int(normalized)
        except ValueError as exc:
            raise ValueError(f"unknown compression level: {level!r}") from exc

    if level in COMPRESSION_SPECS:
        return COMPRESSION_SPECS[int(level)]
    raise ValueError(f"unknown compression level: {level!r}")


def _resampling_filter() -> int:
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def _resize_for_spec(image: Image.Image, spec: CompressionSpec) -> Image.Image:
    rgb = image.convert("RGB")
    if spec.scale == 1.0 and spec.min_short_side is None:
        return rgb.copy()

    width, height = rgb.size
    target_width = max(int(round(width * spec.scale)), 1)
    target_height = max(int(round(height * spec.scale)), 1)
    if spec.min_short_side is not None:
        short_side = min(target_width, target_height)
        if short_side < spec.min_short_side:
            factor = spec.min_short_side / max(short_side, 1)
            target_width = max(int(round(target_width * factor)), 1)
            target_height = max(int(round(target_height * factor)), 1)
    return rgb.resize((target_width, target_height), _resampling_filter())


def compress_image(image: Image.Image, level: str | int) -> CompressionResult:
    """Compress one ROI crop and return the decoded image for model inference."""

    spec = resolve_compression_spec(level)
    if not spec.communication_payload:
        decoded = image.convert("RGB").copy()
        return CompressionResult(
            compression_level=spec.key,
            encoded_format=spec.encoded_format,
            compressed_bytes=0,
            encode_ms=0.0,
            decode_ms=0.0,
            output_width=decoded.width,
            output_height=decoded.height,
            decoded_image=decoded,
        )

    prepared = _resize_for_spec(image, spec)
    buffer = io.BytesIO()
    encode_start = time.perf_counter()
    save_kwargs: dict[str, object] = {}
    if spec.encoded_format == "JPEG":
        save_kwargs.update({"quality": spec.quality, "optimize": True})
    elif spec.encoded_format == "PNG":
        save_kwargs.update({"optimize": True})
    prepared.save(buffer, format=spec.encoded_format, **save_kwargs)
    encode_ms = (time.perf_counter() - encode_start) * 1000.0

    payload = buffer.getvalue()
    decode_start = time.perf_counter()
    with Image.open(io.BytesIO(payload)) as decoded:
        decoded_rgb = decoded.convert("RGB")
        decoded_rgb.load()
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    return CompressionResult(
        compression_level=spec.key,
        encoded_format=spec.encoded_format,
        compressed_bytes=len(payload),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        output_width=decoded_rgb.width,
        output_height=decoded_rgb.height,
        decoded_image=decoded_rgb,
    )


def compress_image_file(path: Path, level: str | int) -> CompressionResult:
    with Image.open(path) as image:
        return compress_image(image, level)
