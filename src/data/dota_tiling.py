"""Sliding-window tiling for DOTA images with original-coordinate mappings."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


Point = tuple[float, float]


@dataclass(frozen=True)
class DotaObject:
    polygon: tuple[Point, Point, Point, Point]
    class_name: str
    difficult: int


@dataclass(frozen=True)
class TileConfig:
    tile_size: int = 1024
    overlap: int = 200
    min_visible_ratio: float = 0.7
    padding_value: int = 0

    @property
    def stride(self) -> int:
        return self.tile_size - self.overlap

    def validate(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if not 0 <= self.overlap < self.tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
        if not 0 <= self.min_visible_ratio <= 1:
            raise ValueError("min_visible_ratio must be in [0, 1]")
        if not 0 <= self.padding_value <= 255:
            raise ValueError("padding_value must be in [0, 255]")


@dataclass(frozen=True)
class TileMapping:
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
    object_count: int
    partial_object_count: int


def sliding_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return starts that cover an axis and align the final window to its end."""
    if length <= 0:
        raise ValueError("axis length must be positive")
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("invalid tile_size or overlap")
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if positions[-1] != final_start:
        positions.append(final_start)
    return positions


def polygon_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def _clip_edge(
    points: Sequence[Point],
    inside,
    intersection,
) -> list[Point]:
    if not points:
        return []
    output: list[Point] = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside
    return output


def clip_polygon_to_rect(
    polygon: Sequence[Point],
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> list[Point]:
    """Clip a polygon to an axis-aligned rectangle."""

    def vertical_intersection(a: Point, b: Point, x: float) -> Point:
        ratio = 0.0 if b[0] == a[0] else (x - a[0]) / (b[0] - a[0])
        return x, a[1] + ratio * (b[1] - a[1])

    def horizontal_intersection(a: Point, b: Point, y: float) -> Point:
        ratio = 0.0 if b[1] == a[1] else (y - a[1]) / (b[1] - a[1])
        return a[0] + ratio * (b[0] - a[0]), y

    result = list(polygon)
    result = _clip_edge(result, lambda p: p[0] >= left, lambda a, b: vertical_intersection(a, b, left))
    result = _clip_edge(result, lambda p: p[0] <= right, lambda a, b: vertical_intersection(a, b, right))
    result = _clip_edge(result, lambda p: p[1] >= top, lambda a, b: horizontal_intersection(a, b, top))
    result = _clip_edge(result, lambda p: p[1] <= bottom, lambda a, b: horizontal_intersection(a, b, bottom))
    return result


def minimum_area_quadrilateral(points: Sequence[Point]) -> tuple[Point, Point, Point, Point]:
    """Approximate a clipped convex polygon with its minimum-area rectangle."""
    if len(points) < 3 or polygon_area(points) <= 0:
        raise ValueError("cannot fit a quadrilateral to a degenerate polygon")

    best: tuple[float, tuple[Point, Point, Point, Point]] | None = None
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        angle = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated = [
            (x * cosine + y * sine, -x * sine + y * cosine)
            for x, y in points
        ]
        min_x = min(x for x, _ in rotated)
        max_x = max(x for x, _ in rotated)
        min_y = min(y for _, y in rotated)
        max_y = max(y for _, y in rotated)
        area = (max_x - min_x) * (max_y - min_y)
        rotated_corners = (
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        )
        corners = tuple(
            (
                x * cosine - y * sine,
                x * sine + y * cosine,
            )
            for x, y in rotated_corners
        )
        if best is None or area < best[0]:
            best = area, corners  # type: ignore[assignment]
    assert best is not None
    return best[1]


def parse_dota_annotations(path: Path) -> tuple[list[str], list[DotaObject]]:
    metadata: list[str] = []
    objects: list[DotaObject] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        fields = line.split()
        if not fields:
            continue
        if fields[0].startswith(("imagesource:", "gsd:")):
            metadata.append(line)
            continue
        if len(fields) < 10:
            raise ValueError(f"{path}:{line_number}: expected at least 10 fields")
        coordinates = tuple(float(value) for value in fields[:8])
        polygon = tuple(
            (coordinates[index], coordinates[index + 1])
            for index in range(0, 8, 2)
        )
        objects.append(
            DotaObject(
                polygon=polygon,  # type: ignore[arg-type]
                class_name=fields[8],
                difficult=int(fields[9]),
            )
        )
    return metadata, objects


def _format_coordinate(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(rounded)
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_dota_object(obj: DotaObject) -> str:
    coordinates = " ".join(
        _format_coordinate(value)
        for point in obj.polygon
        for value in point
    )
    return f"{coordinates} {obj.class_name} {obj.difficult}"


def tile_annotations(
    objects: Iterable[DotaObject],
    *,
    offset_x: int,
    offset_y: int,
    crop_width: int,
    crop_height: int,
    min_visible_ratio: float,
) -> tuple[list[DotaObject], int]:
    """Translate annotations into a tile and mark clipped objects difficult=2."""
    tiled, partial_count, _, _ = _tile_annotations_with_indices(
        list(objects),
        offset_x=offset_x,
        offset_y=offset_y,
        crop_width=crop_width,
        crop_height=crop_height,
        min_visible_ratio=min_visible_ratio,
    )
    return tiled, partial_count


def _tile_annotations_with_indices(
    objects: Sequence[DotaObject],
    *,
    offset_x: int,
    offset_y: int,
    crop_width: int,
    crop_height: int,
    min_visible_ratio: float,
) -> tuple[list[DotaObject], int, set[int], set[int]]:
    tiled: list[DotaObject] = []
    partial_count = 0
    included_indices: set[int] = set()
    fully_included_indices: set[int] = set()
    right = offset_x + crop_width
    bottom = offset_y + crop_height
    for object_index, obj in enumerate(objects):
        original_area = polygon_area(obj.polygon)
        if original_area <= 0:
            continue
        clipped = clip_polygon_to_rect(
            obj.polygon,
            offset_x,
            offset_y,
            right,
            bottom,
        )
        clipped_area = polygon_area(clipped)
        visible_ratio = clipped_area / original_area
        if len(clipped) < 3 or visible_ratio + 1e-9 < min_visible_ratio:
            continue

        if visible_ratio >= 1.0 - 1e-6:
            local_polygon = tuple(
                (x - offset_x, y - offset_y) for x, y in obj.polygon
            )
            difficult = obj.difficult
            fully_included_indices.add(object_index)
        else:
            local_clipped = [
                (x - offset_x, y - offset_y) for x, y in clipped
            ]
            fitted = minimum_area_quadrilateral(local_clipped)
            local_polygon = tuple(
                (
                    min(max(x, 0.0), float(crop_width)),
                    min(max(y, 0.0), float(crop_height)),
                )
                for x, y in fitted
            )
            difficult = 2
            partial_count += 1
        tiled.append(
            DotaObject(
                polygon=local_polygon,  # type: ignore[arg-type]
                class_name=obj.class_name,
                difficult=difficult,
            )
        )
        included_indices.add(object_index)
    return tiled, partial_count, included_indices, fully_included_indices


def restore_polygon(
    polygon: Sequence[Point],
    offset_x: int,
    offset_y: int,
) -> tuple[Point, ...]:
    return tuple((x + offset_x, y + offset_y) for x, y in polygon)


def _find_image_files(image_dir: Path) -> list[Path]:
    supported = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in supported)


def process_dota_split(
    input_split_dir: Path,
    output_split_dir: Path,
    config: TileConfig,
    *,
    max_images: int | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """Tile one already-separated DOTA split without crossing original-image IDs."""
    config.validate()
    image_dir = input_split_dir / "images"
    label_dir = input_split_dir / "labelTxt"
    image_paths = _find_image_files(image_dir)
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        image_paths = image_paths[:max_images]

    output_image_dir = output_split_dir / "images"
    output_label_dir = output_split_dir / "labelTxt"
    mapping_path = output_split_dir / "mapping.jsonl"
    if not dry_run:
        if output_split_dir.exists() and any(output_split_dir.iterdir()):
            raise FileExistsError(f"output split is not empty: {output_split_dir}")
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

    mappings: list[TileMapping] = []
    total_objects = 0
    partial_objects = 0
    empty_tiles = 0
    padded_tiles = 0
    source_objects = 0
    covered_source_objects = 0
    fully_covered_source_objects = 0
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing label for {image_path}: {label_path}")
        metadata, objects = parse_dota_annotations(label_path)
        source_objects += len(objects)
        covered_indices: set[int] = set()
        fully_covered_indices: set[int] = set()
        with Image.open(image_path) as opened:
            width, height = opened.size
            image = None if dry_run else opened.convert("RGB")
        x_positions = sliding_positions(width, config.tile_size, config.overlap)
        y_positions = sliding_positions(height, config.tile_size, config.overlap)
        for offset_y in y_positions:
            for offset_x in x_positions:
                crop_width = min(config.tile_size, width - offset_x)
                crop_height = min(config.tile_size, height - offset_y)
                (
                    tile_objects,
                    tile_partial_count,
                    included_indices,
                    fully_included_indices,
                ) = _tile_annotations_with_indices(
                    objects,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    min_visible_ratio=config.min_visible_ratio,
                )
                covered_indices.update(included_indices)
                fully_covered_indices.update(fully_included_indices)
                tile_id = f"{image_path.stem}__x{offset_x:06d}_y{offset_y:06d}"
                mapping = TileMapping(
                    tile_id=tile_id,
                    source_image_id=image_path.stem,
                    source_image_path=str(image_path.resolve()),
                    source_label_path=str(label_path.resolve()),
                    offset_x=offset_x,
                    offset_y=offset_y,
                    source_width=width,
                    source_height=height,
                    crop_width=crop_width,
                    crop_height=crop_height,
                    tile_width=config.tile_size,
                    tile_height=config.tile_size,
                    object_count=len(tile_objects),
                    partial_object_count=tile_partial_count,
                )
                mappings.append(mapping)
                total_objects += len(tile_objects)
                partial_objects += tile_partial_count
                empty_tiles += int(not tile_objects)
                padded_tiles += int(
                    crop_width < config.tile_size or crop_height < config.tile_size
                )

                if dry_run:
                    continue
                assert image is not None
                crop = image.crop(
                    (
                        offset_x,
                        offset_y,
                        offset_x + crop_width,
                        offset_y + crop_height,
                    )
                )
                if crop.size != (config.tile_size, config.tile_size):
                    padded = Image.new(
                        "RGB",
                        (config.tile_size, config.tile_size),
                        color=(config.padding_value,) * 3,
                    )
                    padded.paste(crop, (0, 0))
                    crop = padded
                crop.save(output_image_dir / f"{tile_id}.png")
                label_lines = [*metadata, *(format_dota_object(obj) for obj in tile_objects)]
                (output_label_dir / f"{tile_id}.txt").write_text(
                    "\n".join(label_lines) + "\n",
                    encoding="utf-8",
                )
        covered_source_objects += len(covered_indices)
        fully_covered_source_objects += len(fully_covered_indices)

    summary: dict[str, object] = {
        "input_split_dir": str(input_split_dir.resolve()),
        "output_split_dir": str(output_split_dir.resolve()),
        "config": asdict(config),
        "source_images": len(image_paths),
        "source_objects": source_objects,
        "covered_source_objects": covered_source_objects,
        "fully_covered_source_objects": fully_covered_source_objects,
        "partial_only_source_objects": covered_source_objects - fully_covered_source_objects,
        "dropped_source_objects": source_objects - covered_source_objects,
        "tiles": len(mappings),
        "objects_in_tiles": total_objects,
        "partial_objects": partial_objects,
        "empty_tiles": empty_tiles,
        "padded_tiles": padded_tiles,
    }
    if not dry_run:
        mapping_path.write_text(
            "".join(json.dumps(asdict(mapping), ensure_ascii=False) + "\n" for mapping in mappings),
            encoding="utf-8",
        )
        (output_split_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary
