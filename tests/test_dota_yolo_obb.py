from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.dota_tiling import DotaObject
from src.data.dota_yolo_obb import (
    CLASS_TO_ID,
    ULTRALYTICS_DOTA_V1_CLASSES,
    YoloTileRecord,
    format_yolo_obb_object,
    prepare_yolo_obb_dataset,
    select_smoke_records,
    validate_yolo_obb_line,
)


class DotaYoloObbTests(unittest.TestCase):
    def test_official_class_order_is_fixed(self) -> None:
        self.assertEqual(CLASS_TO_ID["plane"], 0)
        self.assertEqual(CLASS_TO_ID["ship"], 1)
        self.assertEqual(CLASS_TO_ID["small-vehicle"], 10)
        self.assertEqual(CLASS_TO_ID["swimming-pool"], 14)

    def test_conversion_normalizes_and_filters_partial_difficulty(self) -> None:
        obj = DotaObject(
            polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)),
            class_name="ship",
            difficult=1,
        )
        line = format_yolo_obb_object(obj, 200, 100)
        self.assertIsNotNone(line)
        class_id, coordinates = validate_yolo_obb_line(line or "")
        self.assertEqual(class_id, 1)
        self.assertEqual(coordinates, (0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5))
        partial = DotaObject(obj.polygon, "ship", 2)
        self.assertIsNone(format_yolo_obb_object(partial, 200, 100))

    def test_validation_rejects_bad_lines(self) -> None:
        with self.assertRaisesRegex(ValueError, "9 fields"):
            validate_yolo_obb_line("0 0 0")
        with self.assertRaisesRegex(ValueError, "out of range"):
            validate_yolo_obb_line("15 0 0 1 0 1 1 0 1")
        with self.assertRaisesRegex(ValueError, "degenerate"):
            validate_yolo_obb_line("0 0 0 0 0 0 0 0 0")

    def test_smoke_selection_is_unique_deterministic_and_covers_classes(self) -> None:
        records = [
            YoloTileRecord(
                tile_id=f"P0001__x{index:06d}_y000000",
                image_path=f"/images/{index}.png",
                label_path=f"/labels/{index}.txt",
                source_image_id="P0001",
                class_ids=(index % len(ULTRALYTICS_DOTA_V1_CLASSES),),
                object_count=1 + index % 4,
            )
            for index in range(45)
        ]
        first = select_smoke_records(records, 32, seed=42)
        second = select_smoke_records(records, 32, seed=42)
        self.assertEqual([record.tile_id for record in first], [record.tile_id for record in second])
        self.assertEqual(len({record.tile_id for record in first}), 32)
        self.assertEqual(
            set().union(*(set(record.class_ids) for record in first)),
            set(range(len(ULTRALYTICS_DOTA_V1_CLASSES))),
        )

    def test_prepare_preserves_split_isolation_and_writes_empty_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiles_root = root / "tiles"
            output_root = root / "yolo"
            for split_index, split in enumerate(("train", "val", "test")):
                image_dir = tiles_root / split / "images"
                label_dir = tiles_root / split / "labelTxt"
                image_dir.mkdir(parents=True)
                label_dir.mkdir()
                count = 2 if split == "train" else 1
                for tile_index in range(count):
                    image_id = f"P{split_index}{tile_index:03d}"
                    tile_id = f"{image_id}__x000000_y000000"
                    Image.new("RGB", (64, 64)).save(image_dir / f"{tile_id}.png")
                    label_text = (
                        "imagesource:test\n"
                        "gsd:1\n"
                        "0 0 32 0 32 32 0 32 plane 0\n"
                        if tile_index == 0
                        else "imagesource:test\ngsd:1\n"
                    )
                    (label_dir / f"{tile_id}.txt").write_text(label_text, encoding="utf-8")

            summary = prepare_yolo_obb_dataset(
                tiles_root,
                output_root,
                smoke_size=1,
            )

            self.assertEqual(summary["splits"]["train"]["images"], 2)
            self.assertEqual(summary["splits"]["train"]["empty_labels"], 1)
            empty_label = output_root / "labels" / "train" / "P0001__x000000_y000000.txt"
            self.assertEqual(empty_label.read_text(encoding="utf-8"), "")
            manifest = json.loads(
                (output_root / "overfit1_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["covered_classes"], ["plane"])


if __name__ == "__main__":
    unittest.main()
