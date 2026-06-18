from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.dota_tiling import (
    DotaObject,
    TileConfig,
    parse_dota_annotations,
    process_dota_split,
    restore_polygon,
    sliding_positions,
    tile_annotations,
)


class DotaTilingTests(unittest.TestCase):
    def test_sliding_positions_cover_variable_axis_sizes(self) -> None:
        self.assertEqual(sliding_positions(400, 1024, 200), [0])
        self.assertEqual(sliding_positions(1024, 1024, 200), [0])
        positions = sliding_positions(4000, 1024, 200)
        self.assertEqual(positions, [0, 824, 1648, 2472, 2976])
        self.assertEqual(positions[-1] + 1024, 4000)

    def test_full_object_translation_can_be_restored_exactly(self) -> None:
        original = DotaObject(
            polygon=((110.0, 210.0), (150.0, 210.0), (150.0, 250.0), (110.0, 250.0)),
            class_name="plane",
            difficult=0,
        )
        tiled, partial_count = tile_annotations(
            [original],
            offset_x=100,
            offset_y=200,
            crop_width=1024,
            crop_height=1024,
            min_visible_ratio=0.7,
        )
        self.assertEqual(partial_count, 0)
        self.assertEqual(
            restore_polygon(tiled[0].polygon, 100, 200),
            original.polygon,
        )

    def test_partial_object_is_clipped_and_marked_difficult(self) -> None:
        crossing = DotaObject(
            polygon=((-20.0, 20.0), (80.0, 20.0), (80.0, 80.0), (-20.0, 80.0)),
            class_name="ship",
            difficult=0,
        )
        tiled, partial_count = tile_annotations(
            [crossing],
            offset_x=0,
            offset_y=0,
            crop_width=100,
            crop_height=100,
            min_visible_ratio=0.7,
        )
        self.assertEqual(len(tiled), 1)
        self.assertEqual(partial_count, 1)
        self.assertEqual(tiled[0].difficult, 2)
        self.assertTrue(
            all(0 <= x <= 100 and 0 <= y <= 100 for x, y in tiled[0].polygon)
        )

    def test_end_to_end_pads_small_image_and_writes_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_split = root / "input" / "train"
            output_split = root / "output" / "train"
            (input_split / "images").mkdir(parents=True)
            (input_split / "labelTxt").mkdir()
            Image.new("RGB", (400, 300), color="white").save(
                input_split / "images" / "P0001.png"
            )
            (input_split / "labelTxt" / "P0001.txt").write_text(
                "imagesource:GoogleEarth\n"
                "gsd:0.5\n"
                "10 10 30 10 30 30 10 30 plane 0\n",
                encoding="utf-8",
            )

            summary = process_dota_split(
                input_split,
                output_split,
                TileConfig(tile_size=512, overlap=100),
            )

            self.assertEqual(summary["tiles"], 1)
            self.assertEqual(summary["padded_tiles"], 1)
            self.assertEqual(summary["source_objects"], 1)
            self.assertEqual(summary["covered_source_objects"], 1)
            self.assertEqual(summary["fully_covered_source_objects"], 1)
            self.assertEqual(summary["partial_only_source_objects"], 0)
            self.assertEqual(summary["dropped_source_objects"], 0)
            tile_path = next((output_split / "images").iterdir())
            with Image.open(tile_path) as image:
                self.assertEqual(image.size, (512, 512))
            mapping = json.loads((output_split / "mapping.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(mapping["source_width"], 400)
            self.assertEqual(mapping["source_height"], 300)
            _, objects = parse_dota_annotations(next((output_split / "labelTxt").iterdir()))
            self.assertEqual(len(objects), 1)


if __name__ == "__main__":
    unittest.main()
