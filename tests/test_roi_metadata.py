from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.roi_metadata import (
    ROI_FIELDNAMES,
    TASK_CLASS_PRIORITIES,
    ImageInfo,
    aosi_grid_value,
    build_roi_metadata_dataset,
    grid_assignment,
    make_roi_records,
    semantic_value,
    validate_class_priorities,
    write_rotated_crop,
)
from src.detector.dota_original_eval import (
    RestoredPrediction,
    ULTRALYTICS_DOTA_V1_CLASSES,
)


def square(left: float, top: float, size: float):
    return (
        (left, top),
        (left + size, top),
        (left + size, top + size),
        (left, top + size),
    )


def write_nms_jsonl(path: Path, predictions: list[RestoredPrediction]) -> None:
    lines = []
    for prediction in predictions:
        lines.append(
            json.dumps(
                {
                    "tile_id": prediction.tile_id,
                    "source_image_id": prediction.source_image_id,
                    "class_id": prediction.class_id,
                    "class_name": prediction.class_name,
                    "confidence": prediction.confidence,
                    "polygon": [[x, y] for x, y in prediction.polygon],
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RoiMetadataTests(unittest.TestCase):
    def test_semantic_value_formula_and_clip(self) -> None:
        self.assertAlmostEqual(
            semantic_value(class_priority=0.6, confidence=0.5, area_ratio=0.1),
            0.50 * 0.6 + 0.35 * 0.5 + 0.15 * 0.1,
        )
        self.assertEqual(
            semantic_value(class_priority=1.0, confidence=1.0, area_ratio=10.0),
            1.0,
        )
        self.assertEqual(
            semantic_value(class_priority=-1.0, confidence=-1.0, area_ratio=-1.0),
            0.0,
        )

    def test_class_priority_table_covers_all_dota_classes(self) -> None:
        validate_class_priorities()
        self.assertEqual(set(TASK_CLASS_PRIORITIES), set(ULTRALYTICS_DOTA_V1_CLASSES))

    def test_grid_assignment_clamps_boundaries(self) -> None:
        self.assertEqual(grid_assignment(0.0, 0.0, 100, 80), (0, 0, 0))
        self.assertEqual(grid_assignment(99.0, 79.0, 100, 80), (7, 7, 63))
        self.assertEqual(grid_assignment(100.0, 80.0, 100, 80), (7, 7, 63))
        self.assertEqual(grid_assignment(-5.0, -5.0, 100, 80), (0, 0, 0))

    def test_area_ratio_uses_original_image_area(self) -> None:
        image_info = ImageInfo(
            image_id="P0001",
            path=Path("P0001.png"),
            width=100,
            height=200,
        )
        prediction = RestoredPrediction(
            tile_id="P0001__x000000_y000000",
            source_image_id="P0001",
            class_id=0,
            class_name="plane",
            confidence=0.8,
            polygon=square(10, 20, 10),
        )

        records = make_roi_records(
            [prediction],
            {"P0001": image_info},
            Path("out"),
            make_crops=False,
        )

        self.assertAlmostEqual(records[0].area_pixels, 100.0)
        self.assertAlmostEqual(records[0].area_ratio, 100.0 / (100.0 * 200.0))

    def test_aosi_grid_value_uses_independent_product(self) -> None:
        self.assertAlmostEqual(aosi_grid_value([0.5, 0.2]), 1.0 - (0.5 * 0.8))
        self.assertAlmostEqual(aosi_grid_value([]), 0.0)

    def test_rotated_crop_file_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "P0001.png"
            output_path = root / "crop.png"
            Image.new("RGB", (64, 64), color=(20, 80, 140)).save(image_path)

            write_rotated_crop(image_path, square(8, 12, 24), output_path)

            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as crop:
                self.assertGreater(crop.size[0], 0)
                self.assertGreater(crop.size[1], 0)

    def test_dataset_outputs_schema_crops_and_grid_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            output_dir = root / "roi"
            image_dir.mkdir()
            Image.new("RGB", (100, 80), color="white").save(image_dir / "P0001.png")
            predictions_path = root / "nms_predictions.jsonl"
            predictions = [
                RestoredPrediction(
                    tile_id="P0001__x000000_y000000",
                    source_image_id="P0001",
                    class_id=0,
                    class_name="plane",
                    confidence=0.9,
                    polygon=square(10, 10, 20),
                ),
                RestoredPrediction(
                    tile_id="P0001__x000010_y000000",
                    source_image_id="P0001",
                    class_id=3,
                    class_name="baseball-diamond",
                    confidence=0.5,
                    polygon=square(70, 60, 10),
                ),
            ]
            write_nms_jsonl(predictions_path, predictions)

            summary = build_roi_metadata_dataset(
                predictions_path,
                image_dir,
                output_dir,
            )

            self.assertEqual(summary["counts"]["roi_count"], 2)
            self.assertEqual(summary["counts"]["crop_count"], 2)
            self.assertTrue((output_dir / "roi_metadata.jsonl").is_file())
            self.assertTrue((output_dir / "roi_metadata.csv").is_file())
            self.assertTrue((output_dir / "grid_summary.json").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertEqual(len(list((output_dir / "crops").glob("*.png"))), 2)

            json_record = json.loads(
                (output_dir / "roi_metadata.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(tuple(json_record.keys()), ROI_FIELDNAMES)
            self.assertEqual(json_record["difficult"], 0)
            self.assertEqual(json_record["grid_id"], json_record["grid_row"] * 8 + json_record["grid_col"])

            with (output_dir / "roi_metadata.csv").open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or []), ROI_FIELDNAMES)
                csv_rows = list(reader)
            self.assertEqual(len(csv_rows), 2)

            grid_summary = json.loads((output_dir / "grid_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(grid_summary["image_count"], 1)
            self.assertEqual(grid_summary["cell_count"], 64)
            self.assertGreaterEqual(grid_summary["non_empty_cell_count"], 1)


if __name__ == "__main__":
    unittest.main()
