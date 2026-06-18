from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.detector.dota_original_eval import (
    GroundTruth,
    RestoredPrediction,
    TileInfo,
    TilePrediction,
    evaluate_predictions,
    global_rotated_nms,
    restore_tile_prediction,
    rotated_iou,
    write_dota_task1,
)


def square(left: float, top: float, size: float):
    return (
        (left, top),
        (left + size, top),
        (left + size, top + size),
        (left, top + size),
    )


class DotaOriginalEvalTests(unittest.TestCase):
    def test_restore_adds_tile_offset(self) -> None:
        mapping = TileInfo(
            tile_id="P0001__x000100_y000200",
            source_image_id="P0001",
            source_image_path="P0001.png",
            source_label_path="P0001.txt",
            offset_x=100,
            offset_y=200,
            source_width=500,
            source_height=600,
            crop_width=256,
            crop_height=256,
            tile_width=256,
            tile_height=256,
        )
        prediction = TilePrediction("P0001__x000100_y000200", 0, "plane", 0.9, square(1, 2, 10))

        restored = restore_tile_prediction(prediction, mapping)

        self.assertIsNotNone(restored)
        self.assertEqual(
            restored.polygon if restored else None,
            ((101.0, 202.0), (111.0, 202.0), (111.0, 212.0), (101.0, 212.0)),
        )

    def test_padding_center_prediction_is_dropped(self) -> None:
        mapping = TileInfo(
            tile_id="P0001__x000000_y000000",
            source_image_id="P0001",
            source_image_path="P0001.png",
            source_label_path="P0001.txt",
            offset_x=0,
            offset_y=0,
            source_width=50,
            source_height=50,
            crop_width=50,
            crop_height=50,
            tile_width=100,
            tile_height=100,
        )
        prediction = TilePrediction("P0001__x000000_y000000", 0, "plane", 0.9, square(70, 10, 10))

        self.assertIsNone(restore_tile_prediction(prediction, mapping))

    def test_rotated_iou_handles_overlap_cases(self) -> None:
        self.assertAlmostEqual(rotated_iou(square(0, 0, 10), square(0, 0, 10)), 1.0)
        self.assertAlmostEqual(rotated_iou(square(0, 0, 10), square(20, 20, 5)), 0.0)
        self.assertAlmostEqual(
            rotated_iou(square(0, 0, 10), square(5, 5, 10)),
            25.0 / 175.0,
        )

    def test_global_nms_is_per_image_and_per_class(self) -> None:
        predictions = [
            RestoredPrediction("tile_a", "P0001", 0, "plane", 0.9, square(0, 0, 10)),
            RestoredPrediction("tile_b", "P0001", 0, "plane", 0.8, square(0, 0, 10)),
            RestoredPrediction("tile_c", "P0001", 1, "ship", 0.7, square(0, 0, 10)),
            RestoredPrediction("tile_d", "P0002", 0, "plane", 0.6, square(0, 0, 10)),
        ]

        kept = global_rotated_nms(predictions, iou_threshold=0.5)

        self.assertEqual({prediction.tile_id for prediction in kept}, {"tile_a", "tile_c", "tile_d"})

    def test_evaluation_counts_duplicates_and_ignores_difficult_gt(self) -> None:
        predictions = [
            RestoredPrediction("tile_a", "P0001", 0, "plane", 0.9, square(0, 0, 10)),
            RestoredPrediction("tile_b", "P0001", 0, "plane", 0.8, square(0, 0, 10)),
            RestoredPrediction("tile_c", "P0001", 0, "plane", 0.7, square(30, 0, 10)),
        ]
        ground_truths = [
            GroundTruth("P0001", 0, "plane", square(0, 0, 10), difficult=0),
            GroundTruth("P0001", 0, "plane", square(30, 0, 10), difficult=1),
        ]

        report = evaluate_predictions(predictions, ground_truths, iou_threshold=0.5)
        plane = report["per_class"]["plane"]

        self.assertEqual(plane["true_positives"], 1)
        self.assertEqual(plane["false_positives"], 1)
        self.assertEqual(plane["ignored_predictions"], 1)
        self.assertEqual(plane["ground_truth"], 1)
        self.assertEqual(plane["ignored_ground_truth"], 1)
        self.assertAlmostEqual(plane["AP50"], 1.0)

    def test_write_dota_task1_format(self) -> None:
        prediction = RestoredPrediction("tile_a", "P0001", 0, "plane", 0.987654, square(0, 0, 10))
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_dota_task1(output_dir, [prediction], class_names=("plane", "ship"))

            self.assertEqual(
                (output_dir / "Task1_plane.txt").read_text(encoding="utf-8"),
                "P0001 0.987654 0.00 0.00 10.00 0.00 10.00 10.00 0.00 10.00\n",
            )
            self.assertEqual((output_dir / "Task1_ship.txt").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()

