from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.profiling.profile_generation import (
    ExitProfile,
    PROFILE_FIELDNAMES,
    load_roi_metadata,
    profile_roi_records,
    write_profile_csv,
)


class RoiProfilingTests(unittest.TestCase):
    def test_profile_rows_cover_roi_exit_and_compression_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_path = root / "crop.png"
            Image.new("RGB", (24, 32), color=(200, 60, 30)).save(crop_path)
            metadata_path = root / "roi_metadata_labeled.jsonl"
            metadata_path.write_text(
                json.dumps(
                    {
                        "roi_id": "P0001__roi000000",
                        "image_id": "P0001",
                        "crop_path": str(crop_path),
                        "predicted_class": "plane",
                        "class_id": 0,
                        "confidence": 0.8,
                        "area_ratio": 0.01,
                        "semantic_value": 0.9,
                        "grid_id": 7,
                        "class_priority": 1.0,
                        "quality_label": "true_positive",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            records = load_roi_metadata(metadata_path)

            def fake_predictor(_image, _record):
                return [
                    ExitProfile(1, 1.5, 0.25, 0, 0.5),
                    ExitProfile(2, 2.5, 0.75, 0, 0.9),
                ]

            rows = profile_roi_records(records, fake_predictor, compression_levels=[0, 1])

            self.assertEqual(len(rows), 4)
            self.assertEqual({row.exit_level for row in rows}, {1, 2})
            self.assertEqual({row.compression_level for row in rows}, {"beta_0", "beta_1"})
            for row in rows:
                self.assertIn("task_quality", PROFILE_FIELDNAMES)
                self.assertIn("detector_confidence", PROFILE_FIELDNAMES)
                self.assertIn("exit_confidence", PROFILE_FIELDNAMES)
                self.assertNotIn("confidence", PROFILE_FIELDNAMES)
                self.assertEqual(row.class_id, 0)
                self.assertAlmostEqual(row.detector_confidence, 0.8)
                self.assertAlmostEqual(row.area_ratio, 0.01)
                self.assertAlmostEqual(row.semantic_value, 0.9)
                self.assertEqual(row.grid_id, 7)
                self.assertAlmostEqual(row.class_priority, 1.0)
                self.assertGreater(row.compressed_bytes, 0)
                self.assertGreaterEqual(row.task_quality, 0.0)
                self.assertLessEqual(row.task_quality, 1.0)

            csv_path = root / "profile.csv"
            write_profile_csv(csv_path, rows)
            header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertEqual(tuple(header), PROFILE_FIELDNAMES)


if __name__ == "__main__":
    unittest.main()
