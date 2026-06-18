from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.detector.yolo_obb_reporting import best_training_row, read_training_rows


class YoloObbReportingTests(unittest.TestCase):
    def test_reads_results_and_selects_best_map_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["epoch", " metrics/mAP50-95(B)", " train/box_loss"],
                )
                writer.writeheader()
                writer.writerow(
                    {"epoch": "1", " metrics/mAP50-95(B)": "0.2", " train/box_loss": "1.2"}
                )
                writer.writerow(
                    {"epoch": "2", " metrics/mAP50-95(B)": "0.7", " train/box_loss": "0.8"}
                )
            rows = read_training_rows(path)
            self.assertEqual(best_training_row(rows)["epoch"], 2.0)
            self.assertEqual(best_training_row(rows)["train/box_loss"], 0.8)


if __name__ == "__main__":
    unittest.main()
