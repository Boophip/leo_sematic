from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.gt_roi_crops import (
    GT_ROI_FIELDNAMES,
    build_gt_roi_crop_dataset,
    load_split_manifest,
)


class GtRoiCropTests(unittest.TestCase):
    def test_build_gt_roi_crop_dataset_writes_split_isolated_crops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            label_dir = root / "labelTxt"
            output_dir = root / "gt_rois" / "train"
            image_dir.mkdir()
            label_dir.mkdir()
            image_path = image_dir / "P0001.png"
            label_path = label_dir / "P0001.txt"
            Image.new("RGB", (100, 80), color=(30, 80, 140)).save(image_path)
            label_path.write_text(
                "imagesource:GoogleEarth\n"
                "gsd:0.5\n"
                "10 10 30 10 30 30 10 30 plane 0\n"
                "40 40 50 40 50 50 40 50 ship 1\n",
                encoding="utf-8",
            )
            manifest_path = root / "train.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_id", "image_path", "label_path", "classes", "objects"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "image_id": "P0001",
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "classes": "plane ship",
                        "objects": "2",
                    }
                )

            summary = build_gt_roi_crop_dataset(
                manifest_path,
                output_dir,
                split="train",
                include_difficult=False,
            )

            self.assertEqual(summary["counts"]["roi_count"], 1)
            self.assertEqual(summary["counts"]["crop_count"], 1)
            self.assertTrue((output_dir / "roi_metadata_labeled.jsonl").is_file())
            self.assertTrue((output_dir / "roi_metadata_labeled.csv").is_file())
            self.assertEqual(len(list((output_dir / "crops").glob("*.png"))), 1)

            row = json.loads(
                (output_dir / "roi_metadata_labeled.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(tuple(row.keys()), GT_ROI_FIELDNAMES)
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["quality_label"], "true_positive")
            self.assertEqual(row["predicted_class"], "plane")
            self.assertEqual(row["confidence"], 1.0)

    def test_load_split_manifest_respects_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "train.csv"
            with manifest_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["image_id", "image_path", "label_path", "classes", "objects"],
                )
                writer.writeheader()
                for index in range(3):
                    writer.writerow(
                        {
                            "image_id": f"P{index:04d}",
                            "image_path": f"P{index:04d}.png",
                            "label_path": f"P{index:04d}.txt",
                            "classes": "plane",
                            "objects": "1",
                        }
                    )

            records = load_split_manifest(manifest_path, limit_images=2)

            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
