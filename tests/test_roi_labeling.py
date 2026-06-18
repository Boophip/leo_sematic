from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.roi_labeling import (
    ROI_LABEL_FIELDNAMES,
    build_labeled_roi_metadata_dataset,
    label_roi_records,
    load_indexed_ground_truths,
    load_roi_metadata,
)


def square(left: float, top: float, size: float):
    return (
        (left, top),
        (left + size, top),
        (left + size, top + size),
        (left, top + size),
    )


def roi_payload(
    roi_id: str,
    image_id: str,
    class_id: int,
    predicted_class: str,
    confidence: float,
    polygon,
) -> dict[str, object]:
    center_x = sum(point[0] for point in polygon) / 4.0
    center_y = sum(point[1] for point in polygon) / 4.0
    return {
        "roi_id": roi_id,
        "image_id": image_id,
        "source_image_path": f"{image_id}.png",
        "crop_path": f"{roi_id}.png",
        "predicted_class": predicted_class,
        "class_id": class_id,
        "confidence": confidence,
        "global_obb": [[x, y] for x, y in polygon],
        "center_x": center_x,
        "center_y": center_y,
        "area_pixels": 100.0,
        "area_ratio": 0.01,
        "grid_row": 0,
        "grid_col": 0,
        "grid_id": 0,
        "semantic_value": 0.9,
        "class_priority": 1.0,
        "difficult": 0,
        "source_tile_id": f"{image_id}__x000000_y000000",
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class RoiLabelingTests(unittest.TestCase):
    def test_labels_tp_duplicate_ignored_mismatch_and_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "roi_metadata.jsonl"
            label_dir = root / "labelTxt"
            label_dir.mkdir()
            (label_dir / "P0001.txt").write_text(
                "0 0 10 0 10 10 0 10 plane 0\n"
                "20 0 30 0 30 10 20 10 ship 1\n"
                "40 0 50 0 50 10 40 10 bridge 0\n",
                encoding="utf-8",
            )
            write_jsonl(
                metadata_path,
                [
                    roi_payload("roi_low_duplicate", "P0001", 0, "plane", 0.10, square(0, 0, 10)),
                    roi_payload("roi_high_tp", "P0001", 0, "plane", 0.90, square(0, 0, 10)),
                    roi_payload("roi_ignored", "P0001", 1, "ship", 0.80, square(20, 0, 10)),
                    roi_payload("roi_mismatch", "P0001", 0, "plane", 0.70, square(40, 0, 10)),
                    roi_payload("roi_background", "P0001", 0, "plane", 0.60, square(70, 0, 10)),
                ],
            )

            rois = load_roi_metadata(metadata_path)
            ground_truths = load_indexed_ground_truths(label_dir)
            rows = label_roi_records(rois, ground_truths, iou_threshold=0.5)
            by_id = {str(row["roi_id"]): row for row in rows}

            self.assertEqual(by_id["roi_high_tp"]["quality_label"], "true_positive")
            self.assertEqual(by_id["roi_high_tp"]["is_true_positive"], 1)
            self.assertEqual(by_id["roi_low_duplicate"]["quality_label"], "duplicate_detection")
            self.assertEqual(by_id["roi_low_duplicate"]["is_false_positive"], 1)
            self.assertEqual(by_id["roi_ignored"]["quality_label"], "ignored_difficult")
            self.assertEqual(by_id["roi_ignored"]["is_ignored"], 1)
            self.assertEqual(by_id["roi_ignored"]["is_false_positive"], 0)
            self.assertEqual(by_id["roi_mismatch"]["quality_label"], "class_mismatch")
            self.assertEqual(by_id["roi_mismatch"]["matched_gt_class"], "bridge")
            self.assertEqual(by_id["roi_background"]["quality_label"], "background_fp")

    def test_dataset_outputs_labeled_schema_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "roi_metadata.jsonl"
            label_dir = root / "labelTxt"
            output_dir = root / "out"
            label_dir.mkdir()
            (label_dir / "P0001.txt").write_text(
                "0 0 10 0 10 10 0 10 plane 0\n",
                encoding="utf-8",
            )
            write_jsonl(
                metadata_path,
                [
                    roi_payload("roi_fp", "P0001", 0, "plane", 0.10, square(30, 0, 10)),
                    roi_payload("roi_tp", "P0001", 0, "plane", 0.90, square(0, 0, 10)),
                ],
            )

            summary = build_labeled_roi_metadata_dataset(
                metadata_path,
                label_dir,
                output_dir,
                iou_threshold=0.5,
            )

            self.assertEqual(summary["counts"]["roi_count"], 2)
            self.assertEqual(summary["counts"]["true_positive"], 1)
            self.assertEqual(summary["counts"]["background_fp"], 1)
            self.assertTrue((output_dir / "roi_metadata_labeled.jsonl").is_file())
            self.assertTrue((output_dir / "roi_metadata_labeled.csv").is_file())
            self.assertTrue((output_dir / "roi_label_summary.json").is_file())

            json_record = json.loads(
                (output_dir / "roi_metadata_labeled.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(tuple(json_record.keys()), ROI_LABEL_FIELDNAMES)

            with (output_dir / "roi_metadata_labeled.csv").open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or []), ROI_LABEL_FIELDNAMES)
                self.assertEqual(len(list(reader)), 2)


if __name__ == "__main__":
    unittest.main()
