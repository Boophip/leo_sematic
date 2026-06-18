from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.data.dota_lite_split import (
    DOTA_V1_CLASSES,
    ImageRecord,
    discover_records,
    parse_dota_label,
    stratified_split,
)


class DotaLiteSplitTests(unittest.TestCase):
    def test_parse_dota_label_ignores_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "P0001.txt"
            path.write_text(
                "imagesource:GoogleEarth\n"
                "gsd:0.5\n"
                "0 0 1 0 1 1 0 1 plane 0\n"
                "2 2 3 2 3 3 2 3 plane 1\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_dota_label(path), Counter({"plane": 2}))

    def test_split_is_exact_disjoint_deterministic_and_covers_classes(self) -> None:
        records = []
        for index in range(120):
            counts = Counter(
                {
                    class_name: 1 + (index % 3)
                    for class_index, class_name in enumerate(DOTA_V1_CLASSES)
                    if (index + class_index) % 5 == 0
                }
            )
            records.append(
                ImageRecord(
                    image_id=f"P{index:04d}",
                    label_path=Path(f"P{index:04d}.txt"),
                    image_path=Path(f"P{index:04d}.png"),
                    class_counts=counts,
                )
            )

        kwargs = {
            "requested_sizes": {"train": 50, "val": 20, "test": 20},
            "seed": 7,
            "min_class_images": 1,
            "refinement_steps": 500,
        }
        first = stratified_split(records, **kwargs)
        second = stratified_split(records, **kwargs)

        self.assertEqual({name: len(items) for name, items in first.items()}, kwargs["requested_sizes"])
        selected_ids = [record.image_id for items in first.values() for record in items]
        self.assertEqual(len(selected_ids), len(set(selected_ids)))
        self.assertEqual(
            {name: [record.image_id for record in items] for name, items in first.items()},
            {name: [record.image_id for record in items] for name, items in second.items()},
        )
        for items in first.values():
            covered = set().union(*(record.classes for record in items))
            self.assertEqual(covered, set(DOTA_V1_CLASSES))

    def test_rejects_request_larger_than_available_records(self) -> None:
        record = ImageRecord("P0001", Path("P0001.txt"), None, Counter({"plane": 1}))
        with self.assertRaisesRegex(ValueError, "requested 2 images"):
            stratified_split([record], {"train": 1, "val": 1})

    def test_discovery_excludes_empty_labels_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labelTxt-v1.0" / "labelTxt"
            images = root / "images"
            labels.mkdir(parents=True)
            images.mkdir()
            (labels / "P0001.txt").write_text(
                "imagesource:GoogleEarth\ngsd:0.5\n",
                encoding="utf-8",
            )
            (images / "P0001.png").touch()

            records, report = discover_records(root)
            included, included_report = discover_records(root, include_empty=True)

            self.assertEqual(records, [])
            self.assertEqual(report.empty_label_ids, ("P0001",))
            self.assertEqual(report.usable_record_count, 0)
            self.assertEqual(len(included), 1)
            self.assertEqual(included_report.usable_record_count, 1)

    def test_minimum_class_images_is_a_hard_constraint(self) -> None:
        records = [
            ImageRecord(
                f"P{index:04d}",
                Path(f"P{index:04d}.txt"),
                None,
                Counter({"plane": 1}) if index < 9 else Counter(),
            )
            for index in range(30)
        ]
        splits = stratified_split(
            records,
            {"train": 10, "val": 5, "test": 5},
            min_class_images=3,
            refinement_steps=100,
        )
        for items in splits.values():
            self.assertGreaterEqual(
                sum("plane" in record.classes for record in items),
                3,
            )


if __name__ == "__main__":
    unittest.main()
