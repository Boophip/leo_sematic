from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from src.multi_exit.dataset import RoiCropDataset
    from src.multi_exit.model import TinyMultiExitCNN, multi_exit_cross_entropy
    from src.multi_exit.training import train_multi_exit_model
except Exception as exc:  # pragma: no cover - only used when torch is unavailable.
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_IMPORT_ERROR}")
class MultiExitTests(unittest.TestCase):
    def test_forward_returns_raw_logits_for_each_exit(self) -> None:
        model = TinyMultiExitCNN(num_classes=15)
        inputs = torch.rand(2, 3, 64, 64)

        outputs = model(inputs)

        self.assertEqual(len(outputs), 4)
        for logits in outputs:
            self.assertEqual(tuple(logits.shape), (2, 15))
            self.assertFalse(torch.allclose(logits.sum(dim=1), torch.ones(2)))

    def test_cross_entropy_uses_raw_logits_not_softmax_probabilities(self) -> None:
        logits = torch.tensor([[4.0, -1.0], [-2.0, 3.0]])
        targets = torch.tensor([0, 1])

        loss = multi_exit_cross_entropy([logits], targets)
        raw_loss = F.cross_entropy(logits, targets)
        softmax_loss = F.cross_entropy(F.softmax(logits, dim=1), targets)

        self.assertTrue(torch.allclose(loss, raw_loss))
        self.assertFalse(torch.allclose(loss, softmax_loss))

    def test_roi_crop_dataset_filters_true_positive_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop_path = root / "crop.png"
            Image.new("RGB", (20, 30), color=(10, 20, 30)).save(crop_path)
            metadata_path = root / "roi_metadata_labeled.jsonl"
            rows = [
                {
                    "roi_id": "tp",
                    "crop_path": str(crop_path),
                    "class_id": 0,
                    "quality_label": "true_positive",
                },
                {
                    "roi_id": "fp",
                    "crop_path": str(crop_path),
                    "class_id": 0,
                    "quality_label": "background_fp",
                },
            ]
            metadata_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            dataset = RoiCropDataset(metadata_path, input_size=32)
            image, label = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(int(label.item()), 0)

    def test_formal_training_writes_best_last_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_metadata = root / "train.jsonl"
            val_metadata = root / "val.jsonl"
            output_dir = root / "out"
            rows = []
            for index, class_id in enumerate([0, 1, 0, 1]):
                crop_path = root / f"crop_{index}.png"
                Image.new("RGB", (20, 30), color=(20 + index, 30, 40)).save(crop_path)
                rows.append(
                    {
                        "roi_id": f"roi_{index}",
                        "crop_path": str(crop_path),
                        "class_id": class_id,
                        "quality_label": "true_positive",
                    }
                )
            train_metadata.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[:2]),
                encoding="utf-8",
            )
            val_metadata.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[2:]),
                encoding="utf-8",
            )

            summary = train_multi_exit_model(
                train_metadata,
                val_metadata,
                output_dir,
                epochs=1,
                batch_size=2,
                input_size=32,
                device="cpu",
            )

            self.assertTrue((output_dir / "best.pt").is_file())
            self.assertTrue((output_dir / "last.pt").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertEqual(summary["counts"]["train_roi_count"], 2)
            self.assertEqual(summary["counts"]["val_roi_count"], 2)
            self.assertIn("history", summary)


if __name__ == "__main__":
    unittest.main()
