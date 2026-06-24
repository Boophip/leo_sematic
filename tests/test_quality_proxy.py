from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.proxy.quality_proxy import (
    FEATURE_COLUMNS,
    IMAGE_FEATURE_COLUMNS,
    LEAKAGE_FIELDS,
    MODEL_TYPES,
    load_quality_proxy,
    split_profile_frame,
    train_quality_proxy_strict,
    train_quality_proxy_smoke,
    validate_feature_policy,
)


def make_profile_frame() -> pd.DataFrame:
    rows = []
    for image_index, image_id in enumerate(["P0001", "P0002", "P0003"]):
        for exit_level in [1, 2]:
            for compression_level in ["beta_0", "beta_1"]:
                quality = 0.2 + 0.2 * exit_level - 0.05 * image_index
                if compression_level == "beta_1":
                    quality -= 0.03
                rows.append(
                    {
                        "roi_id": f"{image_id}__roi000000",
                        "image_id": image_id,
                        "predicted_class": "plane" if image_index < 2 else "ship",
                        "class_id": image_index % 2,
                        "detector_confidence": 0.7 + 0.05 * image_index,
                        "area_ratio": 0.01 + 0.001 * image_index,
                        "semantic_value": 0.8 - 0.05 * image_index,
                        "grid_id": image_index,
                        "class_priority": 1.0,
                        "quality_label": "true_positive",
                        "exit_level": exit_level,
                        "compression_level": compression_level,
                        "compressed_bytes": 1000 - 100 * image_index,
                        "encode_ms": 0.1,
                        "decode_ms": 0.1,
                        "inference_ms": 1.0 + exit_level,
                        "task_quality": max(min(quality, 1.0), 0.0),
                        "encoded_format": "PNG",
                        "output_width": 32,
                        "output_height": 48,
                        "crop_width": 32,
                        "crop_height": 48,
                        "crop_aspect_ratio": 32 / 48,
                        "brightness_mean": 0.4 + 0.05 * image_index,
                        "brightness_std": 0.1 + 0.02 * exit_level,
                        "rgb_mean_r": 0.3,
                        "rgb_mean_g": 0.4,
                        "rgb_mean_b": 0.5,
                        "rgb_std_r": 0.1,
                        "rgb_std_g": 0.2,
                        "rgb_std_b": 0.3,
                        "laplacian_var": 0.2 + 0.1 * exit_level,
                        "edge_density": 0.05 + 0.01 * image_index,
                        "entropy": 1.0 + 0.1 * image_index,
                        "predicted_class_id": 0,
                        "exit_confidence": 0.9,
                    }
                )
    return pd.DataFrame(rows)


class QualityProxyTests(unittest.TestCase):
    def test_feature_policy_excludes_leakage_fields(self) -> None:
        validate_feature_policy()
        self.assertFalse(set(FEATURE_COLUMNS) & set(LEAKAGE_FIELDS))
        self.assertTrue(set(IMAGE_FEATURE_COLUMNS).issubset(set(FEATURE_COLUMNS)))

    def test_group_split_keeps_image_ids_disjoint(self) -> None:
        frame = make_profile_frame()

        split = split_profile_frame(frame, test_size=0.34, seed=7)

        self.assertEqual(split.summary["strategy"], "group_shuffle_split")
        train_images = set(split.train["image_id"])
        test_images = set(split.test["image_id"])
        self.assertFalse(train_images & test_images)

    def test_single_image_falls_back_to_smoke_row_split(self) -> None:
        frame = make_profile_frame()
        one_image = frame[frame["image_id"] == "P0001"].reset_index(drop=True)

        split = split_profile_frame(one_image, test_size=0.5, seed=7)

        self.assertEqual(split.summary["strategy"], "row_split_smoke_only")
        self.assertGreater(len(split.train), 0)
        self.assertGreater(len(split.test), 0)

    def test_train_save_load_and_predict_quality(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_csv = root / "profile.csv"
            output_dir = root / "proxy"
            make_profile_frame().to_csv(profile_csv, index=False)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                summary = train_quality_proxy_smoke(
                    profile_csv,
                    output_dir,
                    test_size=0.34,
                    seed=3,
                    max_iter=50,
                    hidden_layer_sizes=(8,),
                    model_type="mlp",
                    include_image_features=True,
                    quality_threshold=0.5,
                )
            model = load_quality_proxy(output_dir / "quality_proxy.joblib")
            predictions = model.predict_quality(make_profile_frame().head(3))

            self.assertTrue((output_dir / "quality_proxy.joblib").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertIn("mae", summary["metrics"]["test"])
            self.assertIn("mse", summary["metrics"]["test"])
            self.assertIn("r2", summary["metrics"]["test"])
            self.assertEqual(summary["configuration"]["model_type"], "mlp")
            self.assertIn("threshold_metrics", summary["metrics"]["test"])
            self.assertIn("action_ranking", summary["metrics"]["test"])
            self.assertIn("high_value", summary["metrics"]["test"])
            self.assertEqual(len(predictions), 3)
            self.assertTrue(((predictions >= 0.0) & (predictions <= 1.0)).all())

    def test_train_save_load_and_predict_quality_with_each_model_type(self) -> None:
        for model_type in MODEL_TYPES:
            with self.subTest(model_type=model_type):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    profile_csv = root / "profile.csv"
                    output_dir = root / "proxy"
                    make_profile_frame().to_csv(profile_csv, index=False)

                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        summary = train_quality_proxy_smoke(
                            profile_csv,
                            output_dir,
                            test_size=0.34,
                            seed=3,
                            max_iter=50,
                            hidden_layer_sizes=(8,),
                            model_type=model_type,
                            include_image_features=True,
                            quality_threshold=0.5,
                        )
                    model = load_quality_proxy(output_dir / "quality_proxy.joblib")
                    predictions = model.predict_quality(make_profile_frame().head(2))

                    self.assertEqual(summary["configuration"]["model_type"], model_type)
                    self.assertEqual(model.model_type, model_type)
                    self.assertEqual(len(predictions), 2)
                    self.assertTrue(((predictions >= 0.0) & (predictions <= 1.0)).all())

    def test_strict_training_uses_explicit_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv = root / "train.csv"
            val_csv = root / "val.csv"
            test_csv = root / "test.csv"
            output_dir = root / "strict_proxy"
            frame = make_profile_frame()
            frame[frame["image_id"] == "P0001"].to_csv(train_csv, index=False)
            frame[frame["image_id"] == "P0002"].to_csv(val_csv, index=False)
            frame[frame["image_id"] == "P0003"].to_csv(test_csv, index=False)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                summary = train_quality_proxy_strict(
                    train_csv,
                    val_csv,
                    test_csv,
                    output_dir,
                    seed=3,
                    max_iter=50,
                    hidden_layer_sizes=(8,),
                    model_type="mlp",
                    include_image_features=True,
                    quality_threshold=0.5,
                )

            model = load_quality_proxy(output_dir / "quality_proxy.joblib")
            predictions = model.predict_quality(frame.head(2))

            self.assertEqual(summary["split"]["strategy"], "explicit_train_val_test")
            self.assertFalse(summary["split"]["image_overlaps"]["train_val"])
            self.assertFalse(summary["split"]["image_overlaps"]["train_test"])
            self.assertFalse(summary["split"]["image_overlaps"]["val_test"])
            self.assertIn("val", summary["metrics"])
            self.assertIn("test", summary["metrics"])
            self.assertIn("action_ranking", summary["metrics"]["test"])
            self.assertEqual(len(predictions), 2)

    def test_strict_training_rejects_overlapping_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_csv = root / "train.csv"
            val_csv = root / "val.csv"
            test_csv = root / "test.csv"
            frame = make_profile_frame()
            frame[frame["image_id"] == "P0001"].to_csv(train_csv, index=False)
            frame[frame["image_id"] == "P0001"].to_csv(val_csv, index=False)
            frame[frame["image_id"] == "P0003"].to_csv(test_csv, index=False)

            with self.assertRaisesRegex(ValueError, "overlap"):
                train_quality_proxy_strict(
                    train_csv,
                    val_csv,
                    test_csv,
                    root / "strict_proxy",
                    model_type="mlp",
                    max_iter=10,
                )


if __name__ == "__main__":
    unittest.main()
