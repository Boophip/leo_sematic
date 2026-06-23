from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "19_compare_quality_proxy_models.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("compare_quality_proxy_models", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class QualityProxyModelComparisonTests(unittest.TestCase):
    def test_choose_primary_model_prefers_high_value_mae_then_r2_then_f1(self) -> None:
        module = load_script_module()
        rows = [
            {
                "model_type": "mlp",
                "test_r2": 0.90,
                "test_threshold_f1": 0.90,
                "test_high_value_mae": 0.20,
                "model_size_bytes": 100_000,
            },
            {
                "model_type": "hist_gbdt",
                "test_r2": 0.70,
                "test_threshold_f1": 0.70,
                "test_high_value_mae": 0.10,
                "model_size_bytes": 100_000,
            },
            {
                "model_type": "extra_trees",
                "test_r2": 0.80,
                "test_threshold_f1": 0.85,
                "test_high_value_mae": 0.10,
                "model_size_bytes": 100_000,
            },
        ]

        selected = module.choose_primary_model(rows)

        self.assertEqual(selected["model_type"], "extra_trees")

    def test_choose_primary_model_respects_lightweight_size_cap(self) -> None:
        module = load_script_module()
        rows = [
            {
                "model_type": "mlp",
                "test_r2": 0.64,
                "test_threshold_f1": 0.86,
                "test_high_value_mae": 0.13,
                "model_size_bytes": 600_000,
            },
            {
                "model_type": "extra_trees",
                "test_r2": 0.66,
                "test_threshold_f1": 0.87,
                "test_high_value_mae": 0.12,
                "model_size_bytes": 800 * 1024 * 1024,
            },
        ]

        selected = module.choose_primary_model(rows, max_primary_model_mb=50.0)

        self.assertEqual(selected["model_type"], "mlp")

    def test_choose_primary_model_falls_back_when_all_models_exceed_size_cap(self) -> None:
        module = load_script_module()
        rows = [
            {
                "model_type": "random_forest",
                "test_r2": 0.61,
                "test_threshold_f1": 0.84,
                "test_high_value_mae": 0.15,
                "model_size_bytes": 600 * 1024 * 1024,
            },
            {
                "model_type": "extra_trees",
                "test_r2": 0.66,
                "test_threshold_f1": 0.87,
                "test_high_value_mae": 0.12,
                "model_size_bytes": 800 * 1024 * 1024,
            },
        ]

        selected = module.choose_primary_model(rows, max_primary_model_mb=50.0)

        self.assertEqual(selected["model_type"], "extra_trees")


if __name__ == "__main__":
    unittest.main()
