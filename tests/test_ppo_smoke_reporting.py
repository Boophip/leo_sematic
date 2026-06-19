from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PpoSmokeReportingTests(unittest.TestCase):
    def test_multi_scenario_aggregate_ranks_by_canonical_qoe(self) -> None:
        runner = _load_runner_module()
        rows = [
            _metric_row("default", "Proposed-RL", 1.0, delay_ms=40.0),
            _metric_row("default", "Semantic-Greedy", 2.5, delay_ms=20.0),
            _metric_row("low_snr", "Proposed-RL", 3.0, delay_ms=25.0),
            _metric_row("low_snr", "Semantic-Greedy", 1.0, delay_ms=30.0),
        ]

        aggregate = runner._build_multi_scenario_aggregate(rows)

        default_summary = aggregate["scenario_summaries"][0]
        low_snr_summary = aggregate["scenario_summaries"][1]
        self.assertEqual(default_summary["best_policy"], "Semantic-Greedy")
        self.assertEqual(default_summary["proposed_rl_rank"], 2)
        self.assertAlmostEqual(default_summary["proposed_rl_gap_to_best_qoe"], -1.5)
        self.assertEqual(low_snr_summary["best_policy"], "Proposed-RL")
        self.assertEqual(low_snr_summary["proposed_rl_rank"], 1)

        proposed = next(row for row in aggregate["policy_aggregate"] if row["policy"] == "Proposed-RL")
        self.assertEqual(proposed["scenario_count"], 2)
        self.assertEqual(proposed["best_scenario_count"], 1)
        self.assertAlmostEqual(proposed["mean_qoe"], 2.0)
        self.assertAlmostEqual(proposed["mean_rank"], 1.5)

    def test_report_table_formats_string_and_rank_columns(self) -> None:
        runner = _load_runner_module()
        table = runner._markdown_table_with_columns(
            [
                {
                    "scenario": "default",
                    "best_policy": "Semantic-Greedy",
                    "best_qoe": 2.5,
                    "proposed_rl_qoe": 1.0,
                    "proposed_rl_rank": 2,
                    "proposed_rl_gap_to_best_qoe": -1.5,
                }
            ],
            runner.SCENARIO_SUMMARY_COLUMNS,
        )

        self.assertIn("Semantic-Greedy", table)
        self.assertIn("| default | Semantic-Greedy | 2.5000 | 1.0000 | 2 | -1.5000 |", table)


def _metric_row(scenario: str, policy: str, qoe: float, *, delay_ms: float) -> dict[str, object]:
    return {
        "scenario": scenario,
        "policy": policy,
        "qoe_total": qoe,
        "success_rate": 0.5,
        "semantic_success_rate": 0.5,
        "mean_delay_ms": delay_ms,
        "total_energy_j": 0.1,
        "total_aosi_cost": 1.0,
    }


def _load_runner_module():
    # The script filename starts with a number, so tests load it by path.
    script_path = ROOT / "scripts" / "17_train_ppo_smoke.py"
    spec = importlib.util.spec_from_file_location("train_ppo_smoke", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
