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

    def test_ppo_hyperparameter_overrides_are_applied(self) -> None:
        runner = _load_runner_module()

        hyperparameters = runner._ppo_hyperparameters(
            1024,
            n_steps=64,
            batch_size=32,
            n_epochs=3,
            learning_rate=1e-4,
            gamma=0.95,
            ent_coef=0.01,
        )

        self.assertEqual(hyperparameters["n_steps"], 64)
        self.assertEqual(hyperparameters["batch_size"], 32)
        self.assertEqual(hyperparameters["n_epochs"], 3)
        self.assertEqual(hyperparameters["learning_rate"], 1e-4)
        self.assertEqual(hyperparameters["gamma"], 0.95)
        self.assertEqual(hyperparameters["ent_coef"], 0.01)

    def test_select_best_checkpoint_requires_eval_frequency(self) -> None:
        runner = _load_runner_module()
        settings = runner.PpoSmokeSettings(
            config_path=None,
            config_loaded=False,
            profile_csv=Path(__file__),
            quality_proxy=Path(__file__),
            output_dir=ROOT / "outputs" / "rl" / "tmp_smoke_test",
            limit_rois=64,
            total_timesteps=128,
            seed=42,
            scenario="default",
            all_scenarios=False,
            eval_only=False,
            model_path=None,
            exist_ok=True,
            ppo_n_steps=None,
            ppo_batch_size=None,
            ppo_n_epochs=None,
            ppo_learning_rate=None,
            ppo_gamma=None,
            ppo_ent_coef=None,
            eval_frequency=0,
            checkpoint_frequency=0,
            select_best_checkpoint=True,
            best_checkpoint_min_success_rate=0.0,
            reward_quality_deficit_weight=0.0,
            reward_delay_excess_weight=0.0,
            reward_virtual_queue_weight=0.0,
            bandwidth_hz=runner.DEFAULT_LINK_BANDWIDTH_HZ,
            base_config=runner.SimulationConfig(),
            node_configs=runner.default_node_configs(),
            policy_names=(),
        )

        with self.assertRaisesRegex(ValueError, "--select-best-checkpoint"):
            runner._validate_settings(settings)


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
