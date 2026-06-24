from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PpoFormalExperimentTests(unittest.TestCase):
    def test_default_candidate_mode_is_feasible_topk_main_method(self) -> None:
        runner = _load_runner_module()
        original_argv = sys.argv
        try:
            sys.argv = ["18_run_ppo_formal_experiment.py"]
            args = runner.parse_args()
        finally:
            sys.argv = original_argv

        self.assertEqual(args.candidate_mode, "feasible-topk")
        self.assertEqual(args.candidate_top_k, 12)

    def test_formal_aggregate_computes_policy_mean_std_and_scenario_summary(self) -> None:
        runner = _load_runner_module()
        seed_results = [
            {
                "seed": 42,
                "summary": _seed_summary(
                    [
                        _metric_row("default", "Proposed-RL", 2.0, rank=1, is_best=True),
                        _metric_row("default", "Best-SNR", 1.0, rank=2, is_best=False),
                        _metric_row("low_snr", "Proposed-RL", 0.0, rank=2, is_best=False),
                        _metric_row("low_snr", "Best-SNR", 3.0, rank=1, is_best=True),
                    ],
                    [
                        _winner_row("default", "Proposed-RL", 2.0, 1, 0.0),
                        _winner_row("low_snr", "Best-SNR", 0.0, 2, -3.0),
                    ],
                ),
            },
            {
                "seed": 43,
                "summary": _seed_summary(
                    [
                        _metric_row("default", "Proposed-RL", 4.0, rank=1, is_best=True),
                        _metric_row("default", "Best-SNR", 1.0, rank=2, is_best=False),
                        _metric_row("low_snr", "Proposed-RL", 2.0, rank=1, is_best=True),
                        _metric_row("low_snr", "Best-SNR", 1.0, rank=2, is_best=False),
                    ],
                    [
                        _winner_row("default", "Proposed-RL", 4.0, 1, 0.0),
                        _winner_row("low_snr", "Proposed-RL", 2.0, 1, 0.0),
                    ],
                ),
            },
        ]

        aggregate = runner._build_formal_aggregate(seed_results)

        proposed = next(row for row in aggregate["policy_formal_aggregate"] if row["policy"] == "Proposed-RL")
        self.assertEqual(proposed["sample_count"], 4)
        self.assertEqual(proposed["seed_count"], 2)
        self.assertEqual(proposed["scenario_count"], 2)
        self.assertEqual(proposed["best_count"], 3)
        self.assertAlmostEqual(proposed["mean_qoe"], 2.0)
        self.assertAlmostEqual(proposed["std_qoe"], 2.0 ** 0.5)

        low_snr = next(row for row in aggregate["scenario_formal_summary"] if row["scenario"] == "low_snr")
        self.assertEqual(low_snr["best_policy"], "Best-SNR")
        self.assertAlmostEqual(low_snr["best_mean_qoe"], 2.0)
        self.assertAlmostEqual(low_snr["proposed_rl_mean_qoe"], 1.0)
        self.assertAlmostEqual(low_snr["proposed_rl_mean_gap_to_best"], -1.0)

    def test_seed_command_passes_all_scenarios_and_seed(self) -> None:
        runner = _load_runner_module()
        settings = runner.FormalExperimentSettings(
            seeds=(7,),
            output_dir=ROOT / "outputs" / "rl" / "tmp_formal_test",
            limit_rois=64,
            total_timesteps=128,
            python_executable=Path(sys.executable),
            config=None,
            profile_csv=None,
            quality_proxy=None,
            rois_per_slot=None,
            deadline_ms=None,
            quality_threshold=None,
            ppo_n_steps=64,
            ppo_batch_size=32,
            ppo_n_epochs=3,
            ppo_learning_rate=1e-4,
            ppo_gamma=0.95,
            ppo_ent_coef=0.01,
            eval_frequency=128,
            checkpoint_frequency=256,
            select_best_checkpoint=True,
            best_checkpoint_min_success_rate=0.5,
            reward_quality_deficit_weight=0.3,
            reward_delay_excess_weight=0.5,
            reward_virtual_queue_weight=0.1,
            candidate_mode="legal",
            candidate_top_k=12,
            exist_ok=True,
            dry_run=False,
            skip_plots=True,
        )

        command = runner._build_seed_command(settings, 7)

        self.assertIn("--all-scenarios", command)
        self.assertIn("--seed", command)
        self.assertIn("7", command)
        self.assertIn("--exist-ok", command)
        self.assertIn("--ppo-ent-coef", command)
        self.assertIn("0.01", command)
        self.assertIn("--eval-frequency", command)
        self.assertIn("128", command)
        self.assertIn("--checkpoint-frequency", command)
        self.assertIn("256", command)
        self.assertIn("--select-best-checkpoint", command)
        self.assertIn("--best-checkpoint-min-success-rate", command)
        self.assertIn("0.5", command)
        self.assertIn("--reward-quality-deficit-weight", command)
        self.assertIn("0.3", command)
        self.assertIn("--reward-delay-excess-weight", command)
        self.assertIn("0.5", command)
        self.assertIn("--reward-virtual-queue-weight", command)
        self.assertIn("0.1", command)
        self.assertIn("--candidate-mode", command)
        self.assertIn("legal", command)
        self.assertIn("--candidate-top-k", command)
        self.assertIn("12", command)

    def test_runtime_estimate_scales_with_seed_and_scenario_count(self) -> None:
        runner = _load_runner_module()
        settings = runner.FormalExperimentSettings(
            seeds=(42, 43, 44),
            output_dir=ROOT / "outputs" / "rl" / "tmp_formal_test",
            limit_rois=512,
            total_timesteps=5000,
            python_executable=Path(sys.executable),
            config=None,
            profile_csv=None,
            quality_proxy=None,
            rois_per_slot=None,
            deadline_ms=None,
            quality_threshold=None,
            ppo_n_steps=None,
            ppo_batch_size=None,
            ppo_n_epochs=None,
            ppo_learning_rate=None,
            ppo_gamma=None,
            ppo_ent_coef=None,
            eval_frequency=0,
            checkpoint_frequency=0,
            select_best_checkpoint=False,
            best_checkpoint_min_success_rate=0.0,
            reward_quality_deficit_weight=0.0,
            reward_delay_excess_weight=0.0,
            reward_virtual_queue_weight=0.0,
            candidate_mode="fixed",
            candidate_top_k=12,
            exist_ok=True,
            dry_run=False,
            skip_plots=True,
        )

        estimate = runner._runtime_estimate(settings)

        self.assertIn("24-96 minutes", estimate)


def _seed_summary(
    policy_metrics: list[dict[str, object]],
    scenario_summaries: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "policy_metrics": policy_metrics,
        "scenario_summaries": scenario_summaries,
    }


def _metric_row(
    scenario: str,
    policy: str,
    qoe: float,
    *,
    rank: int,
    is_best: bool,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "policy": policy,
        "qoe_total": qoe,
        "scenario_rank": rank,
        "is_best_policy": is_best,
        "qoe_gap_to_best": 0.0 if is_best else qoe - 3.0,
        "success_rate": 0.5,
        "semantic_success_rate": 0.5,
        "mean_delay_ms": 10.0,
        "total_energy_j": 0.1,
        "total_aosi_cost": 1.0,
    }


def _winner_row(
    scenario: str,
    best_policy: str,
    proposed_qoe: float,
    proposed_rank: int,
    proposed_gap: float,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "best_policy": best_policy,
        "best_qoe": proposed_qoe - proposed_gap,
        "proposed_rl_qoe": proposed_qoe,
        "proposed_rl_rank": proposed_rank,
        "proposed_rl_gap_to_best_qoe": proposed_gap,
    }


def _load_runner_module():
    script_path = ROOT / "scripts" / "18_run_ppo_formal_experiment.py"
    spec = importlib.util.spec_from_file_location("run_ppo_formal_experiment", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
