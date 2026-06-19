from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

from src.simulation.aosi import update_aosi_cell_soft
from src.simulation.episode import (
    ActionProfileTable,
    AosiGreedyPolicy,
    BestSnrPolicy,
    LocalAdaptivePolicy,
    NoAosiGreedyPolicy,
    PolicyContext,
    SatelliteNodeConfig,
    SimulationConfig,
    build_scenario,
    default_node_configs,
    default_policies,
    legal_actions_for_roi,
    run_policy_episode,
)
from src.simulation.link import LinkState


ROOT = Path(__file__).resolve().parents[1]


def _profile_frame() -> pd.DataFrame:
    """Build a tiny ROI x exit x compression table with the production schema."""

    rows = []
    for roi_id, semantic_value, grid_id in (
        ("roi_a", 0.8, 3),
        ("roi_b", 0.4, 4),
    ):
        for compression_level, compressed_bytes in (("local", 0), ("beta_1", 1000)):
            for exit_level, quality, inference_ms in (
                (1, 0.3, 10.0),
                (2, 0.8, 20.0),
            ):
                rows.append(
                    {
                        "roi_id": roi_id,
                        "image_id": "image_0",
                        "predicted_class": "ship",
                        "class_id": 1,
                        "detector_confidence": 0.9,
                        "area_ratio": 0.01,
                        "semantic_value": semantic_value,
                        "grid_id": grid_id,
                        "class_priority": 1.0,
                        "exit_level": exit_level,
                        "compression_level": compression_level,
                        "compressed_bytes": compressed_bytes,
                        "encode_ms": 0.0 if compression_level == "local" else 1.0,
                        "decode_ms": 0.0 if compression_level == "local" else 1.0,
                        "inference_ms": inference_ms,
                        "task_quality": quality,
                        "output_width": 32,
                        "output_height": 32,
                    }
                )
    return pd.DataFrame(rows)


class AosiSoftResetTests(unittest.TestCase):
    def test_task_achievement_soft_reset_lowers_age_cost(self) -> None:
        stale = update_aosi_cell_soft(
            previous_value=0.2,
            current_value=0.8,
            previous_age_s=5.0,
            delta_t_s=1.0,
            achievement_rate=0.0,
        )
        partial = update_aosi_cell_soft(
            previous_value=0.2,
            current_value=0.8,
            previous_age_s=5.0,
            delta_t_s=1.0,
            achievement_rate=0.5,
        )
        refreshed = update_aosi_cell_soft(
            previous_value=0.2,
            current_value=0.8,
            previous_age_s=5.0,
            delta_t_s=1.0,
            achievement_rate=1.0,
        )

        self.assertGreater(stale.age_s, partial.age_s)
        self.assertGreater(partial.age_s, refreshed.age_s)
        self.assertGreater(stale.cost, refreshed.cost)


class DeterministicEpisodeTests(unittest.TestCase):
    def test_invisible_link_is_not_a_legal_offload_action(self) -> None:
        table = ActionProfileTable.from_frame(_profile_frame())
        actions = legal_actions_for_roi(
            table,
            "roi_a",
            {"sat_1": LinkState(visible=False, snr_db=20.0, bandwidth_hz=10e6)},
        )

        self.assertTrue(any(action.kind == "local" for action in actions))
        self.assertFalse(any(action.kind == "offload" for action in actions))

    def test_local_adaptive_episode_uses_quality_threshold(self) -> None:
        table = ActionProfileTable.from_frame(_profile_frame())
        result = run_policy_episode(
            table,
            LocalAdaptivePolicy(),
            config=SimulationConfig(rois_per_slot=2, deadline_ms=100.0, quality_threshold=0.5),
            node_configs=(
                SatelliteNodeConfig(name="source", frequency_cycles_per_s=1e9),
                SatelliteNodeConfig(name="sat_1", frequency_cycles_per_s=1e9),
            ),
            link_trace={0: {"sat_1": LinkState(visible=True, snr_db=18.0, bandwidth_hz=10e6)}},
        )

        self.assertEqual(result.metrics["processed_count"], 2)
        self.assertEqual(result.metrics["success_count"], 2)
        self.assertTrue(all(row.action_label == "local:e2" for row in result.decisions))

    def test_busy_offload_node_increases_best_snr_delay(self) -> None:
        table = ActionProfileTable.from_frame(_profile_frame())
        config = SimulationConfig(rois_per_slot=2, deadline_ms=10_000.0, quality_threshold=0.5)
        links = {0: {"sat_1": LinkState(visible=True, snr_db=18.0, bandwidth_hz=10e6)}}
        idle = run_policy_episode(
            table,
            BestSnrPolicy(),
            config=config,
            node_configs=(
                SatelliteNodeConfig(name="source", frequency_cycles_per_s=1e9),
                SatelliteNodeConfig(name="sat_1", frequency_cycles_per_s=1e9),
            ),
            link_trace=links,
        )
        busy = run_policy_episode(
            table,
            BestSnrPolicy(),
            config=config,
            node_configs=(
                SatelliteNodeConfig(name="source", frequency_cycles_per_s=1e9),
                SatelliteNodeConfig(name="sat_1", frequency_cycles_per_s=1e9, initial_queue_cycles=5e9),
            ),
            link_trace=links,
        )

        self.assertEqual(idle.metrics["offload_count"], 2)
        self.assertGreater(busy.metrics["mean_delay_ms"], idle.metrics["mean_delay_ms"])

    def test_no_aosi_policy_does_not_depend_on_grid_age(self) -> None:
        table = ActionProfileTable.from_frame(_profile_frame())
        links = {"sat_1": LinkState(visible=True, snr_db=18.0, bandwidth_hz=10e6)}
        actions = legal_actions_for_roi(table, "roi_a", links)
        policy = NoAosiGreedyPolicy()
        base_context = PolicyContext(
            slot_index=0,
            links=links,
            node_queues={},
            grid_ages={3: 0.0},
            config=SimulationConfig(deadline_ms=100.0),
            profiles=table,
        )
        stale_context = PolicyContext(
            slot_index=0,
            links=links,
            node_queues={},
            grid_ages={3: 100.0},
            config=SimulationConfig(deadline_ms=100.0),
            profiles=table,
        )

        self.assertEqual(
            policy.choose(table.roi("roi_a"), actions, base_context).label,
            policy.choose(table.roi("roi_a"), actions, stale_context).label,
        )

    def test_aosi_policy_changes_action_when_grid_is_stale(self) -> None:
        table = ActionProfileTable.from_frame(_profile_frame())
        links = {"sat_1": LinkState(visible=True, snr_db=8.0, bandwidth_hz=0.5e6, propagation_delay_ms=20.0)}
        actions = legal_actions_for_roi(table, "roi_a", links)
        policy = AosiGreedyPolicy()
        fresh_context = PolicyContext(
            slot_index=0,
            links=links,
            node_queues={},
            grid_ages={3: 0.0},
            config=SimulationConfig(deadline_ms=100.0),
            profiles=table,
        )
        stale_context = PolicyContext(
            slot_index=0,
            links=links,
            node_queues={},
            grid_ages={3: 20.0},
            config=SimulationConfig(deadline_ms=100.0),
            profiles=table,
        )

        fresh_action = policy.choose(table.roi("roi_a"), actions, fresh_context)
        stale_action = policy.choose(table.roi("roi_a"), actions, stale_context)

        self.assertEqual(fresh_action.kind, "offload")
        self.assertEqual(stale_action.kind, "local")

    def test_low_snr_scenario_increases_best_snr_communication_delay(self) -> None:
        table = ActionProfileTable.from_frame(_profile_frame())
        config = SimulationConfig(rois_per_slot=2, deadline_ms=10_000.0, quality_threshold=0.5)
        default_bundle = build_scenario(
            "default",
            base_config=config,
            node_configs=default_node_configs(),
            slot_count=1,
            bandwidth_hz=10e6,
        )
        low_snr_bundle = build_scenario(
            "low_snr",
            base_config=config,
            node_configs=default_node_configs(),
            slot_count=1,
            bandwidth_hz=10e6,
        )
        default_run = run_policy_episode(
            table,
            BestSnrPolicy(),
            config=default_bundle.config,
            node_configs=default_bundle.node_configs,
            link_trace=default_bundle.link_trace,
        )
        low_snr_run = run_policy_episode(
            table,
            BestSnrPolicy(),
            config=low_snr_bundle.config,
            node_configs=low_snr_bundle.node_configs,
            link_trace=low_snr_bundle.link_trace,
        )

        self.assertEqual(default_run.metrics["offload_count"], 2)
        self.assertEqual(low_snr_run.metrics["offload_count"], 2)
        self.assertGreater(
            _mean(row.communication_delay_ms for row in low_snr_run.decisions if row.action_kind == "offload"),
            _mean(row.communication_delay_ms for row in default_run.decisions if row.action_kind == "offload"),
        )

    def test_compute_congested_scenario_increases_offload_delay(self) -> None:
        table = ActionProfileTable.from_frame(_profile_frame())
        config = SimulationConfig(rois_per_slot=2, deadline_ms=10_000.0, quality_threshold=0.5)
        default_bundle = build_scenario(
            "default",
            base_config=config,
            node_configs=default_node_configs(),
            slot_count=1,
            bandwidth_hz=10e6,
        )
        congested_bundle = build_scenario(
            "compute_congested",
            base_config=config,
            node_configs=default_node_configs(),
            slot_count=1,
            bandwidth_hz=10e6,
        )
        default_run = run_policy_episode(
            table,
            BestSnrPolicy(),
            config=default_bundle.config,
            node_configs=default_bundle.node_configs,
            link_trace=default_bundle.link_trace,
        )
        congested_run = run_policy_episode(
            table,
            BestSnrPolicy(),
            config=congested_bundle.config,
            node_configs=congested_bundle.node_configs,
            link_trace=congested_bundle.link_trace,
        )

        self.assertGreater(congested_run.metrics["mean_delay_ms"], default_run.metrics["mean_delay_ms"])

    def test_yaml_config_loads_default_core_fields(self) -> None:
        runner = _load_runner_module()
        args = argparse.Namespace(
            config=ROOT / "configs" / "satellite_env.yaml",
            profile_csv=None,
            quality_proxy=None,
            output_dir=None,
            limit_rois=None,
            rois_per_slot=None,
            deadline_ms=None,
            quality_threshold=None,
            seed=None,
            use_oracle_quality=False,
            scenario=None,
            all_scenarios=False,
            exist_ok=False,
        )

        settings = runner._resolve_settings(args)

        self.assertTrue(settings.config_loaded)
        self.assertEqual(settings.base_config.rois_per_slot, SimulationConfig().rois_per_slot)
        self.assertEqual(settings.base_config.deadline_ms, SimulationConfig().deadline_ms)
        self.assertEqual(settings.base_config.quality_threshold, SimulationConfig().quality_threshold)
        self.assertEqual(settings.bandwidth_hz, 20e6)
        self.assertEqual(settings.policy_names, tuple(policy.name for policy in default_policies(seed=42)))


def _mean(values) -> float:
    values = list(values)
    return 0.0 if not values else sum(values) / len(values)


def _load_runner_module():
    script_path = ROOT / "scripts" / "16_run_deterministic_simulation.py"
    spec = importlib.util.spec_from_file_location("run_deterministic_simulation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
