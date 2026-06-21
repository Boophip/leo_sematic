from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.envs import LeoSchedulingEnv, build_fixed_action_space
from src.simulation.episode import (
    ActionProfileTable,
    SatelliteNodeConfig,
    SimulationConfig,
    build_slot_summary,
)
from src.simulation.link import LinkState


def _profile_frame() -> pd.DataFrame:
    """Build a compact production-schema profile table for RL environment tests."""

    rows = []
    for roi_id, semantic_value, grid_id in (
        ("roi_a", 0.8, 3),
        ("roi_b", 0.6, 4),
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


def _env(
    *,
    visible: bool = True,
    rois_per_slot: int = 2,
    deadline_ms: float = 100.0,
) -> LeoSchedulingEnv:
    table = ActionProfileTable.from_frame(_profile_frame())
    return LeoSchedulingEnv(
        table,
        config=SimulationConfig(
            rois_per_slot=rois_per_slot,
            deadline_ms=deadline_ms,
            quality_threshold=0.5,
            illegal_penalty=1.0,
        ),
        node_configs=(
            SatelliteNodeConfig(name="source", frequency_cycles_per_s=1e9),
            SatelliteNodeConfig(name="sat_1", frequency_cycles_per_s=1e9),
        ),
        link_trace={
            0: {"sat_1": LinkState(visible=visible, snr_db=18.0, bandwidth_hz=10e6)},
            1: {"sat_1": LinkState(visible=visible, snr_db=18.0, bandwidth_hz=10e6)},
        },
    )


class FixedActionSpaceTests(unittest.TestCase):
    def test_action_map_covers_drop_local_and_offload(self) -> None:
        actions = build_fixed_action_space(("sat_1",))

        self.assertEqual(actions[0].kind, "drop")
        self.assertTrue(any(action.label == "local:e1" for action in actions))
        self.assertTrue(any(action.label == "local:e4" for action in actions))
        self.assertTrue(any(action.label == "sat_1:e1:beta_0" for action in actions))
        self.assertTrue(any(action.label == "sat_1:e4:beta_3" for action in actions))


class LeoSchedulingEnvTests(unittest.TestCase):
    def test_reset_and_step_follow_gymnasium_api(self) -> None:
        env = _env()
        obs, info = env.reset(seed=123)

        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(info["roi_id"], "roi_a")

        next_obs, reward, terminated, truncated, step_info = env.step(
            env.action_index(kind="drop")
        )

        self.assertIsInstance(next_obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["action_kind"], "drop")

    def test_invisible_offload_is_illegal_with_penalty(self) -> None:
        env = _env(visible=False)
        env.reset()
        _, reward, _, _, info = env.step(
            env.action_index(
                kind="offload",
                target_node="sat_1",
                exit_level=2,
                compression_level="beta_1",
            )
        )

        self.assertTrue(info["illegal"])
        self.assertLessEqual(reward, -1.0)
        self.assertEqual(info["success"], False)

    def test_local_and_offload_actions_affect_costs_and_queue_arrivals(self) -> None:
        local_env = _env()
        local_env.reset()
        _, _, _, _, local_info = local_env.step(
            local_env.action_index(kind="local", exit_level=2, compression_level="local")
        )

        self.assertEqual(local_info["action_kind"], "local")
        self.assertGreater(local_info["total_delay_ms"], 0.0)
        self.assertEqual(local_info["communication_delay_ms"], 0.0)
        self.assertGreater(local_info["total_energy_j"], 0.0)
        self.assertGreater(local_env.pending_cycles["source"], 0.0)

        offload_env = _env()
        offload_env.reset()
        _, _, _, _, offload_info = offload_env.step(
            offload_env.action_index(
                kind="offload",
                target_node="sat_1",
                exit_level=2,
                compression_level="beta_1",
            )
        )

        self.assertEqual(offload_info["action_kind"], "offload")
        self.assertGreater(offload_info["communication_delay_ms"], 0.0)
        self.assertEqual(offload_info["compressed_bytes"], 1000)
        self.assertGreater(offload_env.pending_cycles["sat_1"], 0.0)

    def test_slot_boundary_updates_queue_and_aosi(self) -> None:
        env = _env(rois_per_slot=1)
        env.reset()
        _, _, _, _, info = env.step(
            env.action_index(kind="local", exit_level=2, compression_level="local")
        )

        self.assertTrue(info["slot_ended"])
        self.assertEqual(len(env.slots), 1)
        self.assertIn(3, env.grid_ages)
        self.assertGreater(env.node_states["source"].queue_cycles, 0.0)
        self.assertIn("quality_virtual_queue", info)
        self.assertIn("delay_virtual_queue_ms", info)

    def test_virtual_queues_reset_and_follow_slot_constraints(self) -> None:
        env = _env(rois_per_slot=1)
        env.reset()

        self.assertEqual(env.quality_virtual_queue, 0.0)
        self.assertEqual(env.delay_virtual_queue_ms, 0.0)

        env.step(env.action_index(kind="local", exit_level=1, compression_level="local"))
        self.assertGreater(env.quality_virtual_queue, 0.0)

        env.step(env.action_index(kind="local", exit_level=2, compression_level="local"))
        self.assertGreaterEqual(env.quality_virtual_queue, 0.0)
        self.assertLess(env.quality_virtual_queue, 0.5)

    def test_delay_virtual_queue_only_increases_on_deadline_excess(self) -> None:
        relaxed_env = _env(rois_per_slot=1, deadline_ms=100.0)
        relaxed_env.reset()
        relaxed_env.step(relaxed_env.action_index(kind="local", exit_level=1, compression_level="local"))
        self.assertEqual(relaxed_env.delay_virtual_queue_ms, 0.0)

        tight_env = _env(rois_per_slot=1, deadline_ms=5.0)
        tight_env.reset()
        tight_env.step(tight_env.action_index(kind="local", exit_level=2, compression_level="local"))
        self.assertGreater(tight_env.delay_virtual_queue_ms, 0.0)

    def test_slot_summary_uses_canonical_qoe_builder(self) -> None:
        env = _env(rois_per_slot=2)
        env.reset()
        env.step(env.action_index(kind="local", exit_level=2, compression_level="local"))
        _, _, terminated, _, info = env.step(
            env.action_index(kind="local", exit_level=2, compression_level="local")
        )

        self.assertTrue(terminated)
        self.assertTrue(info["slot_ended"])
        expected = build_slot_summary(
            "Proposed-RL",
            0,
            env.decisions,
            aosi_cost=env.slots[0].aosi_cost,
            config=env.config,
        )
        self.assertAlmostEqual(env.slots[0].reward, expected.reward)
        self.assertAlmostEqual(env.metrics()["qoe_total"], expected.reward)

    def test_training_reward_is_separate_from_reported_qoe(self) -> None:
        env = _env(rois_per_slot=2)
        env.reset()
        env.step(env.action_index(kind="local", exit_level=2, compression_level="local"))
        env.step(env.action_index(kind="local", exit_level=2, compression_level="local"))

        self.assertNotAlmostEqual(env.training_reward_total, env.metrics()["qoe_total"])

    def test_canonical_qoe_applies_constraint_penalties(self) -> None:
        env = _env(visible=False, rois_per_slot=1)
        env.reset()
        _, _, _, _, info = env.step(
            env.action_index(
                kind="offload",
                target_node="sat_1",
                exit_level=2,
                compression_level="beta_1",
            )
        )

        self.assertTrue(info["illegal"])
        self.assertLessEqual(env.metrics()["qoe_total"], -env.config.illegal_penalty)

    def test_observation_feature_names_exclude_leakage_fields(self) -> None:
        env = _env()
        forbidden = {"task_quality", "quality_label", "exit_confidence", "predicted_class_id"}

        joined = " ".join(env.observation_feature_names)
        self.assertIn("quality_virtual_queue_norm", joined)
        self.assertIn("delay_virtual_queue_norm", joined)
        for field in forbidden:
            self.assertNotIn(field, joined)


if __name__ == "__main__":
    unittest.main()
