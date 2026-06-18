from __future__ import annotations

import unittest

from src.simulation.aosi import (
    grid_semantic_value,
    semantic_change_factor,
    update_aosi_cell,
)
from src.simulation.compute import (
    ComputeNode,
    compute_delay_ms,
    compute_energy_joules,
    cycles_from_inference_ms,
    evolve_compute_queue,
    queue_wait_ms,
)
from src.simulation.deterministic import (
    TaskCostProfile,
    evaluate_local_task,
    evaluate_offload_task,
)
from src.simulation.link import (
    IllegalActionError,
    LinkState,
    offload_communication_delay_ms,
    spectral_efficiency_for_snr,
    transmission_delay_ms,
)


class LinkModelTests(unittest.TestCase):
    def test_snr_amc_lookup_is_stepwise(self) -> None:
        self.assertEqual(spectral_efficiency_for_snr(-1.0), 0.0)
        self.assertEqual(spectral_efficiency_for_snr(0.0), 0.5)
        self.assertEqual(spectral_efficiency_for_snr(7.9), 1.0)
        self.assertEqual(spectral_efficiency_for_snr(8.0), 2.0)

    def test_higher_snr_reduces_transmission_delay(self) -> None:
        weak = LinkState(visible=True, snr_db=2.0, bandwidth_hz=10e6)
        strong = LinkState(visible=True, snr_db=18.0, bandwidth_hz=10e6)

        self.assertGreater(
            transmission_delay_ms(10_000, weak),
            transmission_delay_ms(10_000, strong),
        )

    def test_invisible_link_is_illegal_not_large_delay(self) -> None:
        invisible = LinkState(visible=False, snr_db=30.0, bandwidth_hz=10e6)

        with self.assertRaises(IllegalActionError):
            offload_communication_delay_ms(10_000, invisible)


class ComputeModelTests(unittest.TestCase):
    def test_queue_wait_and_evolution_follow_project_formula(self) -> None:
        self.assertAlmostEqual(queue_wait_ms(2e9, 2e9), 1000.0)
        self.assertAlmostEqual(
            evolve_compute_queue(
                queue_cycles=5e9,
                frequency_cycles_per_s=2e9,
                delta_t_s=1.0,
                arrival_cycles=1e9,
            ),
            4e9,
        )

    def test_compute_delay_and_energy_use_fixed_frequency(self) -> None:
        node = ComputeNode(frequency_cycles_per_s=2e9, queue_cycles=1e9, energy_per_cycle_j=2e-9)
        task_cycles = cycles_from_inference_ms(250.0, node.frequency_cycles_per_s)

        self.assertAlmostEqual(task_cycles, 5e8)
        self.assertAlmostEqual(compute_delay_ms(task_cycles, node), 750.0)
        self.assertAlmostEqual(compute_energy_joules(task_cycles, node), 1.0)


class AosiModelTests(unittest.TestCase):
    def test_grid_value_uses_probability_or(self) -> None:
        self.assertAlmostEqual(grid_semantic_value([0.5, 0.2]), 1.0 - 0.5 * 0.8)
        self.assertEqual(grid_semantic_value([]), 0.0)

    def test_change_factor_and_refresh_update_age_cost(self) -> None:
        self.assertAlmostEqual(
            semantic_change_factor(0.7, 0.2, alpha=0.2),
            0.2 + 0.8 * 0.5,
        )

        aged = update_aosi_cell(
            previous_value=0.2,
            current_value=0.7,
            previous_age_s=3.0,
            delta_t_s=1.0,
            updated=False,
        )
        refreshed = update_aosi_cell(
            previous_value=0.2,
            current_value=0.7,
            previous_age_s=3.0,
            delta_t_s=1.0,
            updated=True,
        )

        self.assertEqual(aged.age_s, 4.0)
        self.assertEqual(refreshed.age_s, 0.0)
        self.assertGreater(aged.cost, refreshed.cost)


class DeterministicEvaluationTests(unittest.TestCase):
    def test_offload_causality_uses_link_and_queue_state(self) -> None:
        profile = TaskCostProfile(
            compressed_bytes=10_000,
            encode_ms=1.0,
            decode_ms=1.0,
            inference_ms=50.0,
        )
        idle_node = ComputeNode(frequency_cycles_per_s=2e9, queue_cycles=0.0)
        busy_node = ComputeNode(frequency_cycles_per_s=2e9, queue_cycles=2e9)
        weak = LinkState(visible=True, snr_db=2.0, bandwidth_hz=10e6, propagation_delay_ms=1.0)
        strong = LinkState(visible=True, snr_db=18.0, bandwidth_hz=10e6, propagation_delay_ms=1.0)

        weak_eval = evaluate_offload_task(profile, weak, idle_node)
        strong_idle_eval = evaluate_offload_task(profile, strong, idle_node)
        strong_busy_eval = evaluate_offload_task(profile, strong, busy_node)
        local_eval = evaluate_local_task(profile, idle_node)

        self.assertGreater(weak_eval.communication_delay_ms, strong_idle_eval.communication_delay_ms)
        self.assertGreater(strong_busy_eval.total_delay_ms, strong_idle_eval.total_delay_ms)
        self.assertEqual(local_eval.communication_delay_ms, 0.0)
        self.assertGreater(strong_idle_eval.communication_energy_j, 0.0)


if __name__ == "__main__":
    unittest.main()
