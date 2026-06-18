"""Validate deterministic link, queue, energy, and AoSI causality."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulation.aosi import grid_semantic_value, update_aosi_cell  # noqa: E402
from src.simulation.compute import ComputeNode, evolve_compute_queue  # noqa: E402
from src.simulation.deterministic import (  # noqa: E402
    TaskCostProfile,
    evaluate_local_task,
    evaluate_offload_task,
)
from src.simulation.link import IllegalActionError, LinkState  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=(
            ROOT
            / "data"
            / "profiling"
            / "dota_demo_1024_o200"
            / "test_smoke"
            / "roi_profile_smoke.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "simulation" / "smoke",
    )
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _prepare_output_dir(path: Path, *, exist_ok: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not exist_ok:
            raise FileExistsError(f"output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _first_profile_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return row
    raise ValueError(f"profile CSV has no rows: {path}")


def _task_profile_from_row(row: dict[str, str]) -> TaskCostProfile:
    return TaskCostProfile(
        compressed_bytes=int(float(row["compressed_bytes"])),
        encode_ms=float(row["encode_ms"]),
        decode_ms=float(row["decode_ms"]),
        inference_ms=float(row["inference_ms"]),
        result_bytes=64,
    )


def main() -> int:
    args = parse_args()
    _prepare_output_dir(args.output_dir, exist_ok=args.exist_ok)
    row = _first_profile_row(args.profile_csv)
    profile = _task_profile_from_row(row)

    idle_node = ComputeNode(frequency_cycles_per_s=2e9, queue_cycles=0.0)
    busy_node = ComputeNode(frequency_cycles_per_s=2e9, queue_cycles=4e9)
    good_link = LinkState(
        visible=True,
        snr_db=18.0,
        bandwidth_hz=20e6,
        propagation_delay_ms=2.0,
    )
    weak_link = LinkState(
        visible=True,
        snr_db=2.0,
        bandwidth_hz=20e6,
        propagation_delay_ms=2.0,
    )
    invisible_link = LinkState(
        visible=False,
        snr_db=30.0,
        bandwidth_hz=20e6,
        propagation_delay_ms=2.0,
    )

    local_idle = evaluate_local_task(profile, idle_node)
    offload_good_idle = evaluate_offload_task(profile, good_link, idle_node)
    offload_weak_idle = evaluate_offload_task(profile, weak_link, idle_node)
    offload_good_busy = evaluate_offload_task(profile, good_link, busy_node)
    invisible_is_illegal = False
    try:
        evaluate_offload_task(profile, invisible_link, idle_node)
    except IllegalActionError:
        invisible_is_illegal = True

    next_queue_cycles = evolve_compute_queue(
        queue_cycles=busy_node.queue_cycles,
        frequency_cycles_per_s=busy_node.frequency_cycles_per_s,
        delta_t_s=1.0,
        arrival_cycles=profile.inference_ms / 1000.0 * busy_node.frequency_cycles_per_s,
    )
    semantic_value = grid_semantic_value([0.5, 0.2])
    aged_cell = update_aosi_cell(
        previous_value=0.2,
        current_value=semantic_value,
        previous_age_s=3.0,
        delta_t_s=1.0,
        updated=False,
    )
    refreshed_cell = update_aosi_cell(
        previous_value=0.2,
        current_value=semantic_value,
        previous_age_s=3.0,
        delta_t_s=1.0,
        updated=True,
    )

    summary = {
        "purpose": "Deterministic simulation smoke validation; not an RL run.",
        "profile_csv": str(args.profile_csv.resolve()),
        "profile_row": {
            "roi_id": row.get("roi_id", ""),
            "exit_level": row.get("exit_level", ""),
            "compression_level": row.get("compression_level", ""),
        },
        "delays_ms": {
            "local_idle_total": local_idle.total_delay_ms,
            "offload_good_idle_total": offload_good_idle.total_delay_ms,
            "offload_weak_idle_total": offload_weak_idle.total_delay_ms,
            "offload_good_busy_total": offload_good_busy.total_delay_ms,
        },
        "energies_j": {
            "local_idle_total": local_idle.total_energy_j,
            "offload_good_idle_total": offload_good_idle.total_energy_j,
        },
        "queue": {
            "busy_initial_cycles": busy_node.queue_cycles,
            "next_queue_cycles": next_queue_cycles,
        },
        "aosi": {
            "semantic_value": semantic_value,
            "aged_cost": aged_cell.cost,
            "refreshed_cost": refreshed_cell.cost,
        },
        "causality_checks": {
            "weak_link_slower_than_good_link": (
                offload_weak_idle.communication_delay_ms
                > offload_good_idle.communication_delay_ms
            ),
            "busy_queue_slower_than_idle_queue": (
                offload_good_busy.total_delay_ms > offload_good_idle.total_delay_ms
            ),
            "invisible_link_illegal": invisible_is_illegal,
            "refreshed_aosi_lower_than_aged": refreshed_cell.cost < aged_cell.cost,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
