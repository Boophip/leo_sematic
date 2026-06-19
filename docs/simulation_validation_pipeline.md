# Deterministic Simulation Validation Pipeline

This stage adds the first deterministic satellite-simulation primitives before
any RL training. It is an engineering implementation of the physical causal
relationships described in `PROJECT_CONTEXT.md`; it does not change research
formulas.

## Implemented Primitives

- SNR-AMC lookup with fixed spectral-efficiency table.
- Communication delay:

```text
throughput = bandwidth * spectral_efficiency(SNR)
transmission_delay = compressed_size / throughput
```

- Invisible offload links raise an illegal-action error instead of using a large
placeholder delay.
- Fixed-frequency compute queue:

```text
Q_i(t + 1) = max(Q_i(t) - f_i * delta_t, 0) + arrival_cycles_i(t)
```

- Compute and communication energy under fixed per-cycle and fixed transmit
power assumptions.
- Grid-level AoSI update using probability-OR grid value, semantic change
  factor, and age.
- Grid-level AoSI soft reset using task-achievement rate:

```text
A_g(t + 1) = A_g(t) * (1 - R_g(t)) + delta_t
```

where `R_g(t)` is the semantic-value share of processed ROI in grid `g` that
meets both the quality threshold and the deadline.

## Code Position

`src/simulation/` is the deterministic simulation core layer. It provides
physics and accounting functions that later Gym/RL environments can call without
running vision models online.

`src/simulation/episode.py` is the stage-three episode layer. It loads offline
profiling rows, optionally applies the saved quality proxy, generates legal
actions, runs deterministic baseline policies, updates queues and AoSI across
time slots, and emits policy metrics.

`configs/satellite_env.yaml` is the experiment-parameter layer for deterministic
stage-three runs. It records profile/proxy paths, node compute settings, link
bandwidth, reward weights, seed, deadline, quality threshold, and the default
baseline list. CLI arguments override this YAML file.

## Smoke Entry Point

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\13_validate_simulation_smoke.py --exist-ok
```

The script reads one row from the smoke profiling CSV, evaluates local and
offloaded processing under different link/queue states, and writes:

```text
outputs/simulation/smoke/summary.json
```

It is not a training run.

## Deterministic Baseline Entry Point

Full stage-three deterministic baseline run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\16_run_deterministic_simulation.py --exist-ok
```

Run a specific deterministic stress scenario:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\16_run_deterministic_simulation.py --scenario low_snr --exist-ok
```

Run all deterministic closure scenarios:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\16_run_deterministic_simulation.py --all-scenarios --exist-ok
```

Default inputs:

```text
data/profiling/dota_v1_lite_300_100_100/test_full/roi_profile_smoke.csv
outputs/proxy/test_full/quality_proxy.joblib
```

Default outputs:

```text
outputs/simulation/deterministic/summary.json
outputs/simulation/deterministic/policy_metrics.csv
outputs/simulation/deterministic/decisions.csv
outputs/simulation/deterministic/slot_metrics.csv
outputs/simulation/deterministic/comparison_report.md
```

With `--all-scenarios`, each scenario writes the same file set under:

```text
outputs/simulation/deterministic/default/
outputs/simulation/deterministic/low_snr/
outputs/simulation/deterministic/compute_congested/
outputs/simulation/deterministic/tight_deadline/
```

and the root directory adds:

```text
outputs/simulation/deterministic/summary_all.json
outputs/simulation/deterministic/policy_metrics_all.csv
outputs/simulation/deterministic/comparison_report.md
```

The default run evaluates:

```text
Random
Local-Deep
Local-Adaptive
Best-SNR
Semantic-Greedy
No-AoSI
AoSI-Greedy
```

This is still deterministic environment validation, not PPO or MADRL training.
The first PPO smoke closure is documented separately in
`docs/rl_smoke_pipeline.md` and writes to `outputs/rl/ppo_smoke/`.

Scenario meanings:

- `default`: baseline synthetic visibility/SNR trace and default queues.
- `low_snr`: same visibility timing, but all visible SNR values are shifted down
  to expose AMC and communication-delay sensitivity.
- `compute_congested`: collaborator satellites start with larger queues to
  expose queue-causality effects on offload delay.
- `tight_deadline`: deadline is capped at 120 ms to expose timeout sensitivity.

## Expected Causality Checks

- Weak SNR link is slower than high SNR link.
- Busy compute queue is slower than idle queue.
- Invisible link is illegal.
- Refreshed AoSI cost is lower than aged AoSI cost.
- Policies that select different legal actions produce different delay, energy,
  quality, data-volume, and AoSI outcomes under the same ROI stream and link
  trace.

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_simulation_models.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_deterministic_episode.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```
