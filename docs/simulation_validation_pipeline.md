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

## Code Position

`src/simulation/` is the deterministic simulation core layer. It provides
physics and accounting functions that later Gym/RL environments can call without
running vision models online.

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

## Expected Causality Checks

- Weak SNR link is slower than high SNR link.
- Busy compute queue is slower than idle queue.
- Invisible link is illegal.
- Refreshed AoSI cost is lower than aged AoSI cost.

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_simulation_models.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```
