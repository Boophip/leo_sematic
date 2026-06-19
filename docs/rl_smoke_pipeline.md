# PPO Smoke Scheduling Pipeline

This stage adds the first single-agent PPO closure on top of the deterministic
satellite simulation. It is an engineering smoke run for the paper's
`Proposed-RL` path; it does not change the research formulas and it is not a
formal paper result.

## Code Position

`src/envs/` is the RL environment layer. It wraps the existing deterministic
simulation, offline profiling rows, and saved quality proxy in a Gymnasium
interface. The environment does not run detector, multi-exit, or compression
models online.

`scripts/17_train_ppo_smoke.py` is the executable smoke-training entry point.
It reads `configs/satellite_env.yaml`, trains PPO, evaluates the learned policy,
and compares it with the deterministic baselines on the same ROI stream and
synthetic link trace.

## Entry Point

Small verification run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\17_train_ppo_smoke.py --limit-rois 128 --total-timesteps 256 --exist-ok
```

Default smoke run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\17_train_ppo_smoke.py --exist-ok
```

Before training starts, the script prints a runtime estimate. The default
`5000` timestep run is expected to take about 2-8 minutes on the current
development machine.

## Default Inputs

```text
data/profiling/dota_v1_lite_300_100_100/test_full/roi_profile_smoke.csv
outputs/proxy/test_full/quality_proxy.joblib
configs/satellite_env.yaml
```

The link trace remains the reproducible synthetic trace from stage three. No
STK, TLE, or external orbit simulator is required for this smoke closure.

## Outputs

```text
outputs/rl/ppo_smoke/model.zip
outputs/rl/ppo_smoke/summary.json
outputs/rl/ppo_smoke/eval_decisions.csv
outputs/rl/ppo_smoke/eval_metrics.json
outputs/rl/ppo_smoke/comparison_metrics.csv
outputs/rl/ppo_smoke/comparison_report.md
```

The comparison report includes `Proposed-RL` plus:

```text
Random
Local-Deep
Local-Adaptive
Best-SNR
Semantic-Greedy
No-AoSI
AoSI-Greedy
```

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_rl_env.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```
