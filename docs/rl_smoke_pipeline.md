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

`scripts/17_train_ppo_smoke.py` is the executable smoke-training and evaluation
entry point. It reads `configs/satellite_env.yaml`, trains or loads PPO,
evaluates the learned policy, and compares it with the deterministic baselines
on the same ROI stream and synthetic link trace.

## Entry Point

Small verification run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\17_train_ppo_smoke.py --limit-rois 128 --total-timesteps 256 --exist-ok
```

Default smoke run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\17_train_ppo_smoke.py --exist-ok
```

All deterministic stress scenarios:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\17_train_ppo_smoke.py --limit-rois 64 --total-timesteps 128 --all-scenarios --exist-ok
```

Diagnostic training run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\17_train_ppo_smoke.py --limit-rois 64 --total-timesteps 256 --eval-frequency 128 --checkpoint-frequency 128 --exist-ok
```

Constraint-shaping diagnostic run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\17_train_ppo_smoke.py --limit-rois 128 --total-timesteps 1024 --eval-frequency 256 --checkpoint-frequency 256 --select-best-checkpoint --best-checkpoint-min-success-rate 0.5 --reward-quality-deficit-weight 0.5 --reward-delay-excess-weight 0.5 --reward-virtual-queue-weight 0.1 --exist-ok
```

Reuse an existing checkpoint without training:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\17_train_ppo_smoke.py --eval-only --model-path outputs\rl\ppo_smoke\model.zip --exist-ok
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
outputs/rl/ppo_smoke/final_model.zip
outputs/rl/ppo_smoke/summary.json
outputs/rl/ppo_smoke/eval_decisions.csv
outputs/rl/ppo_smoke/eval_metrics.json
outputs/rl/ppo_smoke/comparison_metrics.csv
outputs/rl/ppo_smoke/comparison_report.md
```

Diagnostic runs with evaluation/checkpoint frequency can additionally write:

```text
outputs/rl/ppo_smoke/best_model.zip
outputs/rl/ppo_smoke/training_curve.csv
outputs/rl/ppo_smoke/checkpoints/
outputs/rl/ppo_smoke/figures/training_curve_qoe.png
outputs/rl/ppo_smoke/figures/training_curve_actions.png
outputs/rl/ppo_smoke/figures/training_curve_violations.png
outputs/rl/ppo_smoke/figures/training_curve_queues.png
```

With `--all-scenarios`, each scenario writes the same files under its own
subdirectory, and the root output directory also writes:

```text
outputs/rl/ppo_smoke/comparison_metrics_all.csv
outputs/rl/ppo_smoke/scenario_winners.csv
outputs/rl/ppo_smoke/policy_aggregate.csv
outputs/rl/ppo_smoke/comparison_report.md
outputs/rl/ppo_smoke/summary_all.json
```

`training_reward_total` is the PPO learning signal accumulated by the Gym
environment. `qoe_total` in reports is the canonical slot-level QoE shared with
the deterministic baseline evaluator, so it is the value to use for policy
comparison.

For all-scenario runs, `scenario_winners.csv` records the best policy, PPO
rank, and PPO QoE gap in each stress preset. `policy_aggregate.csv` records
cross-scenario averages and best-counts for each policy.

`--ppo-n-steps`, `--ppo-batch-size`, `--ppo-n-epochs`,
`--ppo-learning-rate`, `--ppo-gamma`, and `--ppo-ent-coef` override PPO
training hyperparameters for diagnostics. `--eval-frequency` writes canonical
QoE evaluation rows during training, and `--checkpoint-frequency` writes
intermediate PPO checkpoints.

`--select-best-checkpoint` requires `--eval-frequency > 0`. When enabled,
`model.zip` is the checkpoint with the best canonical evaluation QoE during
training. `final_model.zip` still preserves the last PPO update, and
`best_model.zip` preserves the best evaluation checkpoint.
Use `--best-checkpoint-min-success-rate` to avoid selecting a high-QoE but
degenerate all-drop checkpoint.

`--reward-quality-deficit-weight`, `--reward-delay-excess-weight`, and
`--reward-virtual-queue-weight` add optional constraint-aware shaping to the PPO
training reward. They are diagnostic stabilizers inspired by the paper's
quality and delay virtual queues. They do not change the canonical slot-level
`qoe_total` formula used for reports and baseline comparison.

The RL observation includes quality and delay virtual queues following the
paper's constraint-tracking idea. These queues are reported separately and do
not change the canonical slot-level `qoe_total` used for policy comparison.

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

The multi-seed formal experiment manager that repeats this runner and writes
mean/std reports is documented in `docs/ppo_formal_experiment_pipeline.md`.
