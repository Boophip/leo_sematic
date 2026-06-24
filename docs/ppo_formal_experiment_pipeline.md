# PPO Formal Experiment Pipeline

This stage promotes the PPO scheduling closure from one smoke run to a
multi-seed experiment manager. It does not change the simulation formulas,
reward accounting, quality proxy, or synthetic link-trace model. It repeats the
existing PPO all-scenario runner, passes the main candidate-action generation
settings through, aggregates mean and standard deviation statistics, and writes
paper-facing CSV, Markdown, and PNG artifacts.

## Code Position

`scripts/18_run_ppo_formal_experiment.py` is the experiment orchestration layer.
It calls `scripts/17_train_ppo_smoke.py` once per seed with `--all-scenarios`,
then aggregates each seed's `summary_all.json`.

`tests/test_ppo_formal_experiment.py` covers the report-only aggregation logic
and command construction without launching PPO training.

## Formal Default Run

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --exist-ok
```

The default run uses:

```text
seeds: 42 43 44
limit_rois: 512
total_timesteps: 5000 per scenario
scenarios: default, low_snr, compute_congested, tight_deadline
candidate_mode: feasible-topk
candidate_top_k: 12
```

Before any training starts, the script prints an aggregate runtime estimate. On
the current development machine, the default three-seed run is expected to take
roughly 24-96 minutes because it trains one PPO policy per seed and scenario.

## Short Verification Run

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --seeds 42 43 --limit-rois 64 --total-timesteps 128 --output-dir outputs\rl\ppo_formal_pilot --exist-ok
```

Use `--dry-run` to print the exact underlying PPO commands without starting
training.

Diagnostic pilot with training curves:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --seeds 42 43 --limit-rois 128 --total-timesteps 1024 --eval-frequency 256 --ppo-ent-coef 0.01 --output-dir outputs\rl\ppo_diagnostic_pilot --exist-ok
```

Constraint-shaping diagnostic pilot:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --seeds 42 43 --limit-rois 128 --total-timesteps 2048 --eval-frequency 256 --checkpoint-frequency 512 --select-best-checkpoint --best-checkpoint-min-success-rate 0.5 --ppo-ent-coef 0.01 --reward-quality-deficit-weight 0.5 --reward-delay-excess-weight 0.5 --reward-virtual-queue-weight 0.1 --output-dir outputs\rl\ppo_constraint_diagnostic --exist-ok
```

Full legal action-space control:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --seeds 42 43 --limit-rois 128 --total-timesteps 2048 --candidate-mode legal --eval-frequency 256 --checkpoint-frequency 512 --select-best-checkpoint --best-checkpoint-min-success-rate 0.5 --ppo-ent-coef 0.01 --reward-quality-deficit-weight 0.5 --reward-delay-excess-weight 0.5 --reward-virtual-queue-weight 0.1 --output-dir outputs\rl\ppo_candidate_legal_pilot --exist-ok
```

Main Proxy-Guided Feasible Top-k PPO pilot:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --seeds 42 43 --limit-rois 128 --total-timesteps 2048 --candidate-mode feasible-topk --candidate-top-k 12 --eval-frequency 256 --checkpoint-frequency 512 --select-best-checkpoint --best-checkpoint-min-success-rate 0.5 --ppo-ent-coef 0.01 --reward-quality-deficit-weight 0.5 --reward-delay-excess-weight 0.5 --reward-virtual-queue-weight 0.1 --output-dir outputs\rl\ppo_candidate_topk_pilot --exist-ok
```

Paper-facing strict proxy main run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\18_run_ppo_formal_experiment.py --candidate-mode feasible-topk --candidate-top-k 12 --seeds 42 43 44 --limit-rois 512 --total-timesteps 100000 --eval-frequency 2000 --checkpoint-frequency 20000 --select-best-checkpoint --best-checkpoint-min-success-rate 0.5 --ppo-ent-coef 0.01 --reward-quality-deficit-weight 0.5 --reward-delay-excess-weight 0.5 --reward-virtual-queue-weight 0.1 --profile-csv data\profiling\dota_v1_lite_300_100_100\test_strict_image_features\roi_profile_smoke.csv --quality-proxy outputs\proxy\strict_image_features\quality_proxy.joblib --output-dir outputs\rl\ppo_topk_strict_proxy_shaping_100k --exist-ok
```

## Outputs

```text
outputs/rl/ppo_formal/seed_<seed>/
outputs/rl/ppo_formal/policy_seed_scenario_metrics.csv
outputs/rl/ppo_formal/scenario_seed_winners.csv
outputs/rl/ppo_formal/scenario_policy_aggregate.csv
outputs/rl/ppo_formal/policy_formal_aggregate.csv
outputs/rl/ppo_formal/scenario_formal_summary.csv
outputs/rl/ppo_formal/training_curve_all.csv
outputs/rl/ppo_formal/comparison_report.md
outputs/rl/ppo_formal/summary.json
outputs/rl/ppo_formal/figures/qoe_by_policy.png
outputs/rl/ppo_formal/figures/qoe_by_scenario_policy.png
outputs/rl/ppo_formal/figures/qoe_cdf_by_policy.png
outputs/rl/ppo_formal/figures/training_curve_qoe.png
outputs/rl/ppo_formal/figures/training_curve_actions.png
outputs/rl/ppo_formal/figures/training_curve_violations.png
outputs/rl/ppo_formal/figures/training_curve_queues.png
```

The aggregate reports use canonical slot-level `qoe_total`, not the PPO
learning reward. The per-seed subdirectories preserve the full PPO model,
decisions, metrics, and all-scenario reports generated by the smoke runner.

The formal runner passes through PPO overrides such as `--ppo-ent-coef` and
diagnostic controls such as `--eval-frequency` and `--checkpoint-frequency`.
When evaluation frequency is enabled, `training_curve_all.csv` aggregates
per-seed/per-scenario canonical QoE curves.

`--select-best-checkpoint` passes through to the smoke runner and selects the
best canonical evaluation checkpoint for final PPO-vs-baseline comparison.
Use `--best-checkpoint-min-success-rate` to prevent all-drop or otherwise
degenerate policies from being selected only because their delay and energy
costs are near zero.
The optional `--reward-quality-deficit-weight`,
`--reward-delay-excess-weight`, and `--reward-virtual-queue-weight` parameters
shape only the PPO training reward. The aggregate reports still rank policies
with canonical `qoe_total`, so shaped and unshaped runs should be reported as
separate training-reward variants.

The optional `--candidate-mode` parameter is passed through to the smoke runner.
`feasible-topk` is the main method setting: it keeps `drop`, filters physically
invalid local/offload choices, ranks legal non-drop actions by proxy-estimated
quality, delay and compressed bytes, and exposes only top-k candidates to PPO.
`legal` keeps every physically legal action and is the full action-space control
that demonstrates the learning difficulty without candidate generation. `fixed`
preserves the original raw action mapping for low-level environment debugging.
These modes do not modify canonical QoE; they only affect which physically
feasible action is executed for a raw PPO index. Formal CSVs include candidate
remap count, mean candidate count, and executed illegal count so candidate
experiments can be audited separately from QoE.

## Current Scope

This is the first formal experiment manager over the reproducible synthetic
LEO link traces. It does not connect STK, TLE, ns-3, or real orbital traces.
Those are high-fidelity follow-up inputs, not prerequisites for this experiment
automation layer.

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_ppo_formal_experiment.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```
