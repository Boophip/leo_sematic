# Quality Proxy Pipeline

This step trains task-quality proxy candidates from ROI profiling rows. It is an
engineering baseline for the stage-two closure and does not change the paper
formula:

```text
q_k = Proxy(ROI metadata, exit_level, compression_level)
```

## Inputs

The training CSV is produced by `scripts/10_profile_roi_compression.py` and must
contain `task_quality`, `image_id`, `exit_level`, `compression_level`,
`predicted_class`, ROI metadata fields, and compression cost fields.

Allowed proxy inputs are online-available or table-available values:

```text
predicted_class
compression_level
class_id
detector_confidence
area_ratio
semantic_value
grid_id
class_priority
exit_level
compressed_bytes
output_width
output_height
crop_width
crop_height
crop_aspect_ratio
brightness_mean
brightness_std
rgb_mean_r
rgb_mean_g
rgb_mean_b
rgb_std_r
rgb_std_g
rgb_std_b
laplacian_var
edge_density
entropy
```

The crop/image statistics are extracted from the decoded ROI image in
`src/proxy/image_features.py`. They are online-safe image quality cues, not
labels or downstream model outputs.

Forbidden leakage fields are:

```text
quality_label
task_quality
predicted_class_id
exit_confidence
```

These fields may appear in the profiling CSV for supervision and diagnostics,
but they must not be used as model inputs.

## Entry Point

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\12_train_quality_proxy_smoke.py --model-type hist_gbdt --include-image-features --exist-ok
```

Before training, the script prints the expected runtime. A single sklearn proxy
run usually takes about 1-5 minutes and is not an RL run.

Supported model families:

```text
mlp
hist_gbdt
random_forest
extra_trees
```

To compare all supported model families on the same split:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\19_compare_quality_proxy_models.py --profile-csv data\profiling\dota_v1_lite_300_100_100\test_full\roi_profile_smoke.csv --exist-ok
```

The comparison script records each `.joblib` size and, by default, only models
at or below 50 MB are eligible to become the copied primary proxy. This keeps
the selected proxy aligned with the lightweight-system assumption. Use
`--max-primary-model-mb 0` only when intentionally disabling that cap.

Outputs:

- `outputs/proxy/smoke/quality_proxy.joblib`
- `outputs/proxy/smoke/summary.json`
- `outputs/proxy/model_comparison/model_comparison.csv`
- `outputs/proxy/model_comparison/summary.json`

The `.joblib` file stores the full sklearn feature pipeline, selected regressor,
feature contract, and model type. The summary stores feature lists, leakage
fields, split strategy, model sizes, metrics, grouped errors, and
scheduling-oriented diagnostics.

## Split Policy

If profiling rows contain at least two `image_id` values, training uses
`GroupShuffleSplit` to keep original images isolated between train and test. For
tiny smoke tables with one image only, it falls back to a deterministic row
split and labels the split as `row_split_smoke_only`.

## Metrics

The smoke report includes:

- MAE
- MSE
- R-squared
- grouped MAE/MSE by predicted class, exit level, and compression level
- high-value ROI MAE/MSE/R-squared
- `q >= quality_threshold` precision, recall, and F1
- per-ROI top-1 action agreement and proxy-induced regret

Smoke metrics validate the engineering path only and should not be used as
paper results.

## Strict Train/Val/Test Proxy Flow

The paper-facing proxy workflow uses fixed DOTA image-level splits and does not
select a model from the final test split. It first generates one profiling table
per split, then trains on train, selects the primary model with validation
metrics, and reports final metrics on test.

Before starting the three profiling runs, report this runtime estimate:

```text
Strict train+val+test profiling estimate: about 40-240 minutes, depending on GPU and disk throughput.
```

Commands:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\10_profile_roi_compression.py --metadata data\roi_crops_gt\dota_v1_lite_300_100_100\train\roi_metadata_labeled.jsonl --checkpoint outputs\multi_exit\formal\best.pt --output-dir data\profiling\dota_v1_lite_300_100_100\train_strict_image_features --include-local --compression-levels 0 1 2 3 --exist-ok

F:\anaconda\envs\leo_semantic\python.exe scripts\10_profile_roi_compression.py --metadata data\roi_crops_gt\dota_v1_lite_300_100_100\val\roi_metadata_labeled.jsonl --checkpoint outputs\multi_exit\formal\best.pt --output-dir data\profiling\dota_v1_lite_300_100_100\val_strict_image_features --include-local --compression-levels 0 1 2 3 --exist-ok

F:\anaconda\envs\leo_semantic\python.exe scripts\10_profile_roi_compression.py --metadata data\roi_crops_gt\dota_v1_lite_300_100_100\test\roi_metadata_labeled.jsonl --checkpoint outputs\multi_exit\formal\best.pt --output-dir data\profiling\dota_v1_lite_300_100_100\test_strict_image_features --include-local --compression-levels 0 1 2 3 --exist-ok
```

Expected row counts are:

```text
train: 386720
val:   133120
test:  136840
```

After profiling, compare the lightweight candidates:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\19_compare_quality_proxy_models.py --train-profile-csv data\profiling\dota_v1_lite_300_100_100\train_strict_image_features\roi_profile_smoke.csv --val-profile-csv data\profiling\dota_v1_lite_300_100_100\val_strict_image_features\roi_profile_smoke.csv --test-profile-csv data\profiling\dota_v1_lite_300_100_100\test_strict_image_features\roi_profile_smoke.csv --output-dir outputs\proxy\strict_image_features --model-types mlp hist_gbdt --exist-ok
```

Before starting this proxy comparison, report this runtime estimate:

```text
Strict proxy comparison estimate: about 2-10 minutes for mlp and hist_gbdt.
```

The comparison script copies the selected primary model to:

```text
outputs/proxy/strict_image_features/quality_proxy.joblib
```

Primary selection uses validation high-value MAE, validation R-squared, and
validation threshold F1, with a 50 MB default model-size cap. Test metrics are
reported after selection only; they must not be used to choose the primary
model.

Run deterministic sanity with the strict test profile and strict proxy:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\16_run_deterministic_simulation.py --profile-csv data\profiling\dota_v1_lite_300_100_100\test_strict_image_features\roi_profile_smoke.csv --quality-proxy outputs\proxy\strict_image_features\quality_proxy.joblib --output-dir outputs\simulation\deterministic_strict_proxy_all --all-scenarios --exist-ok
```

The current top-k PPO result remains a candidate-generation ablation unless it
is explicitly promoted to the method. Do not silently replace the paper-aligned
legal action-space result with `feasible-topk`.

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_quality_proxy.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_quality_proxy_model_comparison.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_roi_profiling.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```
