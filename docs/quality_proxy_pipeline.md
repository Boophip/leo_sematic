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

Outputs:

- `outputs/proxy/smoke/quality_proxy.joblib`
- `outputs/proxy/smoke/summary.json`
- `outputs/proxy/model_comparison/model_comparison.csv`
- `outputs/proxy/model_comparison/summary.json`

The `.joblib` file stores the full sklearn feature pipeline, selected regressor,
feature contract, and model type. The summary stores feature lists, leakage
fields, split strategy, metrics, grouped errors, and scheduling-oriented
diagnostics.

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

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_quality_proxy.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_roi_profiling.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```
