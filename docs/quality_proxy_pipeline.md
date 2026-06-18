# Quality Proxy Pipeline

This step trains the first smoke task-quality proxy from ROI profiling rows. It
is an engineering baseline for the stage-two closure and does not change the
paper formula:

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
```

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
F:\anaconda\envs\leo_semantic\python.exe scripts\12_train_quality_proxy_smoke.py --exist-ok
```

Before training, the script prints the expected smoke runtime. The default smoke
run should take about 1-3 minutes and is not a formal experiment.

Outputs:

- `outputs/proxy/smoke/quality_proxy.joblib`
- `outputs/proxy/smoke/summary.json`

The `.joblib` file stores the full sklearn feature pipeline and MLP regressor.
The summary stores feature lists, leakage fields, split strategy, metrics, and
grouped errors.

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

Smoke metrics validate the engineering path only and should not be used as
paper results.

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_quality_proxy.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_roi_profiling.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```
