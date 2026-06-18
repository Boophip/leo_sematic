# Stage 2 Formalization Pipeline

This document turns the smoke-only stage-two closure into a reproducible formal
experiment path. It is an engineering implementation and experimental baseline;
it does not change the paper formula.

## Data Source

Detected ROI metadata currently exists only for the test split, so it must not
be used to train the multi-exit task model. The formal task model uses GT ROI
crops generated from original-image split manifests:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\14_build_gt_roi_crops.py --split all --exist-ok
```

Outputs:

```text
data/roi_crops_gt/dota_v1_lite_300_100_100/{train,val,test}/roi_metadata_labeled.jsonl
```

These crops are supervised task-model data, not detector-produced scheduling
inputs.

## Multi-Exit Training

Small formal validation run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\15_train_multi_exit_formal.py --limit-train-rois 512 --limit-val-rois 256 --epochs 3 --exist-ok
```

Full baseline run:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\15_train_multi_exit_formal.py --epochs 20 --exist-ok
```

Before training starts, the script prints a runtime estimate. Full training is
expected to take about 30-90 minutes depending on GPU and ROI count.

Outputs:

```text
outputs/multi_exit/formal/best.pt
outputs/multi_exit/formal/last.pt
outputs/multi_exit/formal/summary.json
```

## Profiling and Proxy

After a useful multi-exit checkpoint exists, profile split-isolated ROI crops:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\10_profile_roi_compression.py --metadata data\roi_crops_gt\dota_v1_lite_300_100_100\test\roi_metadata_labeled.jsonl --checkpoint outputs\multi_exit\formal\best.pt --output-dir data\profiling\dota_v1_lite_300_100_100\test_full --include-local --exist-ok
```

Then train the quality proxy:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\12_train_quality_proxy_smoke.py --profile-csv data\profiling\dota_v1_lite_300_100_100\test_medium\roi_profile_smoke.csv --output-dir outputs\proxy\medium --exist-ok
```

The `smoke` script name remains for compatibility; the input/output paths decide
whether the run is a tiny smoke run or a larger baseline run.

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_gt_roi_crops.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_multi_exit.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```
