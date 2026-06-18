# ROI Labeling Pipeline

This step labels generated ROI metadata by matching original-image ROI OBBs
against DOTA `labelTxt` ground truth. It is an engineering implementation plus
experimental baseline input generation; it does not change paper formulas and
does not train a model.

## Entry Point

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\09_label_roi_metadata.py --exist-ok
```

Default inputs:

- `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata.jsonl`
- `data/DOTA-demo/test/labelTxt`

Default outputs:

- `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata_labeled.jsonl`
- `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata_labeled.csv`
- `data/roi_metadata/dota_demo_1024_o200/test/roi_label_summary.json`

## Matching Policy

- Matching uses original-image rotated IoU with default threshold `0.5`.
- True positives follow DOTA/VOC-style one-to-one matching per image and class,
  sorted by confidence descending.
- Detections matching same-class `difficult != 0` GT are marked
  `ignored_difficult`, not TP and not FP.
- Every ROI also stores the best geometry-overlap GT class and IoU, even when
  the prediction is a class mismatch.

Key label values:

- `true_positive`
- `duplicate_detection`
- `ignored_difficult`
- `class_mismatch`
- `background_fp`

These labels are offline supervision metadata for later task-quality profiling
and proxy modeling. They should not be described as per-ROI mAP.

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_roi_labeling.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```

Smoke labeling:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\09_label_roi_metadata.py --metadata data\roi_metadata\dota_demo_1024_o200\test_smoke\roi_metadata.jsonl --output-dir data\roi_metadata\dota_demo_1024_o200\test_smoke --limit-images 2 --exist-ok
```
