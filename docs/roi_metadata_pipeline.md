# ROI Metadata Pipeline

This step builds downstream scheduling inputs from original-image NMS detections.
It is an engineering implementation plus experimental baseline input generation;
it does not change the paper formula and does not train a new model.

## Entry Point

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\08_build_roi_metadata.py --exist-ok
```

Default inputs:

- `outputs/detector/yolo11n_obb_full_test_original/nms_predictions.jsonl`
- `data/DOTA-demo/test/images`

Default outputs:

- `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata.jsonl`
- `data/roi_metadata/dota_demo_1024_o200/test/roi_metadata.csv`
- `data/roi_metadata/dota_demo_1024_o200/test/grid_summary.json`
- `data/roi_metadata/dota_demo_1024_o200/test/summary.json`
- `data/roi_metadata/dota_demo_1024_o200/test/crops/*.png`

## Formula Alignment

The ROI semantic value follows the project formula:

```text
v_k = clip(w_class * rho(y_k) + w_confidence * p_hat_k + w_area * a_k / A(x), 0, 1)
```

Current baseline weights are:

```text
w_class = 0.50
w_confidence = 0.35
w_area = 0.15
```

The AoSI grid summary uses the original-image coordinate system, not detector
tiles. Each ROI is assigned to exactly one `8x8` grid cell by its OBB center.
Grid value uses probability-OR aggregation:

```text
V_g = 1 - product(1 - v_k), k belongs to grid g
```

## Verification

Targeted and full tests:

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_roi_metadata.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```

Smoke build:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\08_build_roi_metadata.py --limit-images 2 --output-dir data\roi_metadata\dota_demo_1024_o200\test_smoke --exist-ok
```
