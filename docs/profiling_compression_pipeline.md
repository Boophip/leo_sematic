# ROI Compression and Profiling Pipeline

This step fixes the first reproducible compression policy for stage-two
profiling. It is an engineering implementation and experimental setting; it
does not change the paper formula.

## Compression Policy

Local processing is not compressed because no inter-satellite payload is sent.
Only offloaded ROI crops use `Beta 0..3`.

| Level | Encoding | Resize | Minimum short side | Use |
|---|---|---:|---:|---|
| `local` | none | 1.00x | original | local early-exit baseline |
| `beta_0` | PNG lossless | 1.00x | original | quality upper bound |
| `beta_1` | JPEG quality 90 | 1.00x | original | high-quality offload |
| `beta_2` | JPEG quality 70 | 0.75x | 16 px | medium bandwidth pressure |
| `beta_3` | JPEG quality 45 | 0.50x | 8 px | aggressive low-bandwidth offload |

The minimum short-side rule prevents small DOTA ROI crops from being collapsed
below a usable size before the multi-exit model resizes them to its input
resolution.

## Profiling Output

The smoke profiling script writes a CSV with at least:

```text
roi_id
image_id
predicted_class
class_id
detector_confidence
area_ratio
semantic_value
grid_id
class_priority
quality_label
exit_level
compression_level
compressed_bytes
encode_ms
decode_ms
inference_ms
task_quality
```

Additional columns may include the encoded format, compressed image size,
`exit_confidence`, and predicted class id. `detector_confidence` is the original
YOLO-OBB confidence from ROI metadata; `exit_confidence` is the multi-exit model
output confidence and must not be used as a quality-proxy input feature.

## Entry Points

Compression/profile smoke generation:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\10_profile_roi_compression.py --limit-rois 16 --exist-ok
```

Omit `--limit-rois` for full profiling. Add `--include-local` when building
the full scheduling table so the no-offload local action is profiled alongside
`beta_0..3`.

Tiny multi-exit smoke training:

```powershell
F:\anaconda\envs\leo_semantic\python.exe scripts\11_train_multi_exit_smoke.py --limit-rois 128 --epochs 1 --exist-ok
```

Before any training run, report the expected runtime. The tiny smoke run is
expected to take about 5-15 minutes depending on CPU/GPU availability; it is not
a formal experiment.

## Relation to `project.pdf`

`project.pdf` keeps compression as an offline-profiled action dimension combined
with the early-exit level. It does not prescribe JPEG quality values or resize
ratios. The concrete values above are therefore fixed as the first reproducible
engineering baseline, while the research formula remains unchanged.

## Verification

```powershell
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_roi_compression.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_roi_profiling.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -p test_multi_exit.py -v
F:\anaconda\envs\leo_semantic\python.exe -m unittest discover -s tests -v
```
