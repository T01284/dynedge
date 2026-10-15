# C1 FLOPs vs LUT — B-1/G-1 artifacts

- points: 116 (flops=48, lut=68)
- iso_acc rows: 34
- iso_lat rows: 34
- frontier Y / 精度列: **§8.38 GPU distill test_acc** (gpu_src=116)
- board full-test accuracy filled: 116; ReLU latency overrides (B12): 40
- dtype plotted: NCNN INT8

## Files
- `points.csv`
- `iso_acc.csv` (iso-accuracy nearest neighbor on task acc)
- `iso_lat.csv` (iso-latency nearest neighbor)
- `frontier_{h618,t527}_{ferplus,expw}_int8.png`

## Iso-acc latency ratio (FLOPs/LUT): mean=1.177 median=1.000 min=0.661 max=2.262
