# LUT fidelity H618 (Phase 1 gate)

## Result (2026-07-21)

| metric | value | gate |
|---|---:|---|
| usable e2e rows | 32/32 | — |
| Pearson r | **0.996** | ≥0.95 |
| MAPE | **5.26%** | ≤10% |
| **PASS** | **True** | |

Calibration: per-dtype affine `pred = a·Σop + b`
- fp32: a≈0.879, b≈1187 µs
- int8: a≈0.806, b≈878 µs

Raw sum (a=1,b=0): r=0.993, MAPE≈12.8% — correlation already passes; bias term captures NCNN framework / memory overhead.

## Artifacts

- Enum: `lut/enum_v4_full.json` (n=786; H618 mineable 706)
- Op LUT: `lut/ops_h618_v4_full.csv` (ok=618/706; add Eltwise unsupported → proxy)
- Report: `lut/fidelity_h618_v4.json`
- Predictor: `python lut/predict_e2e.py --fit-per-dtype --fit-affine`
