# Pareto → formal distill candidates (2026-07-21)

From `lut/pareto_h618_v4_sketch.csv` (H618 INT8 LUT + predistill acc):

| tier | config | pred INT8 | FER+ predistill | role |
|---|---|---:|---:|---|
| MCU/fast | w0.35 d2222 @64g | **3.9 ms** | 84.11% | MCU static anchor (already e2e) |
| MCU+ | w0.35 d3333 @64g | **5.1 ms** | 84.14% | MCU fidelity + small bump |
| mid | w0.5 d2222 @96c | **8.8 ms** | 86.17% | Linux mid |
| mid+ | w0.5 d3333 @112c | **15.5 ms** | 86.84% | Linux mid |
| high | w0.75 d3333 @112c | **27.6 ms** | 87.41% | Linux high |
| top | w1.0 d4444 @112c | **46.6 ms** | 88.27% | accuracy end |

**AutoDL**: instance may be offline.  
When up: `bash scripts/remote_formal_lut_v4.sh`（仅 H618；见 `lut/PRECHECK_FORMAL_LUT.md`）

MCU 搜索默认关闭，等 width e2e 补点完成后再 `ENABLE_MCU_LUT_SEARCH=1`。
