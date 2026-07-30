# LUT (first-pass)

| 文件 | 说明 |
|---|---|
| `topo_ops.py` | v3_wide/v4 拓扑 → op 列表（与 `elastic_mbv3` 对齐） |
| `gen_ops.py` | 全枚举 → `enum_v4_full.json`（n≈786） |
| `enum_v4_full.json` | MCU + H618 全 width×res×dtype×depth |
| `mine_ops_h618_batch.py` / `pull_h618_ops_mine.py` | H618 板端批采 |
| `ops_h618_v4_full.csv` | H618 算子 LUT（ok≈618/706） |
| `predict_e2e.py` | LUT 求和 vs e2e；`--fit-per-dtype --fit-affine` |
| `FIDELITY_H618.md` / `fidelity_h618_v4.json` | **Phase 1 验收：r=0.996 MAPE=5.3% PASS** |
| `e2e_v4_firstpass.csv` | MCU+H618 e2e（34 行） |
| `mine_ops_espdl.py` + `tools/espdl_ptq_op.py` | MCU 单算子 PTQ→烧录→`CMD BENCH_SYN` |
| `ops_espdl_v4_firstpass.csv` | MCU 算子 LUT（开采中） |

注意：NCNN `.bin` 每层权重前需 `int32=0`；`ConvolutionDepthWise` 需写全 kernel/stride/pad 与 `7=groups`。
