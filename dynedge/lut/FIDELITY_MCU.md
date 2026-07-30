# MCU LUT（穷举 e2e 真值表，2026-07-21 定稿）

## 方法
MCU **不再**用算子级建模作搜索代价。候选空间小 → 逐配置上板 e2e → LUT = 真值表，保真度 **100%**（无拟合）。

详见 `lut/MCU_E2E_LUT.md`。

## 已测真值（`e2e_mcu_truth.csv`，6/48）
| device | config | e2e p50 |
|---|---|---:|
| esp32p4 | w0.35 d2222 | 19.7 ms |
| esp32p4 | w0.35 d3333 | 28.0 ms |
| esp32p4 | w0.5 d2222 | 31.3 ms |
| esp32s3 | w0.35 d2222 | 58.8 ms |
| esp32s3 | w0.35 d3333 | 86.6 ms |
| esp32s3 | w0.5 d2222 | 95.9 ms |

## §4.5 旁证（算子占墙钟）
| device | Σops/e2e |
|---|---:|
| esp32p4 | **≈27%** |
| esp32s3 | **≈52%** |

`ops_espdl_v4_firstpass.csv`（70/70）仅保留作该分析，不参与 MCU 搜索。

## 失败
- w0.75@64：flash 装不下（不进枚举空间）

## 下一步
补齐 `enum_mcu_e2e.json` 其余 42 次 bench → `search_subnets_mcu_e2e.py` 选锚点。
