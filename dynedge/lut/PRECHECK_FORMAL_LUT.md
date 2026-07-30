# GPU 开窗前预检（更新 2026-07-21 16:30）

| # | 项 | 状态 |
|---|---|---|
| 1 | MCU/NPU **原生 ReLU** 蒸馏 | ✅ `--student-act relu` |
| 2 | H618 **dtype=auto**（min fp32/int8） | ✅ `search_subnets_lut.py` |
| 3 | MCU 代价模型 | ✅ **穷举 e2e 真值表**（可行闭包 23 行）；本地搜索初稿已出 |

H618：`bash scripts/remote_formal_lut_v4.sh`（待 AutoDL）  
MCU：真值 `lut/e2e_mcu_truth.csv` → 搜索 `results/search_lut/search_ferplus_mcu_e2e.json`（见 `lut/MCU_E2E_LUT.md`）；正式精度待全量 val + ReLU 蒸馏。
