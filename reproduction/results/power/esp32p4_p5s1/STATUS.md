# ESP32-P4 P5S1 功耗矩阵 — **NG**

| 项 | 状态 |
|---|---|
| 日期 | 2026-07-24 |
| 延迟（CMD BENCH P50） | ✅ 可用（19.6–47.9 ms，G3×4） |
| 能耗（mJ/inf / idle–active） | ❌ **NG — 勿写入论文主表** |

## 原因

采集曲线约 **8 V / 0.74 mA / ~6 mW**，idle≈active，net≈0。相对 S3 同仪表（~5 V / ~216 mA / ~1 W）不合理，判定 **P5S1 未正确串入 Tab5 供电回路**（或 USB 仍供电）。

## 保留文件

- `SUMMARY.csv` / `*_run1.csv`：原始跑次留档；能耗列作废
- 延迟列可与 LUT / board_check 交叉核对

## 后续

接线按 `bench/p5s1/README.md`（剪断 USB +5V、外供高端串表）修好后，用：

```bat
python scripts/run_mcu_p5s1_matrix.py --device esp32p4 --board-port COM15 --meter-port COM40
```

重采；成功后再把本文件改为 OK。
