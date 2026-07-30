# fer-dynedge — T527 deploy (A55 + NPU)

论文角色：车规/工业边缘；后端 **VIPLite NPU（主，公开 ACUITY→NBG）** + **NCNN CPU（对照 / 回退）**。  
共享 bench：[`../ncnn_edge`](../ncnn_edge)。

> **板子**：MYIR **T527 DEMO CAR**  
> **当前系统**：Buildroot Linux（`myd-lt527`，kernel 5.15.147，aiot-linux-v1.4.8）— 已于 2026-07-23 刷入。  
> 串口：`SERIAL_PORT` @115200；登录 `root` / `root`。  
> 联网：`WIFI_SSID_REDACTED.4G` → `BOARD_IP_REDACTED`；SSH：`ssh root@BOARD_IP_REDACTED`（密码 `root`）。  
> NPU：`/dev/vipcore` + `/usr/lib/libVIPlite.so`（VIPLite 1.13.0.0-AW-2024-02-04）。

## 策略（定稿）

```
Phase A  Android（现在）     → 能做的先做完，数字标 transitional
Phase B  刷 Linux            → VIPLite 运行时 + SSH + 重跑 NCNN
Phase C  公开 ACUITY 到手    → ONNX→NBG→NPU；可重做 A/B 延迟与功耗（论文以 Linux 为准）
```

**两者都要**：NPU 主叙事 + NCNN 对照；找不到 ACUITY 时 T527 仍以 NCNN（A55 SDOT）锚定矩阵。

| 阶段 | 做 | 不做 / 可延后 |
|---|---|---|
| **A · Android** | NCNN FP32/INT8 探路 CSV；锁频；P5S1 功耗（表接好后）；确认 `/dev/vipcore` + `libVIPlite.so` | 正式论文唯一数字；完整 NPU 主表 |
| **B · Linux** | 刷机；NCNN 重测；VIPLite 样例/头文件对齐 | — |
| **C · ACUITY** | 公开 Toolkit → NBG；算子清单；NPU vs CPU；按需 **重测** A/B | 等不到也不阻塞主表 CPU 行 |

Android 探路结果保留在 `results/t527_ncnn_*_probe.csv`，文件名/meta 视为 **transitional**；Linux 重测后写入 `results/t527_linux_*`。

## 目录

| 路径 | 用途 |
|---|---|
| `scripts/check_npu.sh` | Linux BSP 版 NPU 检查 |
| `scripts/check_npu_android.sh` | Android 版 NPU/锁频检查 |
| `scripts/build_android_ncnn.sh` | NDK 交叉编译 ncnn + `fer_ncnn_bench` |
| `scripts/run_ncnn_probe.sh` | 板上 NCNN 探路 |
| `acuity/` | ONNX→NBG（公开 ACUITY）占位 |
| `docs/npu_ops_checklist.md` | NPU 算子支持清单 |

## Phase A 验收（Android · transitional）

1. ~~`check_npu_android.sh`：`/dev/vipcore` + VIPLite so~~ ✅
2. ~~`fer_ncnn_bench` FP32/INT8~~ ✅ → `results/t527_ncnn_{fp32,int8}_probe.csv`
3. **待做**：P5S1 串供电，采推理窗 mJ/inference（`bench/p5s1/`）

## Phase B / C（Linux · 当前正式）

1. ~~`vipcore` + VIPLite + SSH~~ ✅（`BOARD_IP_REDACTED`）
2. ~~NCNN Linux formal~~ ✅ → `results/t527_linux_{fp32,int8,int8_real}_probe.csv` + `t527_linux_ncnn_matrix.csv`
3. ~~ACUITY → NBG → `vpm_run`~~ ✅
   - 真图校准 NBG：**p50≈1.45 ms**（`t527_linux_npu_mnv3_fer_real_probe.csv`）
   - NPU↔NCNN 精度：`results/t527_npu_vs_ncnn_SUMMARY.md`（ImageNet match **0.765**）
   - 全量 LUT：**706/706** → `lut/ops_t527_v4_full.csv`（`lut/ops_t527_SUMMARY.md`）
4. 论文延迟数字：**以 Linux 为准**（NPU 主叙事 + NCNN 对照）
5. **待做**：P5S1 功耗

## 与 H618 对比（§4.5）

同一 `param/bin`：T527 A55 **SDOT** INT8 vs H618 A53 无 SDOT。
