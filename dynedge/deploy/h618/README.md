# fer-dynedge — H618 deploy（正式低端 Linux CPU）

论文角色：**六设备矩阵第四档** — consumer-grade Linux CPU 锚点（Cortex-A53）；后端 **NCNN**（FP32 + INT8）。  
替代原计划中的 T113 档位。共享 bench：[`../ncnn_edge`](../ncnn_edge)。CSV 对齐 `bench/schema.py`（与 MCU / T527 / Orin 可拼表）。

当前板：OrangePi Zero3（H618 / 4×A53），Armbian 25.5.1，SSH `root@BOARD_IP_REDACTED`（WiFi 静态 IP）。

## §4.5 对照

同一探路 `param/bin`：T527 A55 **SDOT** INT8 vs H618 A53 **无 SDOT** — INT8 加速比写入跨 ISA 分析。

## Phase 1 验收顺序（Step 1.3）

1. SSH 进板，`uname -m` 确认 aarch64，写入 `configs/versions.yaml` → `h618:` ✅
2. 板端原生编译 NCNN → `/opt/ncnn/install`（见 `scripts/board_setup_ncnn.sh`）✅
3. 放入 `mobilenetv3_small.{param,bin}`（ONNX→onnx2ncnn）✅
4. `bash scripts/run_ncnn_probe.sh` → `results/h618_ncnn_probe.csv` ✅
5. 主机：`python bench/linux_bench.py --parse-file results/h618_ncnn_probe_raw.txt --device h618 --out results/h618_ncnn_probe.csv`
6. INT8 PTQ + 线程扫描：`bash scripts/board_int8_and_matrix.sh` → `results/h618_ncnn_matrix.csv` ✅
7. 真图 INT8 重校准（FER+/ExpW 500）：`bash scripts/board_int8_real_calib.sh` → `results/h618_int8_real_*.csv/json` ✅
8. 汇总：`results/h618_ncnn_SUMMARY.md`

## 复现（板端）

```bash
sudo bash ../ncnn_edge/scripts/lock_perf.sh
bash scripts/run_ncnn_probe.sh
```

主机经 WiFi：

```bash
python bench/linux_bench.py \
  --ssh root@BOARD_IP_REDACTED \
  --remote-cmd "bash /root/fer-dynedge/deploy/h618/scripts/run_ncnn_probe.sh" \
  --device h618 \
  --out results/h618_ncnn_probe.csv
```
