# fer-dynedge — Orin Nano deploy（边缘 GPU / TensorRT）

论文角色：**六设备矩阵第二档** — Jetson Orin Nano 8GB；后端 **TensorRT**（FP16 主表 + INT8 对照）。  
CSV 对齐 `bench/schema.py`（`device=orin`, `tier=jetson`）。

| 项 | 值 |
|---|---|
| 板 | Orin Nano Super Dev Kit，NVMe |
| JetPack / L4T | 6.2 / R36.4.3 |
| TensorRT | 10.3.0 |
| 功耗模式 | `MAXN_SUPER` + `jetson_clocks` |
| SSH | `ubuntu@BOARD_IP_REDACTED`（WiFi `WIFI_SSID_REDACTED`；密码 `root`） |
| 有线（可选） | `BOARD_IP_REDACTED`（插网线时） |
| 远程根 | `/path/to/board/project |
| P5S1 | DC ~15 V 高端串入；仪表 SERIAL_PORT；±3.3 A |

## Phase 1 验收顺序（Step 1.1）

1. SSH / WiFi / `nvpmodel`+`jetson_clocks` → 填 `configs/versions.yaml` `orin_nano`
2. 推送 6 锚点 ONNX（`results/h618_onnx_v4/`）→ `trtexec` 建 FP16（再 INT8）engine
3. `bench/trt_bench.py`：warmup50 / iters1000 → `results/orin_trt_probe.csv`
4. P5S1 能耗矩阵：`python deploy/orin/scripts/run_p5s1_energy_matrix.py --port SERIAL_PORT`
5. （后续）算子 LUT / 早退 C3 真机预算

## 常用命令

```bash
# 锁频
sudo nvpmodel -m 2 && sudo jetson_clocks

# 主机：推模型并建 FP16 engine
python deploy/orin/scripts/build_engines.py --precision fp16

# 主机：延迟矩阵
python deploy/orin/scripts/run_latency_matrix.py --precision fp16

# 主机：能耗（P5S1 已串 Orin DC）
python deploy/orin/scripts/run_p5s1_energy_matrix.py --port SERIAL_PORT --limit 1 --runs 1  # smoke
python deploy/orin/scripts/run_p5s1_energy_matrix.py --port SERIAL_PORT
```

板端直接测：

```bash
python3 /path/to/board/project \
  --engine models/v4_predistill_trt/subnet_..._fp16.plan \
  --h 64 --w 64 --warmup 50 --iters 1000 --device orin
```
