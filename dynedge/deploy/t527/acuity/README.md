# VIPLite NPU / 公开 ACUITY（T527）

> 策略：Android 先做 NCNN；刷 Linux 后接 NPU；公开 ACUITY 到手后可 **重做** 已完成实验。  
> **不要提交** 专有 SDK 二进制。

## 运行时（板端，Android 已具备）

| 组件 | Android 实测路径 |
|---|---|
| 设备节点 | `/dev/vipcore` |
| 驱动 | VIPLite 1.13.0.0-AW-2023-07-04（dmesg） |
| 用户态 | `/vendor/lib64/libVIPlite.so`, `libVIPuser.so` |
| 其它 | `libfaceid.viplite_v2.so`（人脸 demo，非通用 FER） |

Linux 刷机后应对齐同名 VIPLite API；缺头文件/示例时从公开 BSP / Radxa Cubie 文档补齐。

## 离线工具（x86 主机）

公开 **ACUITY Toolkit**（全志/VeriSilicon 公开包，非必须米尔私有）：

```
ONNX (opset 17)
  → ACUITY quantize (INT8, ~500 calib, 112×112)
  → ACUITY compile → NBG
  → VIPLite load NBG → 板上计时
```

| Item | 值 |
|---|---|
| ACUITY 来源 / 版本 | 全志公开包 → Docker `ubuntu-npu:v1.8.13`；ACUITY **6.45.8** + Vivante IDE **5.8.2** |
| 容器 | `npu_v1.8.13_wang`；工作区 `/workspace/fer_mnv3`（宿主 `/home/wang/docker_data/fer_mnv3`） |
| optimize | `VIP9000NANOSI_PLUS_PID0X10000016`（对齐板上 cid） |
| 探路 NBG | `deploy/t527/models/mobilenetv3_small_fer_uint8.nb` |
| 板上跑 | `scripts/run_linux_npu_mnv3_fer.py` → `results/t527_linux_npu_mnv3_fer_probe.csv`（p50≈1450 µs） |
| 转换脚本 | `acuity/phase1_import.sh` + `acuity/phase2_quantize_export.sh` |

```bash
# 容器内（已挂载 /workspace）
bash /tmp/phase1_import.sh          # onnx → json/data + inputmeta
bash /tmp/phase2_quantize_export.sh # uint8 quantize → network_binary.nb
```

## 与 NCNN 的关系

- **NCNN CPU**：对照与回退；Android 过渡 CSV 已有；Linux 重测后作正式对照行。
- **VIPLite NPU**：主叙事；无 ACUITY 时不阻塞主表 CPU 行。
- 论文数字：**优先 Linux 重测**；Android 仅 transitional。
