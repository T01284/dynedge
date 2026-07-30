# T527 NPU 算子支持清单（Step 1.2.3）

> 用**公开 ACUITY**（或等价 VIP9000 工具）依次转换含下列算子的小网络：ONNX → 量化 → NBG → VIPLite 板上跑。  
> 结果约束 Phase 2 超网搜索空间。填完后冻结。  
> **测量 OS**：以 Linux 正式数字为准；Android 仅作驱动存在性确认。

| 算子 | 支持? (Y/N/Partial) | 回退路径 | 备注 / 版本 | 日期 |
|---|---|---|---|---|
| Conv2d | Y | — | MNV3-FER 全图 NBG 通过 | 2026-07-23 |
| DepthwiseConv | Y | — | IR blocks dwconv | 2026-07-23 |
| FullyConnected / Gemm | Y | — | SE + classifier | 2026-07-23 |
| ReLU | Y | — | 出现在部分 IR 投影 | 2026-07-23 |
| ReLU6 | — | — | 本探路模型未覆盖 | |
| HardSwish | Y | — | via HardSigmoid×Mul | 2026-07-23 |
| HardSigmoid | Y | — | stem / SE | 2026-07-23 |
| GELU | — | — | 本探路模型未覆盖 | |
| BatchNorm (fused?) | Y | — | 导出侧已融合进 Conv | 2026-07-23 |
| LayerNorm | — | — | 本探路模型未覆盖 | |
| Softmax | — | — | 输出为 7-logit，未含 Softmax | |
| GlobalAveragePool | Y | — | SE / head | 2026-07-23 |
| MaxPool | — | — | 本探路模型未覆盖 | |
| Add (residual) | Y | — | inverted residual | 2026-07-23 |
| Mul | Y | — | HardSwish / SE | 2026-07-23 |
| Concat | — | — | 本探路模型未覆盖 | |
| Resize / Upsample | — | — | 本探路模型未覆盖 | |
| MultiHeadAttention | — | — | 本探路模型未覆盖 | |
| 动态 shape | N | 固定 1×3×112×112 | NBG 静态 shape | 2026-07-23 |

## 探路模型结论

- MobileNetV3-Small @112：NPU **全图成功**（`vpm_run` 加载 `network_binary.nb`，无算子回退报错）
- 首次成功的 ACUITY 版本：**Vivante ML Toolkit ACUITY whl 6.45.8** + Vivante IDE 5.8.2；optimize=`VIP9000NANOSI_PLUS_PID0X10000016`；板上 VIPLite **1.13** / cid=`0x10000016`
- 延迟（Linux，`vpm_run -l 200 -b 1`，丢前 10）：
  - 合成校准：**p50 ≈ 1485 µs** → `results/t527_linux_npu_mnv3_fer_probe.csv`
  - **真图校准（FER+/ExpW 128）**：**p50 ≈ 1449 µs** → `results/t527_linux_npu_mnv3_fer_real_probe.csv`
- 对照 NCNN Linux formal@4t：FP32 ≈ 3823 µs；INT8 real ≈ 6772 µs
- 算子微基准（NCNN）：`results/t527_linux_op_smoke.csv`（19/21；add 两项因单输入 bench 未出数）
- 汇总：`results/t527_linux_latency_SUMMARY.md`
- 精度对齐：`results/t527_npu_vs_ncnn_SUMMARY.md`（NPU↔FP32 ImageNet match=**0.765**；NCNN INT8↔FP32=**0.82**）
- 全量 LUT：`lut/ops_t527_v4_full.csv`（**706/706**；见 `lut/ops_t527_SUMMARY.md`）
- 不可用算子导致的超网禁区：暂无（MNV3 所需算子均过）；GELU / MHA / 动态 shape 仍未单测，搜索空间勿默认放开

## 回退开销案例（§4.5）

（尚未构造「单算子强制 CPU 回退」对比；全图 NPU 路径已通。）
