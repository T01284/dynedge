# H618 H1-fix — bake+pnnx 精度 + 同 artifact 延迟

日期: 2026-07-23  
状态: **完成**

## 根因回顾
- 旧 H1 match≈0.15：非 SE 超参错误，而是 `onnx2ncnn` 把 SE `Unsqueeze` int64 轴写成 **NaN MemoryData**
- 修复：与 T527 T1-fix 相同 — **bake + pnnx**

## 精度（同 artifact）
- `int8_acc_500_baked.csv`：**mean match = 0.9644**（16/16）
- 协议：NCNN FP32 ↔ NCNN INT8 @500；FP16 off

## 延迟（同 artifact 重测）
- 正式协议：4t / warmup=50 / iters=1000
- 现行：`matrix_summary.csv`（bake+pnnx）
- 旧表：`matrix_summary_onnx2ncnn_old.csv`
- 对比：`matrix_baked_vs_onnx2ncnn.csv` — mean |Δ| **4.57%**（FP32≈2.0% / INT8≈7.2%，max 12.6%）
- 结论：H618 上新旧图延迟接近但仍有可测差；**主表必须用 bake 矩阵**

## 与旧 H1
| | 旧 H1（坏图） | H1-fix（bake+pnnx） |
|---|---:|---:|
| mean match | 0.149 | **0.964** |
| 延迟 | 旧 onnx2ncnn 矩阵 | **同 bake 重测矩阵** |
| 可写主表 | 否 | **是** |
