# T527 全量算子 LUT（v4 enum，对标 H618）

- 枚举：`lut/enum_v4_full.json` 中 `backend=h618_ncnn` 的 conv/dwconv/gemm/add/gap × fp32/int8
- 结果：`lut/ops_t527_v4_full.csv` — **706/706 OK**
- 条件：Buildroot；A55 performance @1416 MHz；`fer_op_bench` 静态链接；threads=4；warmup10/iters80
- INT8：主机 WSL `ncnn2table`/`ncnn2int8` 预量化；gemm/add/gap 的 int8 行为 **fp32_proxy**（88 项）

## 分算子延迟中位数（µs）

| op | fp32 med | int8 med |
|---|---:|---:|
| conv | 288.5 | 298.2 |
| dwconv | 198.4 | 326.4 |
| gemm | 26.0 | 26.0† |
| add | 76.7 | 72.4† |
| gap | 68.0 | 61.9† |

† int8 行为 fp32_proxy。

## 与 H618 对齐键共有 618 项

整体 **T527/H618 中位延迟比 ≈ 1.32×**（同 shape；时钟/BSP 不同，不作绝对性能结论，仅供 LUT 拟合）。

| op/dtype | n | med(T527/H618) |
|---|---:|---:|
| conv/fp32 | 176 | 1.23 |
| conv/int8 | 176 | 1.35 |
| dwconv/fp32 | 89 | 1.78 |
| dwconv/int8 | 89 | 1.12 |
| gemm/fp32 | 36 | 1.16 |
| gemm/int8 | 36 | 1.14 |

脚本：`deploy/t527/scripts/mine_ops_t527_full.py`、`pull_t527_ops_mine.py`
