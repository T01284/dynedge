# Shared NCNN edge bench (T527 / H618)

C++ CLI emitting the same line protocol as MCU firmware (`READY` / `RESULT` / `MEM` / `BENCH_DONE`).

## Quick start (no board / no NCNN)

```bash
cd deploy/ncnn_edge
bash scripts/build_stub.sh
```

## On-device (with NCNN)

```bash
export NCNN_DIR=/opt/ncnn/install   # aarch64 prefix
bash scripts/build_ncnn.sh
# copy fer_ncnn_bench + models/*.param|bin to the board
sudo bash ../t527/scripts/run_ncnn_probe.sh
# or (official linux_cpu tier)
sudo bash ../h618/scripts/run_ncnn_probe.sh
```

## Model export (probe)

Phase 1 probe: **MobileNetV3-Small**，输入 112×112×3，输出 7-logit（ExpW）。  
导出 ncnn `param`/`bin`（或 onnx2ncnn）放到 `models/`：

```
models/mobilenetv3_small.param
models/mobilenetv3_small.bin
```

默认 blob 名：`input` / `output`（可用编译宏 `FER_NCNN_OUTPUT` 改输出名）。

## Host collector

```bash
python bench/linux_bench.py --cmd "./fer_ncnn_bench --device h618 ..." --out results/h618_ncnn.csv
# or SSH:
python bench/linux_bench.py --ssh root@BOARD_IP_REDACTED --remote-cmd "..." --device h618 --out results/h618_ncnn_probe.csv
```
