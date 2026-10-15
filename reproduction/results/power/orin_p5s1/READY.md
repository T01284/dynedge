# Orin P5S1 ready

- Engines: `<gpu-workdir>/fer-dynedge/models/v4_predistill_trt/*_fp16.plan`（6/6）
- Bench: `<gpu-workdir>/fer-dynedge/bench/trt_bench.py`
- Latency: `results/orin_trt_matrix.csv`（§8.26）
- Energy: `results/power/orin_p5s1/energy_matrix.csv`（18/18 trt_fp16）
- Meter: Orin DC ~15 V high-side，COM40，±3.3 A
- SSH: `ubuntu@redacted`（WiFi）；工况 MAXN_SUPER + jetson_clocks
