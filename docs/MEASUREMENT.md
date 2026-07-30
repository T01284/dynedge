# Measurement protocol (stub — expand to full supplementary text)

This file will host the publishable measurement protocol used for DynEdge:

- Latency: batch=1, real images, warmup excluded; report P50 (and P99 when available).
- H618 / T527 NCNN: threads=4; e2e warmup 50 / iters 1000; op mine warmup 10 / iters 80.
- T527 VIPLite NBG: on-device end-to-end loops.
- Orin TensorRT: MAXN_SUPER + clocks locked; warmup 50 / iters 1000.
- ESP32-S3 / P4 (ESP-DL): on-device `esp_timer`; logits gate 8/8 vs host.
- Energy: P5S1 shunt, 100 Hz, ±3.3 A / 0.01 mA (see `dynedge/bench/p5s1/`).

Wiring photographs (faces / SSIDs / rack labels redacted) will be added after release scrubbing.
Until then, follow the Supplementary Material §Latency / §Energy sections and `dynedge/bench/p5s1/README.md`.
