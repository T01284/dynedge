"""Unified bench CSV schema spanning MCU + Linux edge devices.

All collectors should emit these columns (use empty string / -1 when N/A)
so ``results/*_probe.csv`` can be concatenated into one matrix for the paper.
"""
from __future__ import annotations

# Common latency / identity fields (always filled when possible)
COMMON_FIELDS = [
    "device",
    "tier",  # MCU | linux_cpu | linux_npu | jetson | gpu
    "isa",  # xtensa_lx7 | riscv_pie | armv7 | aarch64 | ...
    "backend",
    "model",
    "fw",
    "input",
    "kind",
    "warmup",
    "iters",
    "latency_us_mean",
    "latency_us_p50",
    "latency_us_p95",
    "latency_us_p99",
    "pred",
    "class",
    "conf",
]

# MCU-only memory
MCU_FIELDS = [
    "heap_free",
    "psram_free",
    "psram_ok",
    "weights_bytes",
    "act_bytes",
]

# Linux process memory
LINUX_FIELDS = [
    "rss_kb",
    "vms_kb",
]

# Board-level energy (P5S1 meter; GPU server uses nvidia-smi instead)
ENERGY_FIELDS = [
    "power_w_mean",
    "power_w_idle",
    "energy_mj_per_infer",  # prefer net (active - idle*T) when available
    "energy_mj_per_infer_gross",
    "n_power_samples",
    "power_meter",  # p5s1 | nvidia_smi | ina_onboard | ...
]

CSV_FIELDS = COMMON_FIELDS + MCU_FIELDS + LINUX_FIELDS + ENERGY_FIELDS

TIER_BY_DEVICE = {
    "esp32s3": "MCU",
    "esp32p4": "MCU",
    "t113": "linux_cpu",  # legacy / abandoned; official tier is h618
    "h618": "linux_cpu",
    "t527": "linux_npu",
    "orin": "jetson",
    "gpu": "gpu",
}

ISA_BY_DEVICE = {
    "esp32s3": "xtensa_lx7",
    "esp32p4": "riscv_pie",
    "t113": "arm",  # legacy stub only
    "h618": "aarch64",
    "t527": "aarch64",
    "orin": "aarch64",
}


def empty_row() -> dict:
    return {k: "" for k in CSV_FIELDS}
