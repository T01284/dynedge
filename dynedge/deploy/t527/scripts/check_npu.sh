#!/usr/bin/env bash
# T527: verify NPU / VIP driver presence (BSP-dependent strings).
set -euo pipefail

echo "=== uname ==="
uname -a

echo "=== dmesg NPU/VIP hints ==="
if dmesg 2>/dev/null | grep -iE 'vip|npu|galcore|vsi' | tail -n 40; then
  :
else
  echo "WARN no dmesg match for vip|npu|galcore|vsi"
fi

echo "=== module / device nodes ==="
lsmod 2>/dev/null | grep -iE 'vip|galcore|npu' || echo "WARN no matching lsmod"
ls -l /dev/*vip* /dev/*npu* /dev/galcore 2>/dev/null || echo "WARN no /dev NPU nodes"

echo "=== CPU ==="
grep -E 'model name|Features|CPU part' /proc/cpuinfo | head -n 20 || true
nproc

echo "DONE check_npu — fill configs/versions.yaml t527.bsp when green"
