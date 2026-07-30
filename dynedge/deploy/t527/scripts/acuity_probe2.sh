#!/usr/bin/env bash
set -euo pipefail
docker exec npu_v1.8.13_wang bash -lc '
PEG="python3 /root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/pegasus.py"
echo "==== acuity.spec snippets ===="
grep -iE "VIP|PID0|optimize|T527|9000|device" /root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/acuity.spec 2>/dev/null | head -n 50 || true
echo "==== IDE ===="
ls /root/Vivante_IDE/
ls /root/Vivante_IDE/*/cmdtools 2>/dev/null | head || true
echo "==== export ovxlib help ===="
$PEG export ovxlib -h 2>&1 | head -n 100
echo "==== quantize help ===="
$PEG quantize -h 2>&1 | head -n 80
'
