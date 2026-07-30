#!/usr/bin/env bash
# Convert MobileNetV3-Small FER ONNX -> NBG inside ACUITY docker.
set -euo pipefail
CONTAINER=npu_v1.8.13_wang
WORKDIR=/workspace/mnv3_fer
PEG=python3:/root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/pegasus.py
# actually:
PEG_CMD="python3 /root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/pegasus.py"

docker exec "$CONTAINER" bash -lc "
set -e
cd $WORKDIR
ls -la
export ACUITY_PATH=/root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin
export PATH=\$ACUITY_PATH:\$PATH
$PEG_CMD help import 2>&1 | head -n 80
echo '==== onnx help ===='
$PEG_CMD import onnx -h 2>&1 | head -n 80 || true
$PEG_CMD help 2>&1 | head -n 5
"
