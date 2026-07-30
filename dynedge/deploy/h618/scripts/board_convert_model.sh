#!/bin/bash
set -euo pipefail
export PATH=/opt/ncnn/install/bin:$PATH
ls /opt/ncnn/install/bin
cd /root/fer-dynedge/deploy/ncnn_edge/models
ONNX2NCNN=""
if command -v onnx2ncnn >/dev/null; then
  ONNX2NCNN=$(command -v onnx2ncnn)
elif [[ -x /opt/src/ncnn/build/tools/onnx/onnx2ncnn ]]; then
  ONNX2NCNN=/opt/src/ncnn/build/tools/onnx/onnx2ncnn
fi
if [[ -z "$ONNX2NCNN" ]]; then
  echo "NO onnx2ncnn"
  find /opt/src/ncnn/build/tools -type f -executable 2>/dev/null | head -40
  exit 2
fi
echo "using $ONNX2NCNN"
"$ONNX2NCNN" mobilenetv3_small.onnx mobilenetv3_small.param mobilenetv3_small.bin
ls -la mobilenetv3_small.*
echo "=== param head ==="
head -8 mobilenetv3_small.param
# show input/output layer names
grep -E 'Input|input|output' mobilenetv3_small.param | head -20
