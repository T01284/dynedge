#!/bin/bash
set -euo pipefail
ONNX2NCNN=/opt/src/ncnn/build/tools/onnx/onnx2ncnn
cp -f "$ONNX2NCNN" /opt/ncnn/install/bin/onnx2ncnn
cd /root/fer-dynedge/deploy/ncnn_edge/models
"$ONNX2NCNN" mobilenetv3_small.onnx mobilenetv3_small.param mobilenetv3_small.bin
# optional optimize
if [[ -x /opt/ncnn/install/bin/ncnnoptimize ]]; then
  /opt/ncnn/install/bin/ncnnoptimize mobilenetv3_small.param mobilenetv3_small.bin \
    mobilenetv3_small_opt.param mobilenetv3_small_opt.bin 0 || true
  if [[ -f mobilenetv3_small_opt.param ]]; then
    mv -f mobilenetv3_small_opt.param mobilenetv3_small.param
    mv -f mobilenetv3_small_opt.bin mobilenetv3_small.bin
    echo "optimized model applied"
  fi
fi
ls -la mobilenetv3_small.param mobilenetv3_small.bin
echo "=== param head ==="
head -12 mobilenetv3_small.param
# Fix blob names if needed — expect Input named input, last extract as output
grep -nE '^Input |^.* output|input' mobilenetv3_small.param | head -30

echo "===== build fer_ncnn_bench ====="
cd /root/fer-dynedge/deploy/ncnn_edge
export NCNN_DIR=/opt/ncnn/install
bash scripts/build_ncnn.sh
ls -la build-ncnn/fer_ncnn_bench

echo "===== lock perf + smoke probe ====="
bash scripts/lock_perf.sh
bash /root/fer-dynedge/deploy/h618/scripts/run_ncnn_probe.sh
