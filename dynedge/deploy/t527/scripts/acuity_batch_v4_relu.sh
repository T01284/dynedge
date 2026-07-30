#!/usr/bin/env bash
# T2b: ACUITY convert G4 ReLU ONNX → uint8 NBG (separate from T2a Hardswish).
# Usage (WSL): bash deploy/t527/scripts/acuity_batch_v4_relu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
if [[ "$ROOT" == /mnt/* ]]; then
  :
else
  HOST_ROOT="/mnt/d/paper/lane-edge"
  if [[ -d "$HOST_ROOT" ]]; then ROOT="$HOST_ROOT"; fi
fi

export ONNX_DIR="${ROOT}/results/t527_onnx_v4_relu"
export OUT_HOST="${ROOT}/deploy/t527/models/v4_npu_relu"
DOCKER_IMG="${ACUITY_DOCKER_IMG:-ubuntu-npu:v1.8.13}"

ANCHORS=(
  "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_relu_npu_img64"
  "subnet_ferplus_w0.35_d3333_s64g_seed42_v4_relu_npu_img64"
  "subnet_ferplus_w0.5_d2222_s96c_seed42_v4_relu_npu_img96"
  "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_relu_npu_img112"
  "subnet_ferplus_w0.75_d3333_s112c_seed42_v4_relu_npu_img112"
  "subnet_ferplus_w1_d4444_s112c_seed42_v4_relu_npu_img112"
)

mkdir -p "$OUT_HOST"
echo "DOCKER_IMG=$DOCKER_IMG ONNX_DIR=$ONNX_DIR OUT_HOST=$OUT_HOST"
for a in "${ANCHORS[@]}"; do
  echo "  - $a"
  test -f "$ONNX_DIR/${a}.onnx" || { echo "missing onnx $a"; exit 1; }
done

if ! docker image inspect "$DOCKER_IMG" >/dev/null 2>&1; then
  DOCKER_IMG=ubuntu-npu:v1.8.13
fi

# Ensure LF endings for inner script when edited on Windows
sed -i 's/\r$//' "$ROOT/deploy/t527/acuity/batch_v4_convert_inner.sh" || true

docker run --rm \
  -e ONNX_DIR="/workspace/lane-edge/results/t527_onnx_v4_relu" \
  -e OUT_HOST="/workspace/lane-edge/deploy/t527/models/v4_npu_relu" \
  -v "${ROOT}:/workspace/lane-edge:rw" \
  -w /workspace \
  "$DOCKER_IMG" \
  bash /workspace/lane-edge/deploy/t527/acuity/batch_v4_convert_inner.sh "${ANCHORS[@]}"

echo "DONE host out=$OUT_HOST"
ls -la "$OUT_HOST" || true
