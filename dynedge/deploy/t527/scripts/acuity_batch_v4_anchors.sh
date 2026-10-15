#!/usr/bin/env bash
# Host launcher: batch ACUITY convert for T527 v4 Pareto / MCU anchors → NBG.
# Runs inside Docker image (npu_v1.8.13_wang / ubuntu-npu:v1.8.13).
#
# Usage (from Windows / WSL):
#   bash deploy/t527/scripts/acuity_batch_v4_anchors.sh
#   bash deploy/t527/scripts/acuity_batch_v4_anchors.sh --dry-run
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
# Prefer WSL path when launched from /mnt/d
if [[ "$ROOT" == /mnt/* ]]; then
  HOST_ROOT="$ROOT"
else
  HOST_ROOT="<repo>"
  if [[ -d "$HOST_ROOT" ]]; then ROOT="$HOST_ROOT"; fi
fi

ONNX_DIR="${ROOT}/results/h618_onnx_v4"
OUT_HOST="${ROOT}/deploy/t527/models/v4_npu"
CALIB_HOST="${ROOT}/data/calib/linux112/calib"
SCRIPT_INNER="${ROOT}/deploy/t527/acuity/batch_v4_convert_inner.sh"
DOCKER_IMG="${ACUITY_DOCKER_IMG:-ubuntu-npu:v1.8.13}"

# Pareto / MCU anchors from lut/PARETO_CANDIDATES.md (+ mid/high extras used on H618)
# Prefer FERPlus (8-class) for NPU e2e narrative; ExpW optional via --all-ds
ANCHORS=(
  "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64"
  "subnet_ferplus_w0.35_d3333_s64g_seed42_v4_predistill_img64"
  "subnet_ferplus_w0.5_d2222_s96c_seed42_v4_predistill_img96"
  "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_predistill_img112"
  "subnet_ferplus_w0.75_d3333_s112c_seed42_v4_predistill_img112"
  "subnet_ferplus_w1_d4444_s112c_seed42_v4_predistill_img112"
)

DRY=0
ALL_DS=0
LIMIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --all-ds) ALL_DS=1; shift ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --img) DOCKER_IMG="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

if [[ "$ALL_DS" == 1 ]]; then
  mapfile -t ANCHORS < <(ls "$ONNX_DIR"/subnet_*_v4_predistill_img*.onnx | xargs -n1 basename | sed 's/\.onnx$//')
fi
if [[ "$LIMIT" != 0 ]]; then
  ANCHORS=("${ANCHORS[@]:0:$LIMIT}")
fi

mkdir -p "$OUT_HOST"
echo "DOCKER_IMG=$DOCKER_IMG"
echo "anchors=${#ANCHORS[@]}"
for a in "${ANCHORS[@]}"; do
  echo "  - $a"
  test -f "$ONNX_DIR/${a}.onnx" || { echo "missing onnx $a"; exit 1; }
done

if [[ "$DRY" == 1 ]]; then
  echo "dry-run OK"
  exit 0
fi

# Ensure inner script exists and is LF
if [[ ! -f "$SCRIPT_INNER" ]]; then
  echo "missing $SCRIPT_INNER"
  exit 1
fi

# Prefer named image; fall back to ubuntu-npu:v1.8.13
if ! docker image inspect "$DOCKER_IMG" >/dev/null 2>&1; then
  if docker image inspect ubuntu-npu:v1.8.13 >/dev/null 2>&1; then
    DOCKER_IMG=ubuntu-npu:v1.8.13
  fi
fi

docker run --rm \
  -v "${ROOT}:/workspace/lane-edge:rw" \
  -w /workspace \
  "$DOCKER_IMG" \
  bash /workspace/lane-edge/deploy/t527/acuity/batch_v4_convert_inner.sh "${ANCHORS[@]}"

echo "DONE host out=$OUT_HOST"
ls -la "$OUT_HOST" || true
