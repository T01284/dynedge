#!/usr/bin/env bash
# Run NCNN probe on-device (T527).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEVICE=t527
PARAM="${PARAM:-${ROOT}/ncnn_edge/models/mobilenetv3_small.param}"
BIN="${BIN:-${ROOT}/ncnn_edge/models/mobilenetv3_small.bin}"
BENCH="${BENCH:-${ROOT}/ncnn_edge/build-ncnn/fer_ncnn_bench}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-1000}"
THREADS="${THREADS:-4}"

if [[ ! -x "$BENCH" ]]; then
  echo "ERR missing bench binary: $BENCH" >&2
  echo "HINT: NCNN_DIR=... ${ROOT}/ncnn_edge/scripts/build_ncnn.sh" >&2
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  "${ROOT}/ncnn_edge/scripts/lock_perf.sh"
else
  echo "WARN not root — skip lock_perf (sudo recommended)"
fi

if [[ ! -f "$PARAM" || ! -f "$BIN" ]]; then
  echo "ERR need model files:" >&2
  echo "  $PARAM" >&2
  echo "  $BIN" >&2
  exit 1
fi

exec "$BENCH" --device "$DEVICE" --param "$PARAM" --bin "$BIN" \
  --threads "$THREADS" --warmup "$WARMUP" --iters "$ITERS"
