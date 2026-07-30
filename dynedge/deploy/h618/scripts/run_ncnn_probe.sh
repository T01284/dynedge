#!/usr/bin/env bash
# H618 NCNN probe — same binary protocol as T527 ncnn_cpu (official linux_cpu tier).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEVICE="${FER_DEVICE:-h618}"
PARAM="${PARAM:-${ROOT}/ncnn_edge/models/mobilenetv3_small.param}"
BIN="${BIN:-${ROOT}/ncnn_edge/models/mobilenetv3_small.bin}"
BENCH="${FER_NCNN_BENCH:-${ROOT}/ncnn_edge/build-ncnn/fer_ncnn_bench}"
WARMUP="${FER_WARMUP:-50}"
ITERS="${FER_ITERS:-200}"
THREADS="${FER_THREADS:-4}"

if [[ ! -x "$BENCH" ]]; then
  echo "ERR missing bench binary: $BENCH" >&2
  echo "Build: NCNN_DIR=... bash ${ROOT}/ncnn_edge/scripts/build_ncnn.sh" >&2
  exit 1
fi

if [[ ! -f "$PARAM" || ! -f "$BIN" ]]; then
  echo "ERR need $PARAM and $BIN" >&2
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  "${ROOT}/ncnn_edge/scripts/lock_perf.sh"
else
  echo "WARN not root — skip lock_perf"
fi

exec "$BENCH" \
  --device "$DEVICE" \
  --param "$PARAM" \
  --bin "$BIN" \
  --threads "$THREADS" \
  --warmup "$WARMUP" \
  --iters "$ITERS"
