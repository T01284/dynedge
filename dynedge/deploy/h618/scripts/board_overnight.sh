#!/bin/bash
# H618 overnight: benchncnn suite + v3 subnet matrix (FP32/INT8) when ONNX arrive.
set -euo pipefail
export PATH=/opt/ncnn/install/bin:/opt/src/ncnn/build/benchmark:$PATH
ROOT=/root/fer-dynedge
NCNN_EDGE=$ROOT/deploy/ncnn_edge
MODELS=$NCNN_EDGE/models
V3_DIR=$MODELS/v3_wide
OUT=$ROOT/results/overnight
CALIB_LIST=$MODELS/imagelist_real.txt
mkdir -p "$OUT" "$V3_DIR" "$MODELS"
LOG=$OUT/overnight.log
exec > >(tee -a "$LOG") 2>&1
echo "===== OVERNIGHT START $(date -Iseconds) ====="

bash "$NCNN_EDGE/scripts/lock_perf.sh" || true

# --- monitor helper ---
monitor() {
  {
    echo "---- $(date -Iseconds) ----"
    for f in /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq \
             /sys/class/thermal/thermal_zone*/temp; do
      [[ -f "$f" ]] && echo "$f=$(cat "$f")"
    done
    free -h | head -2
  } >> "$OUT/monitor.log"
}

# ========== Phase A: benchncnn soak (long) ==========
echo "===== Phase A benchncnn ====="
BENCHNCNN=/opt/src/ncnn/build/benchmark/benchncnn
cp -f "$BENCHNCNN" /usr/local/bin/benchncnn || true
cd /opt/src/ncnn/benchmark/models
# rounds × threads — designed ~several hours
for round in 1 2 3 4 5; do
  for th in 1 2 4; do
    monitor
    echo "benchncnn round=$round threads=$th"
    # loop_count=8 cooling=1
    benchncnn 8 "$th" 0 -1 1 | tee -a "$OUT/benchncnn_r${round}_t${th}.log" || true
  done
done
echo "===== Phase A DONE $(date -Iseconds) ====="

# ========== Phase B: wait / process v3 ONNX ==========
echo "===== Phase B v3 subnet matrix ====="
# Rebuild bench for 8-class FERPlus (ExpW models still ok if 7 — separate later)
cd "$NCNN_EDGE"
export NCNN_DIR=/opt/ncnn/install
cmake -S . -B build-ncnn8 -DFER_HAS_NCNN=ON -DNCNN_DIR="$NCNN_DIR" -DFER_NUM_CLASSES=8
cmake --build build-ncnn8 -j"$(nproc)"
BENCH8=$NCNN_EDGE/build-ncnn8/fer_ncnn_bench
# keep 7-class for ExpW
cmake -S . -B build-ncnn7 -DFER_HAS_NCNN=ON -DNCNN_DIR="$NCNN_DIR" -DFER_NUM_CLASSES=7
cmake --build build-ncnn7 -j"$(nproc)"
BENCH7=$NCNN_EDGE/build-ncnn7/fer_ncnn_bench

# imagelist for INT8
if [[ ! -f "$CALIB_LIST" ]]; then
  find /root/fer-dynedge/data/calib/linux112/calib -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | sort > "$CALIB_LIST" || true
fi

process_onnx() {
  local onnx=$1
  local stem
  stem=$(basename "$onnx" .onnx)
  local param="$V3_DIR/${stem}.param"
  local bin="$V3_DIR/${stem}.bin"
  local iparam="$V3_DIR/${stem}_int8.param"
  local ibin="$V3_DIR/${stem}_int8.bin"
  local table="$V3_DIR/${stem}.table"

  if [[ ! -f "$param" ]]; then
    echo "onnx2ncnn $stem"
    onnx2ncnn "$onnx" "$param" "$bin"
    ncnnoptimize "$param" "$bin" "$param" "$bin" 0 || true
  fi

  # pick bench binary
  local BENCH=$BENCH8
  if [[ "$stem" == *expw* ]]; then
    BENCH=$BENCH7
  fi

  # FP32 formal
  if [[ ! -f "$OUT/${stem}_fp32.txt" ]]; then
    monitor
    "$BENCH" --device h618 --param "$param" --bin "$bin" --threads 4 --warmup 50 --iters 1000 \
      | tee "$OUT/${stem}_fp32.txt"
  fi

  # INT8 if calib list exists
  # Quote mean/norm/shape — unquoted [0,0,0] is a glob; with nullglob it vanishes.
  if [[ -s "$CALIB_LIST" && ! -f "$iparam" ]]; then
    echo "INT8 calib $stem"
    ncnn2table "$param" "$bin" "$CALIB_LIST" "$table" \
      "mean=[0,0,0]" "norm=[0.00392157,0.00392157,0.00392157]" "shape=[112,112,3]" pixel=RGB thread=2 method=kl \
      || ncnn2table "$param" "$bin" "$CALIB_LIST" "$table" \
           "mean=[0,0,0]" "norm=[1,1,1]" "shape=[112,112,3]" pixel=RGB thread=2 method=kl
    if [[ -s "$table" ]]; then
      ncnn2int8 "$param" "$bin" "$iparam" "$ibin" "$table"
    else
      echo "WARN no table for $stem — skip int8"
    fi
  fi
  if [[ -f "$iparam" && ! -f "$OUT/${stem}_int8.txt" ]]; then
    monitor
    "$BENCH" --device h618 --param "$iparam" --bin "$ibin" --threads 4 --warmup 50 --iters 1000 \
      | tee "$OUT/${stem}_int8.txt"
  fi
}

# Poll for onnx drops (host uploads to $V3_DIR/*.onnx)
DEADLINE=$((SECONDS + 10*3600))  # up to 10h waiting/processing
while (( SECONDS < DEADLINE )); do
  shopt -s nullglob
  ons=("$V3_DIR"/*.onnx)
  if (( ${#ons[@]} > 0 )); then
    for onnx in "${ons[@]}"; do
      process_onnx "$onnx" || echo "WARN fail $onnx"
    done
  fi
  # if export marker present and all processed, break
  if [[ -f "$V3_DIR/EXPORT_DONE" ]]; then
    pending=0
    for onnx in "$V3_DIR"/*.onnx; do
      stem=$(basename "$onnx" .onnx)
      [[ -f "$OUT/${stem}_fp32.txt" ]] || pending=1
    done
    if (( pending == 0 )); then
      echo "all v3 onnx processed"
      break
    fi
  fi
  echo "waiting onnx... $(date -Iseconds) count=${#ons[@]}"
  sleep 120
done

echo "===== OVERNIGHT DONE $(date -Iseconds) ====="
ls -la "$OUT" | tee "$OUT/listing.txt"
grep -h '^RESULT' "$OUT"/*.txt 2>/dev/null | tee "$OUT/all_results.txt" || true
