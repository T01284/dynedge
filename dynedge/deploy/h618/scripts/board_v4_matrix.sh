#!/bin/bash
# H618: convert + FP32/INT8 bench for v4_predistill ONNX (variable HxW from filename).
set -euo pipefail
export PATH=/opt/ncnn/install/bin:$PATH
ROOT=/root/fer-dynedge
NCNN_EDGE=$ROOT/deploy/ncnn_edge
V4_DIR=$NCNN_EDGE/models/v4_predistill
OUT=$ROOT/results/overnight_v4
CALIB112=$NCNN_EDGE/models/imagelist_real.txt
CALIB64=$NCNN_EDGE/models/imagelist_mcu64.txt
mkdir -p "$OUT" "$V4_DIR"
STATUS=$OUT/v4_status.txt
LOG=$OUT/v4_matrix.log
exec > >(tee -a "$LOG") 2>&1
echo "===== V4 MATRIX START $(date -Iseconds) =====" | tee "$STATUS"

cd "$NCNN_EDGE"
export NCNN_DIR=/opt/ncnn/install
cmake -S . -B build-ncnn8 -DFER_HAS_NCNN=ON -DNCNN_DIR="$NCNN_DIR" -DFER_NUM_CLASSES=8 >/dev/null
cmake --build build-ncnn8 -j"$(nproc)"
cmake -S . -B build-ncnn7 -DFER_HAS_NCNN=ON -DNCNN_DIR="$NCNN_DIR" -DFER_NUM_CLASSES=7 >/dev/null
cmake --build build-ncnn7 -j"$(nproc)"
BENCH8=$NCNN_EDGE/build-ncnn8/fer_ncnn_bench
BENCH7=$NCNN_EDGE/build-ncnn7/fer_ncnn_bench

if [[ ! -f "$CALIB112" ]]; then
  find /root/fer-dynedge/data/calib/linux112/calib -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | sort > "$CALIB112" || true
fi
if [[ ! -f "$CALIB64" ]]; then
  find /root/fer-dynedge/data/calib/ferplus_mcu64 -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | sort | head -200 > "$CALIB64" || cp -f "$CALIB112" "$CALIB64"
fi

shape_for() {
  local stem=$1
  if [[ "$stem" == *img64* || "$stem" == *_s64g_* ]]; then echo "[64,64,3]"; 
  elif [[ "$stem" == *img96* || "$stem" == *_s96c_* ]]; then echo "[96,96,3]";
  else echo "[112,112,3]"; fi
}
hwc_for() {
  # prints: H W C
  local stem=$1
  if [[ "$stem" == *img64* || "$stem" == *_s64g_* ]]; then echo "64 64 3";
  elif [[ "$stem" == *img96* || "$stem" == *_s96c_* ]]; then echo "96 96 3";
  else echo "112 112 3"; fi
}
calib_for() {
  local stem=$1
  if [[ "$stem" == *img64* || "$stem" == *_s64g_* ]]; then echo "$CALIB64"; else echo "$CALIB112"; fi
}

mark() { echo "$1" | tee -a "$STATUS"; }

process_one() {
  local onnx=$1
  local stem; stem=$(basename "$onnx" .onnx)
  local param="$V4_DIR/${stem}.param"
  local bin="$V4_DIR/${stem}.bin"
  local iparam="$V4_DIR/${stem}_int8.param"
  local ibin="$V4_DIR/${stem}_int8.bin"
  local table="$V4_DIR/${stem}.table"
  local fp32_txt="$OUT/${stem}_fp32.txt"
  local int8_txt="$OUT/${stem}_int8.txt"
  local shape; shape=$(shape_for "$stem")
  local calib; calib=$(calib_for "$stem")
  local hwc; hwc=$(hwc_for "$stem")
  read -r IN_H IN_W IN_C <<< "$hwc"
  local BENCH=$BENCH8
  [[ "$stem" == *expw* ]] && BENCH=$BENCH7

  if [[ ! -f "$param" ]]; then
    echo "onnx2ncnn $stem"
    onnx2ncnn "$onnx" "$param" "$bin" || { mark "FAIL onnx2ncnn $stem"; return 1; }
    ncnnoptimize "$param" "$bin" "$param" "$bin" 0 || true
  fi

  if [[ ! -f "$fp32_txt" ]]; then
    echo "===== BENCH FP32 $stem input=${IN_H}x${IN_W}x${IN_C} ====="
    "$BENCH" --device h618 --param "$param" --bin "$bin" \
      --h "$IN_H" --w "$IN_W" --c "$IN_C" \
      --threads 4 --warmup 50 --iters 1000 | tee "$fp32_txt"
    grep -q '^RESULT' "$fp32_txt" || { mark "FAIL fp32 $stem"; return 1; }
  fi

  if [[ ! -f "$iparam" || ! -f "$ibin" ]]; then
    echo "===== ncnn2table $stem shape=$shape ====="
    rm -f "$table"
    shopt -u nullglob
    ncnn2table "$param" "$bin" "$calib" "$table" \
      "mean=[0,0,0]" "norm=[0.00392157,0.00392157,0.00392157]" "shape=${shape}" pixel=RGB thread=2 method=kl \
      || ncnn2table "$param" "$bin" "$calib" "$table" \
      "mean=[0,0,0]" "norm=[1,1,1]" "shape=${shape}" pixel=RGB thread=2 method=kl
    [[ -s "$table" ]] || { mark "FAIL table $stem"; return 1; }
    ncnn2int8 "$param" "$bin" "$iparam" "$ibin" "$table"
    [[ -s "$iparam" && -s "$ibin" ]] || { mark "FAIL int8 $stem"; return 1; }
  fi

  if [[ ! -f "$int8_txt" ]]; then
    echo "===== BENCH INT8 $stem input=${IN_H}x${IN_W}x${IN_C} ====="
    "$BENCH" --device h618 --param "$iparam" --bin "$ibin" \
      --h "$IN_H" --w "$IN_W" --c "$IN_C" \
      --threads 4 --warmup 50 --iters 1000 | tee "$int8_txt"
    grep -q '^RESULT' "$int8_txt" || { mark "FAIL int8bench $stem"; return 1; }
  fi
  mark "OK $stem"
}

shopt -s nullglob
ons=("$V4_DIR"/*.onnx)
echo "onnx count=${#ons[@]}"
shopt -u nullglob
ok=0; fail=0
for onnx in "${ons[@]}"; do
  if process_one "$onnx"; then ok=$((ok+1)); else fail=$((fail+1)); fi
done
echo "===== V4 MATRIX DONE $(date -Iseconds) ok=$ok fail=$fail =====" | tee -a "$STATUS"
grep -h '^RESULT' "$OUT"/*_fp32.txt "$OUT"/*_int8.txt 2>/dev/null | tee "$OUT/all_v4_results.txt" || true
