#!/bin/bash
# H618: INT8 PTQ + formal bench for all v3_wide subnets (FP32 already done).
# Fix vs overnight: verify .table exists before ncnn2int8; don't abort whole queue on one fail.
set +e
export PATH=/opt/ncnn/install/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
ROOT=/root/fer-dynedge
NCNN_EDGE=$ROOT/deploy/ncnn_edge
MODELS=$NCNN_EDGE/models
V3_DIR=$MODELS/v3_wide
OUT=$ROOT/results/overnight
CALIB_LIST=$MODELS/imagelist_real.txt
LOG=$OUT/int8_v3.log
mkdir -p "$OUT" "$V3_DIR"

exec > >(tee -a "$LOG") 2>&1
echo "===== INT8 v3 START $(date -Iseconds) ====="
bash "$NCNN_EDGE/scripts/lock_perf.sh" || true

# ensure calib list
if [[ ! -s "$CALIB_LIST" ]]; then
  find "$ROOT/data/calib/linux112/calib" -type f \( -iname '*.png' -o -iname '*.jpg' \) 2>/dev/null | sort > "$CALIB_LIST"
fi
echo "calib list: $(wc -l < "$CALIB_LIST") lines -> $CALIB_LIST"
head -2 "$CALIB_LIST"
which ncnn2table ncnn2int8

# bench binaries (prefer overnight builds)
BENCH8=$NCNN_EDGE/build-ncnn8/fer_ncnn_bench
BENCH7=$NCNN_EDGE/build-ncnn7/fer_ncnn_bench
if [[ ! -x "$BENCH8" ]]; then
  echo "rebuild bench8"
  cmake -S "$NCNN_EDGE" -B "$NCNN_EDGE/build-ncnn8" -DFER_HAS_NCNN=ON -DNCNN_DIR=/opt/ncnn/install -DFER_NUM_CLASSES=8
  cmake --build "$NCNN_EDGE/build-ncnn8" -j"$(nproc)"
fi
if [[ ! -x "$BENCH7" ]]; then
  echo "rebuild bench7"
  cmake -S "$NCNN_EDGE" -B "$NCNN_EDGE/build-ncnn7" -DFER_HAS_NCNN=ON -DNCNN_DIR=/opt/ncnn/install -DFER_NUM_CLASSES=7
  cmake --build "$NCNN_EDGE/build-ncnn7" -j"$(nproc)"
fi

STATUS=$OUT/int8_v3_status.txt
: > "$STATUS"
mark() { echo "$* $(date -Iseconds)" | tee -a "$STATUS"; }

process_one() {
  local onnx=$1
  local stem
  stem=$(basename "$onnx" .onnx)
  local param="$V3_DIR/${stem}.param"
  local bin="$V3_DIR/${stem}.bin"
  local table="$V3_DIR/${stem}.table"
  local iparam="$V3_DIR/${stem}_int8.param"
  local ibin="$V3_DIR/${stem}_int8.bin"
  local out_txt="$OUT/${stem}_int8.txt"

  mark "START $stem"

  if [[ ! -f "$param" || ! -f "$bin" ]]; then
    echo "onnx2ncnn $stem"
    onnx2ncnn "$onnx" "$param" "$bin" || { mark "FAIL onnx2ncnn $stem"; return 1; }
    ncnnoptimize "$param" "$bin" "$param" "$bin" 0 || true
  fi

  # --- calib ---
  if [[ ! -f "$iparam" || ! -f "$ibin" ]]; then
    echo "===== ncnn2table $stem ====="
    rm -f "$table"
    # MUST quote mean/norm/shape: nullglob would eat unquoted [0,0,0]
    ncnn2table "$param" "$bin" "$CALIB_LIST" "$table" \
      "mean=[0,0,0]" "norm=[0.00392157,0.00392157,0.00392157]" "shape=[112,112,3]" pixel=RGB thread=2 method=kl
    local trc=$?
    if [[ $trc -ne 0 || ! -s "$table" ]]; then
      echo "retry ncnn2table with norm=1"
      ncnn2table "$param" "$bin" "$CALIB_LIST" "$table" \
        "mean=[0,0,0]" "norm=[1,1,1]" "shape=[112,112,3]" pixel=RGB thread=2 method=kl
      trc=$?
    fi
    if [[ $trc -ne 0 || ! -s "$table" ]]; then
      mark "FAIL table $stem rc=$trc size=$(stat -c%s "$table" 2>/dev/null || echo 0)"
      return 1
    fi
    echo "table OK $(stat -c%s "$table") bytes"

    echo "===== ncnn2int8 $stem ====="
    ncnn2int8 "$param" "$bin" "$iparam" "$ibin" "$table"
    if [[ ! -s "$iparam" || ! -s "$ibin" ]]; then
      mark "FAIL int8convert $stem"
      return 1
    fi
    ls -la "$iparam" "$ibin" "$table"
  else
    echo "skip calib (exists) $stem"
  fi

  # --- bench ---
  if [[ -f "$out_txt" ]]; then
    echo "skip bench (exists) $stem"
    mark "OK $stem (cached)"
    return 0
  fi
  local BENCH=$BENCH8
  [[ "$stem" == *expw* ]] && BENCH=$BENCH7
  echo "===== BENCH INT8 $stem ====="
  "$BENCH" --device h618 --param "$iparam" --bin "$ibin" --threads 4 --warmup 50 --iters 1000 \
    | tee "$out_txt"
  local brc=${PIPESTATUS[0]}
  if [[ $brc -ne 0 ]] || ! grep -q '^RESULT' "$out_txt"; then
    mark "FAIL bench $stem rc=$brc"
    return 1
  fi
  grep '^RESULT' "$out_txt"
  mark "OK $stem"
  return 0
}

# also fix overnight for future runs
shopt -s nullglob
ons=("$V3_DIR"/*.onnx)
echo "onnx count=${#ons[@]}"
# disable nullglob for the rest (protect mean=[...] args if any unquoted leftovers)
shopt -u nullglob
ok=0
fail=0
for onnx in "${ons[@]}"; do
  if process_one "$onnx"; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
  fi
done

echo "===== INT8 v3 DONE $(date -Iseconds) ok=$ok fail=$fail =====" | tee -a "$STATUS"
echo "----- status -----"
cat "$STATUS"
echo "----- INT8 RESULT -----"
grep -h '^RESULT' "$OUT"/*_int8.txt 2>/dev/null | tee "$OUT/all_int8_results.txt" || true
ls -la "$V3_DIR"/*int8* "$V3_DIR"/*.table 2>/dev/null | head -60
ls "$OUT"/*_int8.txt 2>/dev/null | wc -l
