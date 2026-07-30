#!/bin/bash
# H618: recalibrate INT8 with real FER+/ExpW 112 images, then latency bench.
set -euo pipefail
export PATH=/opt/ncnn/install/bin:$PATH
MODELS=/root/fer-dynedge/deploy/ncnn_edge/models
BENCH=/root/fer-dynedge/deploy/ncnn_edge/build-ncnn/fer_ncnn_bench
OUT=/root/fer-dynedge/results
CALIB_ROOT=/root/fer-dynedge/data/calib/linux112/calib
mkdir -p "$OUT" "$MODELS"

bash /root/fer-dynedge/deploy/ncnn_edge/scripts/lock_perf.sh

# imagelist: prefer png/jpg under calib/
LIST=$MODELS/imagelist_real.txt
find "$CALIB_ROOT" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | sort > "$LIST"
N=$(wc -l < "$LIST")
echo "calib images: $N"
if [[ "$N" -lt 50 ]]; then
  echo "ERR too few calib images" >&2
  exit 1
fi

cd "$MODELS"
echo "===== ncnn2table (real) ====="
ncnn2table mobilenetv3_small.param mobilenetv3_small.bin imagelist_real.txt \
  mobilenetv3_small_real.table \
  mean=[0,0,0] norm=[0.00392157,0.00392157,0.00392157] shape=[112,112,3] pixel=RGB thread=2 method=kl

echo "===== ncnn2int8 (real) ====="
ncnn2int8 mobilenetv3_small.param mobilenetv3_small.bin \
  mobilenetv3_small_int8_real.param mobilenetv3_small_int8_real.bin \
  mobilenetv3_small_real.table
ls -la mobilenetv3_small_int8_real.* mobilenetv3_small_real.table

: > "$OUT/h618_int8_real_raw.txt"
run_one() {
  local tag="$1" param="$2" bin="$3" threads="$4" warmup="$5" iters="$6"
  echo "===== BENCH tag=$tag =====" | tee -a "$OUT/h618_int8_real_raw.txt"
  "$BENCH" --device h618 --param "$param" --bin "$bin" \
    --threads "$threads" --warmup "$warmup" --iters "$iters" \
    | tee -a "$OUT/h618_int8_real_raw.txt"
  echo "TAG tag=$tag threads=$threads" | tee -a "$OUT/h618_int8_real_raw.txt"
}

# FP32 formal (baseline refresh)
run_one "fp32_formal" "$MODELS/mobilenetv3_small.param" "$MODELS/mobilenetv3_small.bin" 4 50 1000
# old synthetic INT8 formal
run_one "int8_synth_formal" "$MODELS/mobilenetv3_small_int8.param" "$MODELS/mobilenetv3_small_int8.bin" 4 50 1000
# new real INT8
run_one "int8_real_t1" "$MODELS/mobilenetv3_small_int8_real.param" "$MODELS/mobilenetv3_small_int8_real.bin" 1 50 200
run_one "int8_real_t4" "$MODELS/mobilenetv3_small_int8_real.param" "$MODELS/mobilenetv3_small_int8_real.bin" 4 50 200
run_one "int8_real_formal" "$MODELS/mobilenetv3_small_int8_real.param" "$MODELS/mobilenetv3_small_int8_real.bin" 4 50 1000

echo "===== DONE ====="
grep -E '^RESULT|^TAG' "$OUT/h618_int8_real_raw.txt"
