#!/bin/bash
# H618 H1: regenerate v4 INT8 tables with method=aciq (accuracy only; do NOT rebench latency).
set -euo pipefail
export PATH=/opt/ncnn/install/bin:$PATH
ROOT=/root/fer-dynedge
V4_DIR=$ROOT/deploy/ncnn_edge/models/v4_predistill
KL_BAK=$ROOT/deploy/ncnn_edge/models/v4_predistill_kl_bak
OUT=$ROOT/results/overnight_v4
CALIB112=$ROOT/deploy/ncnn_edge/models/imagelist_real.txt
CALIB64=$ROOT/deploy/ncnn_edge/models/imagelist_mcu64.txt
mkdir -p "$OUT" "$KL_BAK"
STATUS=$OUT/v4_aciq_status.txt
LOG=$OUT/v4_aciq_recalib.log
exec > >(tee -a "$LOG") 2>&1
echo "===== V4 ACIQ RECALIB START $(date -Iseconds) =====" | tee "$STATUS"

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
calib_for() {
  local stem=$1
  if [[ "$stem" == *img64* || "$stem" == *_s64g_* ]]; then echo "$CALIB64"; else echo "$CALIB112"; fi
}
mark() { echo "$1" | tee -a "$STATUS"; }

# Backup existing KL int8/table once (do not overwrite prior backup).
shopt -s nullglob
for f in "$V4_DIR"/*.table "$V4_DIR"/*_int8.param "$V4_DIR"/*_int8.bin; do
  base=$(basename "$f")
  if [[ ! -f "$KL_BAK/$base" ]]; then
    cp -a "$f" "$KL_BAK/$base"
  fi
done
shopt -u nullglob
mark "KL_BAK $(ls "$KL_BAK"/*.table 2>/dev/null | wc -l) tables"

ok=0; fail=0
shopt -s nullglob
for param in "$V4_DIR"/*.param; do
  stem=$(basename "$param" .param)
  [[ "$stem" == *_int8 ]] && continue
  bin="$V4_DIR/${stem}.bin"
  iparam="$V4_DIR/${stem}_int8.param"
  ibin="$V4_DIR/${stem}_int8.bin"
  table="$V4_DIR/${stem}.table"
  stamp="$OUT/${stem}_aciq.done"
  [[ -f "$stamp" && -s "$iparam" && -s "$ibin" && -s "$table" ]] && { mark "SKIP $stem"; ok=$((ok+1)); continue; }
  [[ -f "$bin" ]] || { mark "FAIL missing_fp32 $stem"; fail=$((fail+1)); continue; }

  shape=$(shape_for "$stem")
  calib=$(calib_for "$stem")
  echo "===== ACIQ ncnn2table $stem shape=$shape ====="
  rm -f "$table" "$iparam" "$ibin"
  if ! ncnn2table "$param" "$bin" "$calib" "$table" \
      "mean=[0,0,0]" "norm=[0.00392157,0.00392157,0.00392157]" "shape=${shape}" \
      pixel=RGB thread=2 method=aciq; then
    mark "FAIL table $stem"
    fail=$((fail+1))
    continue
  fi
  [[ -s "$table" ]] || { mark "FAIL empty_table $stem"; fail=$((fail+1)); continue; }

  # Patch non-finite scales (same as host prep_v4_ncnn_models.py).
  if grep -qiE 'inf|nan' "$table"; then
    sed -i 's/-inf/1e6/g; s/inf/1e6/g; s/[Nn][Aa][Nn]/1.0/g' "$table"
    echo "WARN patched nonfinite scales in $stem.table"
  fi

  if ! ncnn2int8 "$param" "$bin" "$iparam" "$ibin" "$table"; then
    mark "FAIL int8 $stem"
    fail=$((fail+1))
    continue
  fi
  [[ -s "$iparam" && -s "$ibin" ]] || { mark "FAIL empty_int8 $stem"; fail=$((fail+1)); continue; }
  date -Iseconds > "$stamp"
  mark "OK $stem"
  ok=$((ok+1))
done
shopt -u nullglob

echo "===== V4 ACIQ RECALIB DONE $(date -Iseconds) ok=$ok fail=$fail =====" | tee -a "$STATUS"
# Explicitly do NOT touch *_fp32.txt / *_int8.txt latency artifacts.
