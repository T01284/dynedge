#!/usr/bin/env bash
# Inside ACUITY Docker: convert one-or-more v4 ONNX → uint8 NBG (real-ish calib).
# Args: stem1 stem2 ...  (no .onnx)
set -euo pipefail
PEG="python3 /root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/pegasus.py"
VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools
OPT=VIP9000NANOSI_PLUS_PID0X10000016
ROOT=/workspace/lane-edge
ONNX_DIR=${ONNX_DIR:-$ROOT/results/h618_onnx_v4}
OUT_HOST=${OUT_HOST:-$ROOT/deploy/t527/models/v4_npu}
CALIB_SRC=$ROOT/data/calib/linux112/calib
CALIB64_SRC=$ROOT/data/calib/ferplus_mcu64
echo "ONNX_DIR=$ONNX_DIR OUT_HOST=$OUT_HOST"

mkdir -p "$OUT_HOST"

shape_hw() {
  local stem=$1
  if [[ "$stem" == *img64* || "$stem" == *_s64g_* ]]; then echo 64; 
  elif [[ "$stem" == *img96* || "$stem" == *_s96c_* ]]; then echo 96;
  else echo 112; fi
}

prep_calib() {
  local stem=$1 hw=$2 work=$3
  local n=128
  mkdir -p "$work/calib_real"
  rm -f "$work/dataset_real.txt"
  local src="$CALIB_SRC"
  if [[ "$hw" == 64 ]]; then src="$CALIB64_SRC"; fi
  mapfile -t imgs < <(find "$src" -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) | sort | head -n "$n")
  if [[ ${#imgs[@]} -lt 8 ]]; then
    echo "WARN few calib images (${#imgs[@]}) for $stem; synth fallback"
    python3 - <<PY
from pathlib import Path
import struct, zlib
w = h = ${hw}
rows=[]
for i in range(h):
    row=bytes([0])
    for j in range(w):
        row += bytes([(i*3+j)%256,(i*5+j*2)%256,(i*7+j*3)%256])
    rows.append(row)
raw=b"".join(rows)
def chunk(t,d):
    return struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xFFFFFFFF)
sig=bytes([137,80,78,71,13,10,26,10])
png=sig+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw,9))+chunk(b"IEND",b"")
p=Path("$work/calib_real/synth0.png"); p.write_bytes(png)
Path("$work/dataset_real.txt").write_text("\n".join([f"calib_real/{p.name}"]*16)+"\n")
PY
    return
  fi
  local gray=0
  [[ "$stem" == *s64g* || "$stem" == *img64* ]] && gray=1
  local i=0
  for img in "${imgs[@]}"; do
    local base
    base=$(printf "c%03d.png" "$i")
    local outf="$work/calib_real/$base"
    if [[ "$gray" == 1 ]]; then
      python3 -c "from PIL import Image; im=Image.open(r'''$img''').convert('L').resize(($hw,$hw)); Image.merge('RGB',(im,im,im)).save(r'''$outf''')"
    else
      python3 -c "from PIL import Image; Image.open(r'''$img''').convert('RGB').resize(($hw,$hw)).save(r'''$outf''')" \
        || cp -f "$img" "$outf"
    fi
    [[ -f "$outf" ]] || { echo "FAIL write $outf"; return 1; }
    echo "calib_real/$base" >> "$work/dataset_real.txt"
    i=$((i+1))
  done
  echo "calib lines=$(wc -l < "$work/dataset_real.txt") hw=$hw gray=$gray"
}

patch_meta() {
  local meta=$1 hw=$2
  python3 - <<PY
from pathlib import Path
meta = Path("$meta")
text = meta.read_text()
# point to real dataset
text = text.replace("path: dataset.txt", "path: dataset_real.txt")
if "dataset_real.txt" not in text:
    # some generators use different key layout — append database override if missing
    pass
# RGB /255 (match H618 NCNN + FER training); NOT ImageNet-BGR
text = text.replace("reverse_channel: true", "reverse_channel: false")
import re as _re
text = _re.sub(
    r"mean:\n(?:\s+-\s+[0-9eE.+\-]+\n){3}",
    "mean:\n        - 0\n        - 0\n        - 0\n",
    text,
    count=1,
)
text = _re.sub(
    r"scale:\n(?:\s+-\s+[0-9eE.+\-]+\n){3}",
    "scale:\n        - 0.00392157\n        - 0.00392157\n        - 0.00392157\n",
    text,
    count=1,
)
# force spatial size if listed as 0 / dynamic
# NOTE: must use \\g<2> — bare \\2${hw} becomes \\2NN and breaks (group 2NN).
import re
hw = int("${hw}")
text2 = re.sub(
    r"(size:\s*\n\s*-\s*)\d+(\s*\n\s*-\s*)\d+",
    lambda m: f"{m.group(1)}{hw}{m.group(2)}{hw}",
    text,
    count=1,
)
meta.write_text(text2)
print(meta.read_text())
PY
}

convert_one() {
  local stem=$1
  local hw
  hw=$(shape_hw "$stem")
  mkdir -p "$OUT_HOST"
  if [[ -s "$OUT_HOST/${stem}_uint8.nb" ]]; then
    echo "SKIP existing $OUT_HOST/${stem}_uint8.nb"
    return 0
  fi
  # keep path short-ish; fer_ prefix avoids PowerShell \r issues seen with mnv3_fe
  local work=/workspace/fer_v4_$(echo "$stem" | sed -E 's/subnet_(ferplus|expw)_//;s/_seed42_v4_(predistill|relu_npu|formal_lut)_img[0-9]+//;s/[.]//g')
  echo "======== CONVERT $stem hw=$hw work=$work ========"
  rm -rf "$work"
  mkdir -p "$work"
  cp -f "$ONNX_DIR/${stem}.onnx" "$work/model.onnx"
  cd "$work"
  NAME=model

  echo ==== import ====
  $PEG import onnx --model ${NAME}.onnx --output-model ${NAME}.json --output-data ${NAME}.data
  echo ==== inputmeta ====
  $PEG generate inputmeta --model ${NAME}.json --input-meta-output ${NAME}_inputmeta.yml
  prep_calib "$stem" "$hw" "$work"
  if [[ ! -f dataset.txt ]]; then
    cp -f dataset_real.txt dataset.txt
  fi
  patch_meta ${NAME}_inputmeta.yml "$hw"

  if [ ! -f ${NAME}_postprocess_file.yml ]; then
    $PEG generate postprocess-file --model ${NAME}.json --postprocess-file-output ${NAME}_postprocess_file.yml \
      || $PEG generate postprocess-file --model ${NAME}.json --postprocess-file ${NAME}_postprocess_file.yml \
      || printf 'type: none\n' > ${NAME}_postprocess_file.yml
  fi

  echo ==== quantize ====
  $PEG quantize \
    --model ${NAME}.json \
    --model-data ${NAME}.data \
    --device CPU \
    --with-input-meta ${NAME}_inputmeta.yml \
    --quantizer asymmetric_affine \
    --qtype uint8 \
    --iterations 64 \
    --model-quantize ${NAME}_uint8.quantize

  echo ==== export NBG ====
  mkdir -p wksp
  $PEG export ovxlib \
    --model ${NAME}.json \
    --model-data ${NAME}.data \
    --model-quantize ${NAME}_uint8.quantize \
    --with-input-meta ${NAME}_inputmeta.yml \
    --postprocess-file ${NAME}_postprocess_file.yml \
    --dtype quantized \
    --optimize ${OPT} \
    --viv-sdk ${VIV_SDK} \
    --pack-nbg-unify \
    --output-path ./wksp/${NAME}_uint8/${NAME}_uint8

  local nb
  nb=$(find . -name network_binary.nb | head -n 1 || true)
  if [[ -z "$nb" ]]; then
    echo "FAIL no network_binary.nb for $stem"
    return 1
  fi
  mkdir -p "$OUT_HOST"
  cp -f "$nb" "$OUT_HOST/${stem}_uint8.nb"
  cp -f ${NAME}_inputmeta.yml "$OUT_HOST/${stem}_inputmeta.yml" || true
  # nbg_meta carries input/output scale+zp for accuracy eval
  meta=$(find . -name nbg_meta.json | head -n 1 || true)
  if [[ -n "$meta" ]]; then
    cp -f "$meta" "$OUT_HOST/${stem}_nbg_meta.json"
  else
    echo "WARN no nbg_meta.json for $stem"
  fi
  ls -la "$OUT_HOST/${stem}_uint8.nb"
  echo "OK $stem -> $OUT_HOST/${stem}_uint8.nb"
}

STATUS=$OUT_HOST/batch_status.txt
echo "===== ACUITY V4 BATCH START $(date -Iseconds) =====" | tee -a "$STATUS"
fail=0
ok=0
for stem in "$@"; do
  if convert_one "$stem"; then
    echo "OK $stem" | tee -a "$STATUS"
    ok=$((ok+1))
  else
    echo "FAIL $stem" | tee -a "$STATUS"
    fail=$((fail+1))
  fi
done
echo "===== ACUITY V4 BATCH END ok=$ok fail=$fail $(date -Iseconds) =====" | tee -a "$STATUS"
exit $([[ "$fail" == 0 ]] && echo 0 || echo 1)
