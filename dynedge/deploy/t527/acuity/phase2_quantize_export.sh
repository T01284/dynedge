#!/usr/bin/env bash
set -euo pipefail
cd /workspace/fer_mnv3
NAME=mobilenetv3_small
PEG="python3 /root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/pegasus.py"
VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools
OPT=VIP9000NANOSI_PLUS_PID0X10000016

python3 - <<'PY'
from pathlib import Path
import struct, zlib

# synthetic 112x112 RGB PNG for calibration
w = h = 112
rows = []
for i in range(h):
    row = bytes([0])
    for j in range(w):
        row += bytes([(i * 3 + j) % 256, (i * 5 + j * 2) % 256, (i * 7 + j * 3) % 256])
    rows.append(row)
raw = b"".join(rows)

def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

sig = bytes([137, 80, 78, 71, 13, 10, 26, 10])
png = (
    sig
    + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(raw, 9))
    + chunk(b"IEND", b"")
)
Path("calib0.png").write_bytes(png)
Path("dataset.txt").write_text("\n".join(["calib0.png"] * 16) + "\n")

# ImageNet BGR means (reverse_channel: true in meta)
meta = Path("mobilenetv3_small_inputmeta.yml")
text = meta.read_text()
text = text.replace(
    "        mean:\n        - 0\n        - 0\n        - 0\n",
    "        mean:\n        - 103.53\n        - 116.28\n        - 123.675\n",
)
text = text.replace(
    "        scale:\n        - 1.0\n        - 1.0\n        - 1.0\n",
    "        scale:\n        - 0.01743\n        - 0.01751\n        - 0.01712\n",
)
meta.write_text(text)
print("META PATCHED:")
print(meta.read_text())
PY

# postprocess file (best-effort)
$PEG generate postprocess-file --model ${NAME}.json --postprocess-file-output ${NAME}_postprocess_file.yml \
  || $PEG generate postprocess-file --model ${NAME}.json --postprocess-file ${NAME}_postprocess_file.yml \
  || true
if [ ! -f ${NAME}_postprocess_file.yml ]; then
  printf 'type: none\n' > ${NAME}_postprocess_file.yml
fi

echo ==== quantize ====
$PEG quantize \
  --model ${NAME}.json \
  --model-data ${NAME}.data \
  --device CPU \
  --with-input-meta ${NAME}_inputmeta.yml \
  --quantizer asymmetric_affine \
  --qtype uint8 \
  --iterations 8 \
  --model-quantize ${NAME}_uint8.quantize

echo ==== export ovxlib / NBG ====
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

echo ==== artifacts ====
find wksp -type f | head -n 100
find . -name 'network_binary.nb' -ls || true
find . -name '*.nb' -ls || true
