#!/usr/bin/env bash
# Re-quantize + export NBG with real calib (container).
set -euo pipefail
cd /workspace/fer_mnv3
NAME=mobilenetv3_small
PEG="python3 /root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/pegasus.py"
VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools
OPT=VIP9000NANOSI_PLUS_PID0X10000016

test -f dataset_real.txt
test -d calib_real
n=$(wc -l < dataset_real.txt)
echo "calib lines=$n"

python3 - <<'PY'
from pathlib import Path
meta = Path("mobilenetv3_small_inputmeta.yml")
text = meta.read_text()
# point database to real dataset
if "dataset_real.txt" not in text:
    text = text.replace("path: dataset.txt", "path: dataset_real.txt")
# ensure ImageNet BGR mean/scale (reverse_channel true)
if "103.53" not in text:
    text = text.replace(
        "        mean:\n        - 0\n        - 0\n        - 0\n",
        "        mean:\n        - 103.53\n        - 116.28\n        - 123.675\n",
    )
if "0.01743" not in text:
    text = text.replace(
        "        scale:\n        - 1.0\n        - 1.0\n        - 1.0\n",
        "        scale:\n        - 0.01743\n        - 0.01751\n        - 0.01712\n",
    )
meta.write_text(text)
print(meta.read_text())
PY

if [ ! -f ${NAME}_postprocess_file.yml ]; then
  printf 'type: none\n' > ${NAME}_postprocess_file.yml
fi

echo ==== quantize real ====
$PEG quantize \
  --model ${NAME}.json \
  --model-data ${NAME}.data \
  --device CPU \
  --with-input-meta ${NAME}_inputmeta.yml \
  --quantizer asymmetric_affine \
  --qtype uint8 \
  --iterations 32 \
  --model-quantize ${NAME}_uint8_real.quantize

echo ==== export real NBG ====
mkdir -p wksp
$PEG export ovxlib \
  --model ${NAME}.json \
  --model-data ${NAME}.data \
  --model-quantize ${NAME}_uint8_real.quantize \
  --with-input-meta ${NAME}_inputmeta.yml \
  --postprocess-file ${NAME}_postprocess_file.yml \
  --dtype quantized \
  --optimize ${OPT} \
  --viv-sdk ${VIV_SDK} \
  --pack-nbg-unify \
  --output-path ./wksp/${NAME}_uint8_real/${NAME}_uint8_real

echo ==== artifacts ====
find wksp -name network_binary.nb -ls
ls -la wksp/${NAME}_uint8_real_nbg_unify/ 2>/dev/null || ls -la wksp/*real*nbg* 2>/dev/null || true
