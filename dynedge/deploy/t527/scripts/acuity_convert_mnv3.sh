#!/usr/bin/env bash
# Convert MobileNetV3-Small FER ONNX -> NBG for T527
set -euo pipefail
CONTAINER=npu_v1.8.13_wang
docker exec "$CONTAINER" bash <<'EOS'
set -e
cd /workspace/mnv3_fer
NAME=mobilenetv3_small
PEG="python3 /root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/pegasus.py"
VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools
OPT=VIP9000NANOSI_PLUS_PID0X10000016

python3 - <<'PY'
import onnx
m=onnx.load('mobilenetv3_small.onnx')
print('inputs', [(i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input if i.name not in {x.name for x in m.graph.initializer}])
print('outputs', [(o.name,[d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
PY

echo '==== import ===='
$PEG import onnx --model ${NAME}.onnx --output-model ${NAME}.json --output-data ${NAME}.data

echo '==== inputmeta ===='
$PEG generate inputmeta --model ${NAME}.json --input-meta-output ${NAME}_inputmeta.yml
python3 - <<'PY'
from pathlib import Path
import re
p=Path('mobilenetv3_small_inputmeta.yml')
t=p.read_text()
print('BEFORE:\n', t[:1500])
# force ImageNet-ish mean/scale if placeholders exist
t2=t
# common patterns in acuity yml
t2=re.sub(r'(?m)^( *mean:\s*).*$', r'\1 [123.675, 116.28, 103.53]', t2, count=1)
# sometimes list form
if 'mean:' in t2 and '123.675' not in t2:
    t2=t2.replace('mean:\n', 'mean:\n- 123.675\n- 116.28\n- 103.53\n', 1)
if 'scale:' in t2 and '0.01712' not in t2:
    # replace first scale block roughly
    t2=re.sub(r'(?ms)^( *scale:\s*\n(?:- .*\n){1,3})', 'scale:\n- 0.01712\n- 0.01751\n- 0.01743\n', t2, count=1)
# database path
if 'database' in t2 or 'path' in t2 or 'dataset' in t2:
    pass
p.write_text(t2)
print('AFTER:\n', p.read_text()[:2000])
PY

# synthetic calib set
python3 - <<'PY'
from pathlib import Path
import struct, zlib
w=h=112
raw=b''.join(b'\x00'+bytes([(i*3+j)%256,(i*5+j*2)%256,(i*7+j*3)%256])*w for i in range(h) for j in [0])
# fix: each row
rows=[]
for i in range(h):
  row=b'\x00'
  for j in range(w):
    row+=bytes([(i*3+j)%256,(i*5+j*2)%256,(i*7+j*3)%256])
  rows.append(row)
raw=b''.join(rows)
def chunk(t,d):
  return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b'')
Path('calib0.png').write_bytes(png)
Path('dataset.txt').write_text('\n'.join(['calib0.png']*16)+'\n')
print('calib ok', Path('calib0.png').stat().st_size)
PY

# patch database into yml if needed
python3 - <<'PY'
from pathlib import Path
p=Path('mobilenetv3_small_inputmeta.yml')
t=p.read_text()
if 'dataset.txt' not in t:
  # acuity often uses:
  # databases:
  # - path: dataset.txt
  if 'databases:' in t:
    t=t.replace('databases:', 'databases:\n- path: dataset.txt\n#patched', 1)
  else:
    t += '\ndatabases:\n- path: dataset.txt\n'
  p.write_text(t)
print(p.read_text()[:2500])
PY

$PEG generate postprocess-file --model ${NAME}.json --postprocess-file-output ${NAME}_postprocess_file.yml 2>/dev/null \
  || $PEG generate postprocess-file --model ${NAME}.json --postprocess-file ${NAME}_postprocess_file.yml 2>/dev/null \
  || true
ls -la ${NAME}_postprocess_file.yml 2>/dev/null || echo 'type: none' > ${NAME}_postprocess_file.yml

echo '==== quantize ===='
$PEG quantize \
  --model ${NAME}.json \
  --model-data ${NAME}.data \
  --device CPU \
  --with-input-meta ${NAME}_inputmeta.yml \
  --quantizer asymmetric_affine \
  --qtype uint8 \
  --iterations 8 \
  --model-quantize ${NAME}_uint8.quantize

echo '==== export ===='
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

echo '==== artifacts ===='
find wksp -type f | head -n 50
find . -name 'network_binary.nb'
EOS
