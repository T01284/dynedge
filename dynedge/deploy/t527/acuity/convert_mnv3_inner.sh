#!/usr/bin/env bash
set -euo pipefail
cd /workspace/mnv3_fer
NAME=mobilenetv3_small
PEG="python3 /root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/pegasus.py"
VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools
OPT=VIP9000NANOSI_PLUS_PID0X10000016

python3 - <<'PY'
import onnx
m=onnx.load('mobilenetv3_small.onnx')
ins=[]
init={x.name for x in m.graph.initializer}
for i in m.graph.input:
  if i.name in init: continue
  ins.append((i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]))
outs=[(o.name,[d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
print('inputs', ins)
print('outputs', outs)
PY

echo '==== import ===='
$PEG import onnx --model ${NAME}.onnx --output-model ${NAME}.json --output-data ${NAME}.data

echo '==== inputmeta ===='
$PEG generate inputmeta --model ${NAME}.json --input-meta-output ${NAME}_inputmeta.yml

python3 - <<'PY'
from pathlib import Path
import struct, zlib, re, yaml
p=Path('mobilenetv3_small_inputmeta.yml')
text=p.read_text()
print('RAW META:\n', text[:2000])

# synthetic 112 png
w=h=112
rows=[]
for i in range(h):
  row=b'\x00'
  for j in range(w):
    row += bytes([(i*3+j)%256,(i*5+j*2)%256,(i*7+j*3)%256])
  rows.append(row)
raw=b''.join(rows)
def chunk(t,d):
  return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b'')
Path('calib0.png').write_bytes(png)
Path('dataset.txt').write_text('\n'.join(['calib0.png']*16)+'\n')

# try parse yaml and set mean/scale/database
try:
  data=yaml.safe_load(text)
except Exception as e:
  print('yaml parse fail', e)
  data=None

if isinstance(data, dict):
  # common structures vary; best-effort
  def set_ms(obj):
    if isinstance(obj, dict):
      if 'mean' in obj:
        obj['mean']=[123.675,116.28,103.53]
      if 'scale' in obj:
        obj['scale']=[0.01712,0.01751,0.01743]
      if 'database' in obj and isinstance(obj['database'], str):
        obj['database']='dataset.txt'
      if 'path' in obj and isinstance(obj['path'], str) and 'dataset' in obj.get('path','').lower():
        obj['path']='dataset.txt'
      for v in obj.values():
        set_ms(v)
    elif isinstance(obj, list):
      for v in obj:
        set_ms(v)
  set_ms(data)
  if 'databases' not in data:
    data['databases']=[{'path':'dataset.txt'}]
  p.write_text(yaml.safe_dump(data, sort_keys=False))
else:
  # fallback text patch
  if 'dataset.txt' not in text:
    text += '\ndatabases:\n- path: dataset.txt\n'
  p.write_text(text)
print('META DONE:\n', p.read_text()[:2500])
PY

$PEG generate postprocess-file --model ${NAME}.json --postprocess-file-output ${NAME}_postprocess_file.yml \
  || $PEG generate postprocess-file --model ${NAME}.json --postprocess-file ${NAME}_postprocess_file.yml \
  || true
test -f ${NAME}_postprocess_file.yml || echo 'type: none' > ${NAME}_postprocess_file.yml

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
find wksp -type f | head -n 80
find . -name 'network_binary.nb' -ls
