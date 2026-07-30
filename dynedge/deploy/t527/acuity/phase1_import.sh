#!/usr/bin/env bash
set -euo pipefail
cd /workspace/fer_mnv3
NAME=mobilenetv3_small
PEG="python3 /root/Vivante_ML_Toolkit_acuity_whl_6.45.8/bin/pegasus.py"
pwd; ls -la
python3 - <<'PY'
import onnx
m=onnx.load('mobilenetv3_small.onnx')
init={x.name for x in m.graph.initializer}
ins=[]
for i in m.graph.input:
  if i.name in init: continue
  ins.append((i.name,[d.dim_value for d in i.type.tensor_type.shape.dim]))
outs=[(o.name,[d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
print('inputs', ins)
print('outputs', outs)
PY
echo ==== import ====
$PEG import onnx --model ${NAME}.onnx --output-model ${NAME}.json --output-data ${NAME}.data
echo ==== inputmeta ====
$PEG generate inputmeta --model ${NAME}.json --input-meta-output ${NAME}_inputmeta.yml
echo ==== meta content ====
cat ${NAME}_inputmeta.yml
