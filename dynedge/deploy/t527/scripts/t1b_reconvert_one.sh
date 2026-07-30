#!/usr/bin/env bash
set -euo pipefail
TOOLS="${HOME}/opt/ncnn-x86-tools/bin"
ONNX="/mnt/d/paper/lane-edge/results/h618_onnx_v4/subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64.onnx"
OUT="/tmp/t1b_reconvert"
KEEP="/mnt/d/paper/lane-edge/results/t527_v4/t1b_raw_ncnn"
mkdir -p "$OUT" "$KEEP"
"$TOOLS/onnx2ncnn" "$ONNX" "$OUT/raw.param" "$OUT/raw.bin"
"$TOOLS/ncnnoptimize" "$OUT/raw.param" "$OUT/raw.bin" "$OUT/opt.param" "$OUT/opt.bin" 0
python3 - <<'PY'
import numpy as np
from pathlib import Path
for tag in ("raw", "opt"):
    p = Path(f"/tmp/t1b_reconvert/{tag}.bin")
    a = np.fromfile(p, dtype=np.float32)
    print(f"{tag}: size={a.size} nan={np.isnan(a).sum()} inf={np.isinf(a).sum()} head={a[:16]}")
dep = np.fromfile(
    "/mnt/d/paper/lane-edge/deploy/t527/models/v4_predistill/"
    "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64.bin",
    dtype=np.float32,
)
print(f"deployed: size={dep.size} nan={np.isnan(dep).sum()} inf={np.isinf(dep).sum()}")
PY
cp -f "$OUT"/raw.param "$OUT"/raw.bin "$OUT"/opt.param "$OUT"/opt.bin "$KEEP/"
echo "kept -> $KEEP"
