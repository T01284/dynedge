#!/bin/bash
# H618: INT8 PTQ + thread sweep + formal FP32/INT8 benches.
set -euo pipefail
export PATH=/opt/ncnn/install/bin:$PATH
ROOT=/root/fer-dynedge/deploy
NCNN_EDGE=$ROOT/ncnn_edge
MODELS=$NCNN_EDGE/models
BENCH=$NCNN_EDGE/build-ncnn/fer_ncnn_bench
OUT=/root/fer-dynedge/results
mkdir -p "$OUT" "$MODELS/calib112"

echo "===== lock perf ====="
bash "$NCNN_EDGE/scripts/lock_perf.sh"

echo "===== gen calib jpgs (112x112) ====="
python3 - <<'PY'
from pathlib import Path
import struct, zlib, os
# write minimal RGB PNGs without pillow (ncnn simpleocv may need jpg/png)
# Prefer PPM which some tools accept; also write raw then use convert if available.
# ncnn2table uses stb/simpleocv — supports jpg/png. Generate uncompressed PNG.

def write_png(path, w, h, rgb_bytes):
    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    raw = b''.join(b'\x00' + rgb_bytes[y*w*3:(y+1)*w*3] for y in range(h))
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
    path.write_bytes(data)

out = Path('/root/fer-dynedge/deploy/ncnn_edge/models/calib112')
out.mkdir(parents=True, exist_ok=True)
n = 128
for i in range(n):
    # synthetic face-ish blob + noise, values 40..220
    import random
    rnd = random.Random(1000 + i)
    buf = bytearray(112 * 112 * 3)
    cx, cy = 56 + rnd.randint(-8, 8), 56 + rnd.randint(-8, 8)
    for y in range(112):
        for x in range(112):
            r2 = (x - cx) ** 2 + (y - cy) ** 2
            v = int(40 + 160 * __import__('math').exp(-r2 / (2 * 28 * 28)))
            v = max(0, min(255, v + rnd.randint(-12, 12)))
            o = (y * 112 + x) * 3
            buf[o:o+3] = bytes((v, v, v))
    write_png(out / f'c{i:03d}.png', 112, 112, bytes(buf))
print('wrote', n, 'pngs')
list_path = Path('/root/fer-dynedge/deploy/ncnn_edge/models/imagelist.txt')
list_path.write_text('\n'.join(str(out / f'c{i:03d}.png') for i in range(n)) + '\n')
print('imagelist', list_path)
PY

cd "$MODELS"
echo "===== ncnn2table ====="
ncnn2table mobilenetv3_small.param mobilenetv3_small.bin imagelist.txt mobilenetv3_small.table \
  mean=[0,0,0] norm=[0.00392157,0.00392157,0.00392157] shape=[112,112,3] pixel=RGB thread=2 method=kl

echo "===== ncnn2int8 ====="
ncnn2int8 mobilenetv3_small.param mobilenetv3_small.bin \
  mobilenetv3_small_int8.param mobilenetv3_small_int8.bin \
  mobilenetv3_small.table
ls -la mobilenetv3_small_int8.*

run_one() {
  local tag="$1" param="$2" bin="$3" threads="$4" warmup="$5" iters="$6"
  echo "===== BENCH tag=$tag threads=$threads warmup=$warmup iters=$iters ====="
  "$BENCH" --device h618 --param "$param" --bin "$bin" \
    --threads "$threads" --warmup "$warmup" --iters "$iters" \
    | tee -a "$OUT/h618_matrix_raw.txt"
  # annotate for host parser
  echo "TAG tag=$tag threads=$threads" | tee -a "$OUT/h618_matrix_raw.txt"
}

: > "$OUT/h618_matrix_raw.txt"

# FP32 thread sweep
for t in 1 2 4; do
  run_one "fp32_t${t}" "$MODELS/mobilenetv3_small.param" "$MODELS/mobilenetv3_small.bin" "$t" 50 200
done
# FP32 formal
run_one "fp32_formal" "$MODELS/mobilenetv3_small.param" "$MODELS/mobilenetv3_small.bin" 4 50 1000

# INT8 thread sweep
for t in 1 2 4; do
  run_one "int8_t${t}" "$MODELS/mobilenetv3_small_int8.param" "$MODELS/mobilenetv3_small_int8.bin" "$t" 50 200
done
run_one "int8_formal" "$MODELS/mobilenetv3_small_int8.param" "$MODELS/mobilenetv3_small_int8.bin" 4 50 1000

echo "===== DONE ====="
grep -E '^RESULT|^READY|^TAG|^MEM ' "$OUT/h618_matrix_raw.txt" | tail -80
