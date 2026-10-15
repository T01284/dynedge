#!/usr/bin/env bash
# Fully-static fer_ncnn_bench for T527 Buildroot (glibc 2.33; no 2.39).
# Uses OpenMP-OFF ncnn static prefix (same as formal Linux probe).
set -euo pipefail
WSL_ROOT="<repo>"
NCNN_PREFIX="${HOME}/opt/ncnn-aarch64-static"
NCNN_SRC="${HOME}/src/ncnn"
BUILD_NCNN="${HOME}/build/ncnn-aarch64-static"
TOOLCHAIN_FILE="${WSL_ROOT}/deploy/t527/scripts/aarch64-linux-gnu.cmake"
EDGE="${WSL_ROOT}/deploy/ncnn_edge"

mkdir -p "$(dirname "$TOOLCHAIN_FILE")"
cat > "$TOOLCHAIN_FILE" <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
EOF

# Ensure static ncnn exists with SDOT
need_rebuild=0
if [[ ! -f "$NCNN_PREFIX/lib/libncnn.a" ]]; then
  need_rebuild=1
fi
if [[ -f "$BUILD_NCNN/CMakeCache.txt" ]]; then
  if ! grep -q 'NCNN_ARM82DOT:BOOL=ON' "$BUILD_NCNN/CMakeCache.txt"; then
    need_rebuild=1
  fi
fi

if [[ "$need_rebuild" == 1 ]]; then
  echo "=== rebuild ncnn aarch64 static (OPENMP=OFF, ARM82DOT=ON) ==="
  rm -rf "$BUILD_NCNN"
  cmake -S "$NCNN_SRC" -B "$BUILD_NCNN" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$NCNN_PREFIX" \
    -DNCNN_VULKAN=OFF \
    -DNCNN_BUILD_TOOLS=OFF \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_TESTS=OFF \
    -DNCNN_BUILD_BENCHMARK=OFF \
    -DNCNN_OPENMP=OFF \
    -DNCNN_SIMPLEOCV=ON \
    -DNCNN_SHARED_LIB=OFF \
    -DNCNN_ARM82=ON \
    -DNCNN_ARM82DOT=ON
  cmake --build "$BUILD_NCNN" -j"$(nproc)"
  cmake --install "$BUILD_NCNN"
  grep -q 'NCNN_ARM82DOT:BOOL=ON' "$BUILD_NCNN/CMakeCache.txt"
fi

for NCLS in 7 8; do
  BDIR="${EDGE}/build-linux-aarch64-c${NCLS}"
  echo "=== fer_ncnn_bench classes=${NCLS} ==="
  rm -rf "$BDIR"
  cmake -S "$EDGE" -B "$BDIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DCMAKE_BUILD_TYPE=Release \
    -DFER_HAS_NCNN=ON \
    -DNCNN_DIR="$NCNN_PREFIX" \
    -DFER_NUM_CLASSES="${NCLS}" \
    -DCMAKE_EXE_LINKER_FLAGS="-static"
  cmake --build "$BDIR" --target fer_ncnn_bench -j"$(nproc)"
  file "$BDIR/fer_ncnn_bench"
  aarch64-linux-gnu-readelf -d "$BDIR/fer_ncnn_bench" 2>/dev/null | head -n 5 || echo "(no dynamic section — good)"
  echo "OK classes=${NCLS} -> $BDIR/fer_ncnn_bench"
done
