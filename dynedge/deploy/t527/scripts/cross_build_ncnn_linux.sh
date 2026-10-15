#!/usr/bin/env bash
# Cross-build ncnn + fer_ncnn_bench for T527 Buildroot (glibc aarch64).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WSL_ROOT="<repo>"
NCNN_SRC="${HOME}/src/ncnn"
NCNN_PREFIX="${HOME}/opt/ncnn-aarch64"
BUILD_NCNN="${HOME}/build/ncnn-aarch64"
BUILD_BENCH="${WSL_ROOT}/deploy/ncnn_edge/build-linux-aarch64"
TOOLCHAIN_FILE="${WSL_ROOT}/deploy/t527/scripts/aarch64-linux-gnu.cmake"

mkdir -p "$(dirname "$TOOLCHAIN_FILE")" "$HOME/src" "$HOME/build" "$HOME/opt"

cat > "$TOOLCHAIN_FILE" <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
EOF

if [[ ! -d "$NCNN_SRC/.git" ]]; then
  git clone --depth 1 https://github.com/Tencent/ncnn.git "$NCNN_SRC"
fi
# Prefer same commit used on Android if present
cd "$NCNN_SRC"
git fetch --depth 1 origin a4d2ea1 2>/dev/null || true
git checkout a4d2ea1 2>/dev/null || git rev-parse --short HEAD

rm -rf "$BUILD_NCNN"
cmake -S "$NCNN_SRC" -B "$BUILD_NCNN" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$NCNN_PREFIX" \
  -DNCNN_VULKAN=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_OPENMP=ON \
  -DNCNN_SIMPLEOCV=ON \
  -DNCNN_SHARED_LIB=OFF \
  -DNCNN_ARM82=ON \
  -DNCNN_ARM82DOT=ON \
  -DNinja=OFF
cmake --build "$BUILD_NCNN" -j"$(nproc)"
cmake --install "$BUILD_NCNN"
# Sanity: refuse silent fall-back if compiler dropped DOT
grep -q 'NCNN_ARM82DOT:BOOL=ON' "$BUILD_NCNN/CMakeCache.txt"
grep -q 'NCNN_COMPILER_SUPPORT_ARM82_DOTPROD:INTERNAL=1' "$BUILD_NCNN/CMakeCache.txt"

rm -rf "$BUILD_BENCH"
cmake -S "${WSL_ROOT}/deploy/ncnn_edge" -B "$BUILD_BENCH" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFER_HAS_NCNN=ON \
  -DNCNN_DIR="$NCNN_PREFIX" \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libgcc -static-libstdc++"
cmake --build "$BUILD_BENCH" -j"$(nproc)"

BIN="$BUILD_BENCH/fer_ncnn_bench"
file "$BIN"
aarch64-linux-gnu-readelf -V "$BIN" 2>/dev/null | grep -E 'GLIBC_|Name:' | head -n 40 || true
aarch64-linux-gnu-objdump -T "$BIN" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -u || true
echo "OK $BIN"
echo "NCNN_COMMIT=$(cd "$NCNN_SRC" && git rev-parse --short HEAD)"
