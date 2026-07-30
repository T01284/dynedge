#!/usr/bin/env bash
# Cross-build fer_ncnn_bench for Android arm64-v8a (T527 DEMO CAR / Android 13).
# Env:
#   ANDROID_NDK   e.g. $HOME/src/android-ndk-r26d
#   NCNN_SRC      e.g. $HOME/src/ncnn  (will build install under $NCNN_SRC/build-android-arm64/install)
#   API_LEVEL     default 28
set -eu

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NCNN_EDGE="${ROOT}/ncnn_edge"
: "${ANDROID_NDK:?set ANDROID_NDK}"
: "${NCNN_SRC:?set NCNN_SRC to ncnn source tree}"
API_LEVEL="${API_LEVEL:-28}"
ABI=arm64-v8a

NCNN_BUILD="${NCNN_SRC}/build-android-${ABI}"
NCNN_INSTALL="${NCNN_BUILD}/install"
BENCH_BUILD="${NCNN_EDGE}/build-android-${ABI}"

echo "=== build ncnn (${ABI}, API ${API_LEVEL}) ==="
cmake -S "${NCNN_SRC}" -B "${NCNN_BUILD}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ABI}" \
  -DANDROID_PLATFORM="android-${API_LEVEL}" \
  -DANDROID_STL=c++_static \
  -DNCNN_VULKAN=OFF \
  -DNCNN_OPENMP=ON \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_SHARED_LIB=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${NCNN_INSTALL}"
cmake --build "${NCNN_BUILD}" -j"$(nproc)"
cmake --install "${NCNN_BUILD}"

echo "=== build fer_ncnn_bench ==="
cmake -S "${NCNN_EDGE}" -B "${BENCH_BUILD}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ABI}" \
  -DANDROID_PLATFORM="android-${API_LEVEL}" \
  -DANDROID_STL=c++_static \
  -DFER_HAS_NCNN=ON \
  -DNCNN_DIR="${NCNN_INSTALL}" \
  -Dncnn_DIR="${NCNN_INSTALL}/lib/cmake/ncnn" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${BENCH_BUILD}" -j"$(nproc)"

echo "OK ${BENCH_BUILD}/fer_ncnn_bench"
echo "HINT: adb push ${BENCH_BUILD}/fer_ncnn_bench /data/local/tmp/fer_t527/"
