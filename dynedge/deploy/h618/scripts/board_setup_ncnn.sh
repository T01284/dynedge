#!/bin/bash
# Native NCNN install on OrangePi Zero3 / H618 (Armbian).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
LOG=/root/h618_setup.log
exec > >(tee -a "$LOG") 2>&1
echo "===== START $(date -Iseconds) ====="
apt-get update -qq
apt-get install -y -qq cmake protobuf-compiler libprotobuf-dev git wget unzip pkg-config
mkdir -p /opt/src /opt/ncnn /root/fer-dynedge
if [[ ! -d /opt/src/ncnn/.git ]]; then
  git clone --depth 1 https://github.com/Tencent/ncnn.git /opt/src/ncnn
fi
cd /opt/src/ncnn
COMMIT=$(git rev-parse --short HEAD)
echo "NCNN_COMMIT=$COMMIT"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ncnn/install \
  -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_OPENMP=ON -DNCNN_SIMPLEOCV=ON
cmake --build . -j"$(nproc)"
cmake --install .
echo "===== NCNN DONE $(date -Iseconds) ====="
ls -la /opt/ncnn/install/bin | head
echo "NCNN_COMMIT=$COMMIT" > /opt/ncnn/COMMIT.txt
echo "OK /opt/ncnn/install"
