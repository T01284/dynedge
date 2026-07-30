#!/usr/bin/env bash
# Prepare real FER/ExpW calib images for ACUITY requantize (run on WSL host).
set -euo pipefail
SRC=/mnt/d/paper/lane-edge/data/calib/linux112/calib
DST=/home/wang/docker_data/fer_mnv3/calib_real
N=${1:-128}
rm -rf "$DST"
mkdir -p "$DST"
# sample N images (sorted for reproducibility)
mapfile -t files < <(find "$SRC" -type f \( -iname '*.jpg' -o -iname '*.png' \) | sort | head -n "$N")
i=0
for f in "${files[@]}"; do
  printf -v name "c%04d.jpg" "$i"
  # copy as-is; ACUITY preprocess resizes to 112
  cp -f "$f" "$DST/$name"
  i=$((i+1))
done
(cd "$DST" && ls -1 *.jpg | sort > ../dataset_real.txt)
# rewrite paths relative to fer_mnv3 workdir
sed -i 's|^|calib_real/|' /home/wang/docker_data/fer_mnv3/dataset_real.txt
echo "prepared $i images -> $DST"
wc -l /home/wang/docker_data/fer_mnv3/dataset_real.txt
head -n 3 /home/wang/docker_data/fer_mnv3/dataset_real.txt
