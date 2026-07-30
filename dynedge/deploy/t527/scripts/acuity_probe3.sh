#!/usr/bin/env bash
docker exec npu_v1.8.13_wang bash -lc '
ls /root/Vivante_IDE/VivanteIDE5.8.2/cmdtools/
find /root/Vivante_IDE/VivanteIDE5.8.2/cmdtools -maxdepth 3 -type d
grep -R "VIP9000" /root/Vivante_IDE/VivanteIDE5.8.2/cmdtools --include="*.h" --include="*.txt" --include="*.cfg" --include="*.xml" 2>/dev/null | head -n 40
ls /root/nbinfo 2>/dev/null || true
/root/nbinfo -h 2>&1 | head -n 20 || true
'
