#!/usr/bin/env bash
# On-device (adb shell as root) NPU / CPU bring-up for MYIR T527 DEMO CAR (Android 13).
set -eu

echo "=== uname ==="
uname -a
getprop ro.product.model
getprop ro.build.display.id
getprop ro.build.version.release

echo "=== NPU nodes ==="
ls -l /dev/vipcore /dev/mali0 2>/dev/null || echo "WARN missing nodes"
dmesg 2>/dev/null | grep -iE 'vip|npu|VIPLite' | tail -n 20 || true
ls /vendor/lib64/*vip* 2>/dev/null || true

echo "=== CPU lock performance ==="
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance >"$c" 2>/dev/null || true
done
echo -n "governor="; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo -n "freq0="; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
echo -n "freq7="; cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq 2>/dev/null || echo n/a
grep -E 'CPU part|Features' /proc/cpuinfo | head -n 4
nproc

echo "=== net ==="
ip -4 addr show wlan0 2>/dev/null | sed -n '1,6p' || true

echo "DONE check_npu_android"
