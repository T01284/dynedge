#!/usr/bin/env bash
# Lock CPU governor to performance before latency / power runs.
# Usage: sudo ./lock_perf.sh
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERR need root (sudo)" >&2
  exit 1
fi

shopt -s nullglob
govs=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
if [[ ${#govs[@]} -eq 0 ]]; then
  echo "WARN no cpufreq governors found"
  exit 0
fi

for g in "${govs[@]}"; do
  echo performance >"$g"
  echo "OK $g=$(cat "$g")"
done

# Optional: disable idle deep states if available (board-dependent).
if [[ -d /sys/devices/system/cpu/cpu0/cpuidle ]]; then
  echo "INFO cpuidle present — leave as-is unless jitter is high"
fi

echo "READY lock=performance"
