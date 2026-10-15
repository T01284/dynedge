"""One-shot progress snapshot of the running board/Orin jobs (read-only)."""
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/stageB"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tail_lines(p: Path, pat: str) -> list[str]:
    if not p.exists():
        return []
    raw = p.read_bytes()
    # Windows PowerShell Out-File / *>> write UTF-16LE
    text = raw.decode("utf-16", errors="replace") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") \
        else raw.decode("utf-8", errors="replace")
    return [l.strip() for l in text.splitlines() if l.strip() and pat in l]


def n_rows(p: Path) -> int:
    return sum(1 for _ in csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else 0


print("[T527 chain]", " | ".join(tail_lines(R / "c1_fix/chain_trainprep_t527.log", "")[-3:]))
print(f"  NCNN FER+ acc rows {n_rows(R / 'board_full_acc/fixed/t527.csv')}/28")
for tag, n in (("fix", 6), ("relu_fix", 6), ("relu_old_div255", 6)):
    print(f"  NPU {tag}: {n_rows(R / f'npu_full_acc/{tag}.csv')}/{n}")
lat = tail_lines(R / "npu_full_acc_lat_relu.log", "p50")
print(f"  NPU latency lines: {len(lat)}")

print("[T527 ReLU-vs-Hswish energy]", " | ".join(tail_lines(R / "c1_fix/chain_t527_relu_energy.log", "")[-2:]))
print(f"  B5 windows relu {n_rows(R / 'b5_energy_window/t527/v4_npu_relu_fix/runs.csv')}/48, "
      f"hswish {n_rows(R / 'b5_energy_window/t527/v4_npu_fix/runs.csv')}/(24 old + 32 new)")

print("[H618 chain]", " | ".join(tail_lines(R / "c1_fix/chain_trainprep_h618.log", "")[-3:]))
print(f"  NCNN ExpW acc rows {n_rows(R / 'board_full_acc/expw/fixed/h618.csv')}/34, "
      f"FER+ rows {n_rows(R / 'board_full_acc/fixed/h618.csv')}/28")
last = tail_lines(R / "board_full_acc/expw/fixed/run_h618.log", "[h618]")[-1:] + \
    tail_lines(R / "board_full_acc/fixed/run_h618.log", "[h618]")[-1:]
for l in last:
    print("  last:", l[:140])

try:
    spec = importlib.util.spec_from_file_location("ssh_orin", ROOT / "deploy/orin/scripts/ssh_orin.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    _, out, _ = S.run("cd <gpu-workdir>/fer-dynedge/results/stageB/orin_sweep && "
                      "echo searches=$(ls search/*.json 2>/dev/null | wc -l)/6 "
                      "o4_rows=$(($(cat o4_latency.csv 2>/dev/null | wc -l)-1)) "
                      "&& tail -n 1 ../../../logs/stageB/status.txt", timeout=20)
    print("[Orin]", " ".join(out.split()))
except Exception as e:  # noqa: BLE001
    print("[Orin B10 O1] unreachable:", type(e).__name__, e)
