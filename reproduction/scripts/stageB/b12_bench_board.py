#!/usr/bin/env python3
"""B12 latency: bench the rebuilt C1 NCNN artifacts (deploy/t527/models/*_fix).

H618: upload to deploy/ncnn_edge/models/c1_fix/<arm>/ and run
      scripts/stageB/h618_c1_fix_bench.sh detached (native fer_ncnn_bench).
T527: run deploy/t527/scripts/run_linux_v4_matrix.py per arm dir (same tool as C1).
--pull collects RESULT lines into results/stageB/c1_fix/<board>/<arm>/matrix_summary.csv
and --compare prints new-vs-published latency per model.

Usage:
  python scripts/stageB/b12_bench_board.py --board h618 --launch
  python scripts/stageB/b12_bench_board.py --board h618 --pull
  python scripts/stageB/b12_bench_board.py --board t527 --launch   # synchronous over SSH
  python scripts/stageB/b12_bench_board.py --compare
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "deploy/t527/models"
OUT = ROOT / "results/stageB/c1_fix"
ARMS = ["v4_flops_ctrl_baked", "v4_formal_baked", "v4_formal_baked_expw", "v4_formal_lut_t527_baked"]
H_ROOT = "/root/fer-dynedge"
H_MODELS = f"{H_ROOT}/deploy/ncnn_edge/models/c1_fix"
H_OUT = f"{H_ROOT}/results/c1_fix_bench"
OLD = {
    "h618": {a: ROOT / f"results/h618_{a}/matrix_summary.csv" for a in ARMS},
    "t527": {a: ROOT / f"results/t527_{a}/matrix_summary.csv" for a in ARMS},
}


def ssh(board: str):
    path = ROOT / f"deploy/{board}/scripts/ssh_{board}.py"
    spec = importlib.util.spec_from_file_location(f"ssh_{board}", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"ssh_{board}"] = m
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def fix_files(arm: str) -> list[Path]:
    d = T / f"{arm}_fix"
    return sorted(p for p in d.iterdir() if p.suffix in {".param", ".bin"})


def launch_h618(upload: bool = True) -> None:
    S = ssh("h618")
    for arm in ARMS if upload else []:
        S.run(f"mkdir -p {H_MODELS}/{arm}", S.DEFAULT_HOST)
        for p in fix_files(arm):
            S.put(str(p), f"{H_MODELS}/{arm}/{p.name}", S.DEFAULT_HOST)
        print(f"uploaded {arm}", flush=True)
    sh = f"{H_ROOT}/scripts/stageB/h618_c1_fix_bench.sh"
    S.run(f"mkdir -p {H_ROOT}/scripts/stageB", S.DEFAULT_HOST)
    S.put(str(ROOT / "scripts/stageB/h618_c1_fix_bench.sh"), sh, S.DEFAULT_HOST)
    S.run(f"sed -i 's/\\r$//' {sh} && chmod +x {sh}", S.DEFAULT_HOST)
    code, out, err = S.run(
        f"pgrep -f '^bash {sh}' >/dev/null && echo ALREADY || "
        f"(setsid nohup bash {sh} > /tmp/c1_fix_bench.nohup 2>&1 < /dev/null & echo LAUNCHED)",
        S.DEFAULT_HOST)
    print(out.strip(), err.strip())


def launch_t527() -> None:
    for arm in ARMS:
        cmd = [sys.executable, str(ROOT / "deploy/t527/scripts/run_linux_v4_matrix.py"),
               "--models", str(T / f"{arm}_fix"), "--out", str(OUT / "t527" / arm),
               "--threads", "4", "--warmup", "50", "--iters", "1000"]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def pull_h618() -> None:
    S = ssh("h618")
    code, out, _ = S.run(f"tail -3 {H_OUT}/status.txt", S.DEFAULT_HOST)
    print(out)
    for arm in ARMS:
        local = OUT / "h618" / arm
        local.mkdir(parents=True, exist_ok=True)
        code, names, _ = S.run(f"ls {H_OUT}/{arm}/ 2>/dev/null", S.DEFAULT_HOST)
        rows = []
        for n in names.split():
            S.get(f"{H_OUT}/{arm}/{n}", str(local / n), S.DEFAULT_HOST)
            m = re.search(r"^RESULT\s+(.*)$", (local / n).read_text(errors="replace"), re.M)
            if not m:
                continue
            d = dict(t.split("=", 1) for t in m.group(1).split() if "=" in t)
            stem, dt = n[:-4].rsplit("_", 1)
            rows.append({"model": stem, "dtype": dt, "latency_us_p50": d["latency_us_p50"],
                         "latency_us_mean": d["latency_us_mean"], "latency_us_p95": d["latency_us_p95"],
                         "latency_us_p99": d.get("latency_us_p99", ""), "input": d["input"], "threads": 4})
        if rows:
            with (local / "matrix_summary.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print(f"{arm}: {len(rows)} rows", flush=True)


def compare() -> None:
    import pandas as pd

    relu = {k for k, v in __import__("json").loads(
        (OUT / "rebuild_summary.json").read_text(encoding="utf-8")).items() if v["relu"]}
    out = []
    for board in ("h618", "t527"):
        for arm in ARMS:
            nf = OUT / board / arm / "matrix_summary.csv"
            of = OLD[board][arm]
            if not nf.exists():
                continue
            new = pd.read_csv(nf)
            old = pd.read_csv(of) if of.exists() else pd.DataFrame(columns=["model", "dtype", "latency_us_p50"])
            m = new.merge(old[["model", "dtype", "latency_us_p50"]], on=["model", "dtype"],
                          how="left", suffixes=("_new", "_old"))
            m["board"], m["arm"] = board, arm
            m["relu_rebake"] = m.model.isin(relu)
            out.append(m)
    if not out:
        print("nothing to compare")
        return
    d = pd.concat(out)
    d["rel_change_pct"] = 100 * (d.latency_us_p50_new / d.latency_us_p50_old - 1)
    cols = ["board", "arm", "model", "dtype", "relu_rebake", "latency_us_p50_old", "latency_us_p50_new",
            "rel_change_pct"]
    d[cols].to_csv(OUT / "latency_compare.csv", index=False)
    pd.set_option("display.width", 250)
    print(d[cols].round(2).to_string(index=False))
    print(d.groupby(["board", "relu_rebake", "dtype"]).rel_change_pct.describe().round(2).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=("h618", "t527"))
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    if a.launch and a.board == "h618":
        launch_h618(upload=not a.no_upload)
    if a.launch and a.board == "t527":
        launch_t527()
    if a.pull and a.board == "h618":
        pull_h618()
    if a.compare:
        compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
