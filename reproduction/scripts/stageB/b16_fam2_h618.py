#!/usr/bin/env python3
"""B16 H618 leg: upload the ResNet-family NCNN models, launch the detached bench, pull results.

  python scripts/stageB/b16_fam2_h618.py --launch      # upload + start h618_fam2_bench.sh
  python scripts/stageB/b16_fam2_h618.py --pull        # -> results/stageB/fam2/h618_latency.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "deploy/ncnn_edge/fam2"
OUT = ROOT / "results/stageB/fam2"
H_ROOT = "/root/fer-dynedge"
H_MODELS = f"{H_ROOT}/deploy/ncnn_edge/models/fam2"
H_OUT = f"{H_ROOT}/results/fam2_bench"
SH = f"{H_ROOT}/scripts/stageB/h618_fam2_bench.sh"


def ssh():
    spec = importlib.util.spec_from_file_location("ssh_h618", ROOT / "deploy/h618/scripts/ssh_h618.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ssh_h618"] = m
    spec.loader.exec_module(m)
    return m


def launch(S, upload: bool) -> None:
    host = S.DEFAULT_HOST
    if upload:
        S.run(f"mkdir -p {H_MODELS} {H_ROOT}/scripts/stageB", host)
        files = sorted(p for p in LOCAL.iterdir() if p.suffix in {".param", ".bin"})
        for i, p in enumerate(files):
            S.put(str(p), f"{H_MODELS}/{p.name}", host)
            if i % 40 == 0:
                print(f"uploaded {i + 1}/{len(files)}", flush=True)
    S.put(str(ROOT / "scripts/stageB/h618_fam2_bench.sh"), SH, host)
    S.run(f"sed -i 's/\\r$//' {SH} && chmod +x {SH}", host)
    _, out, err = S.run(f"pgrep -f '^bash {SH}' >/dev/null && echo ALREADY || "
                        f"(setsid nohup bash {SH} > /tmp/fam2_bench.nohup 2>&1 < /dev/null & echo LAUNCHED)", host)
    print(out.strip(), err.strip())


def pull(S) -> None:
    host = S.DEFAULT_HOST
    _, out, _ = S.run(f"tail -2 {H_OUT}/status.txt; grep -c '^OK' {H_OUT}/status.txt", host)
    print(out)
    _, txt, _ = S.run(f"grep -H '^RESULT' {H_OUT}/*.txt", host)
    rows = []
    for line in txt.splitlines():
        path, res = line.split(":", 1)
        name = Path(path).stem
        arch, dt = name.rsplit("_", 1)
        d = dict(t.split("=", 1) for t in res.split()[1:] if "=" in t)
        rows.append({"arch": arch, "precision": dt, "latency_us_p50": d["latency_us_p50"],
                     "latency_us_mean": d["latency_us_mean"], "latency_us_p95": d["latency_us_p95"],
                     "latency_us_p99": d.get("latency_us_p99", "")})
    rows.sort(key=lambda r: (r["arch"], r["precision"]))
    with (OUT / "h618_latency.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT / 'h618_latency.csv'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    S = ssh()
    if a.launch:
        launch(S, upload=not a.no_upload)
    if a.pull:
        pull(S)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
