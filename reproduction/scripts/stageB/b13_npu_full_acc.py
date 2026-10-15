#!/usr/bin/env python3
"""B13: T527 NPU (VIPLite uint8 NBG) top-1 on the full FER+ test split (n=3153).

Inputs come from the B11 cache (data/board_acc/<combo>.tar: PIL resize, gray->3ch for
64g, /255, ImageNet mean/std), quantised with each NBG's input scale/zero-point from
*_nbg_meta.json. Per model: stream the uint8 tensors to the board, run an on-board
vpm_run loop detached (setsid), poll, pull the per-image logits, delete the inputs.

Usage:
  python scripts/stageB/b13_npu_full_acc.py --nbg-dir deploy/t527/models/v4_npu_fix --tag fix
  python scripts/stageB/b13_npu_full_acc.py --nbg-dir deploy/t527/models/v4_npu --tag old_div255 --div255
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/stageB"))
from b11_board_full_acc import CACHE, MEAN, N_TEST, STD, combo_of, ssh_module, stream_untar  # noqa: E402

OUT = ROOT / "results/stageB/npu_full_acc"
REMOTE = "/root/fer-dynedge/npuacc"
LOOP = """#!/bin/sh
D=$1
cd $D || exit 1
: > preds.txt
while read f; do
  printf 'vpm_run is a framework for testing VIPLite driver.\\n\\n[network]\\n%s/network_binary.nb\\n[input]\\n%s\\n\\n' "$D" "$f" > sample.txt
  rm -f output_0.txt
  vpm_run -s sample.txt -l 1 -b 0 --save_txt 1 > /dev/null 2>&1
  echo "$(basename $f .dat) $(tr '\\n' ' ' < output_0.txt 2>/dev/null)" >> preds.txt
done < list.txt
echo DONE > done.txt
"""
FIELDS = ["tag", "model", "combo", "n", "n_bad", "acc", "input_scale", "input_zp", "seconds"]


def build_tar(stem: str, scale: float, zp: int, div255: bool, limit: int) -> bytes:
    combo = combo_of(stem)
    buf = io.BytesIO()
    with tarfile.open(CACHE / f"{combo}.tar") as src, tarfile.open(fileobj=buf, mode="w") as dst:
        for k, m in enumerate(src):
            if limit and k >= limit:
                break
            x = np.frombuffer(src.extractfile(m).read(), np.float32).reshape(3, -1)
            if div255:
                x = x * STD.reshape(3, 1) + MEAN.reshape(3, 1)
            q = np.clip(np.rint(x / scale + zp), 0, 255).astype(np.uint8).tobytes()
            info = tarfile.TarInfo(f"in/{Path(m.name).stem}.dat")
            info.size = len(q)
            dst.addfile(info, io.BytesIO(q))
    return buf.getvalue()


def parse(text: str) -> tuple[np.ndarray, int]:
    preds = np.full(N_TEST, -1, dtype=np.int64)
    bad = 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            vals = np.array([float(v) for v in parts[1:]])
        except ValueError:
            vals = np.array([])
        if vals.size < 2 or not np.isfinite(vals).all():
            bad += 1
            continue
        preds[int(parts[0])] = int(vals.argmax())
    return preds, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbg-dir", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--div255", action="store_true", help="feed /255-only inputs (old NBG convention)")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    a = ap.parse_args()

    S = ssh_module("t527")
    labels = np.load(CACHE / "labels.npy")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preds").mkdir(exist_ok=True)
    rows_csv = OUT / f"{a.tag}.csv"
    rows = list(csv.DictReader(rows_csv.open(encoding="utf-8"))) if rows_csv.exists() else []
    done = {r["model"] for r in rows}
    nbg_dir = a.nbg_dir if a.nbg_dir.is_absolute() else ROOT / a.nbg_dir

    S.run(f"mkdir -p {REMOTE}", S.DEFAULT_HOST)
    with (OUT / "_loop.sh").open("w", newline="\n") as f:
        f.write(LOOP)
    S.put(str(OUT / "_loop.sh"), f"{REMOTE}/loop.sh", S.DEFAULT_HOST)
    S.run(f"chmod +x {REMOTE}/loop.sh; chmod 666 /dev/vipcore 2>/dev/null; true", S.DEFAULT_HOST)

    for nb in sorted(nbg_dir.glob("*_uint8.nb")):
        stem = nb.name.replace("_uint8.nb", "")
        if (a.only and a.only not in stem) or (stem in done and not a.limit):
            continue
        meta = json.loads((nbg_dir / f"{stem}_nbg_meta.json").read_text(encoding="utf-8"))
        q = next(iter(meta["Inputs"].values()))["quantize"]
        scale, zp = float(q["scale"]), int(q["zero_point"])
        rdir = f"{REMOTE}/{a.tag}_{combo_of(stem)}_{abs(hash(stem)) % 10**6}"
        S.run(f"rm -rf {rdir}; mkdir -p {rdir}", S.DEFAULT_HOST)
        S.put(str(nb), f"{rdir}/network_binary.nb", S.DEFAULT_HOST)
        tar_p = OUT / "_inputs.tar"
        tar_p.write_bytes(build_tar(stem, scale, zp, a.div255, a.limit))
        stream_untar(S, tar_p, rdir)
        tar_p.unlink()
        S.run(f"cd {rdir} && ls {rdir}/in/*.dat | sort > list.txt", S.DEFAULT_HOST)
        t0 = time.time()
        S.run(f"setsid nohup sh {REMOTE}/loop.sh {rdir} > {rdir}/loop.log 2>&1 < /dev/null &", S.DEFAULT_HOST)
        print(f"[{a.tag}] {stem} scale={scale:.5f} zp={zp} launched", flush=True)
        while True:
            time.sleep(30)
            try:
                code, out, _ = S.run(f"test -f {rdir}/done.txt && echo DONE; wc -l < {rdir}/preds.txt",
                                     S.DEFAULT_HOST)
            except Exception as e:  # noqa: BLE001
                print(f"  poll error {e}", flush=True)
                continue
            if "DONE" in out:
                break
            print(f"  {out.strip()} / {a.limit or N_TEST}", flush=True)
        dt = time.time() - t0
        S.get(f"{rdir}/preds.txt", str(OUT / "_preds.txt"), S.DEFAULT_HOST)
        preds, bad = parse((OUT / "_preds.txt").read_text(encoding="utf-8", errors="replace"))
        S.run(f"rm -rf {rdir}", S.DEFAULT_HOST)
        ok = preds >= 0
        acc = float((preds[ok] == labels[ok]).mean()) if ok.any() else float("nan")
        print(f"[{a.tag}] {stem} n={int(ok.sum())} bad={bad} acc={acc:.4f} ({dt:.0f}s)", flush=True)
        if a.limit:
            continue
        np.save(OUT / "preds" / f"{a.tag}_{stem}.npy", preds)
        rows.append({"tag": a.tag, "model": stem, "combo": combo_of(stem), "n": int(ok.sum()),
                     "n_bad": bad, "acc": round(acc, 6), "input_scale": scale, "input_zp": zp,
                     "seconds": round(dt, 1)})
        with rows_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
