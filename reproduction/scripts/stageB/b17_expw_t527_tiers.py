#!/usr/bin/env python3
"""B17: NCNN artifacts and T527 latency for the ExpW faithful T527-searched tiers.

Same export and protocol as B12 (fixed bake: ReLU for 64g students, INT8 calibrated
with ImageNet mean/std; T527 NCNN 4 threads, warmup 50, 1000 iters). Checkpoints are
pulled from the Orin into results/stageB/subnets_v4_formal_t527/; every tier with a
local *_faithful_best.pt is exported (already exported stems are skipped).

  python scripts/stageB/b17_expw_t527_tiers.py            # export + INT8
  python scripts/stageB/b17_expw_t527_tiers.py --bench    # then bench on the T527 (over SSH)
Outputs: deploy/t527/models/v4_formal_lut_t527_faithful_expw/,
         results/stageB/expw_t527_faithful/{export_summary.json, t527/matrix_summary.csv}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAG = "v4_formal_lut_t527_faithful"
CKPT = ROOT / "results/stageB/subnets_v4_formal_t527"
DST = ROOT / "deploy/t527/models/v4_formal_lut_t527_faithful_expw"
OUT = ROOT / "results/stageB/expw_t527_faithful"
CALIB64_TRAIN = ROOT / "data/calib/ferplus_mcu64/train"
CALIB112 = ROOT / "data/calib/linux112/calib"

_spec = importlib.util.spec_from_file_location(
    "prep_baked", ROOT / "deploy/t527/scripts/prep_v4_ncnn_baked_pnnx.py")
prep = importlib.util.module_from_spec(_spec)
sys.modules["prep_baked"] = prep
assert _spec.loader is not None
_spec.loader.exec_module(prep)


def stems() -> list[str]:
    out = []
    for p in sorted(CKPT.glob(f"subnet_expw_*_seed42_{TAG}_best.pt")):
        base = p.name[: -len("_best.pt")]
        img = int(prep.parse_stem(base + "_")[3])
        out.append(f"{base}_img{img}")
    return out


def export() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    prep.OUT_DIR, prep.CKPT_DIR = DST, CKPT
    l64 = prep.ensure_imagelist(CALIB64_TRAIN, DST / "imagelist_calib64_train.txt")
    l112 = prep.ensure_imagelist(CALIB112, DST / "imagelist_real.txt")
    summ_p = OUT / "export_summary.json"
    summ = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {}
    for stem in stems():
        if summ.get(stem, {}).get("status") == "ok" and (DST / f"{stem}_int8.param").exists():
            print(f"skip {stem}")
            continue
        relu = prep.parse_stem(stem)[4]
        t0 = time.time()
        rec: dict = {"relu": relu}
        print(f"== {stem} relu={relu}", flush=True)
        try:
            param, binp, c = prep.export_one(stem, OUT / "work" / stem, ckpt_tag=TAG, relu=relu)
            rec["ncnn_vs_baked_cos"] = c
            ptxt = param.read_text(encoding="utf-8").splitlines()
            rec["n_hardswish"] = sum(l.startswith("HardSwish") for l in ptxt)
            rec["n_relu"] = sum(l.startswith("ReLU") for l in ptxt)
            if relu and rec["n_hardswish"]:
                raise RuntimeError(f"relu export still has {rec['n_hardswish']} HardSwish layers")
            prep.build_int8(stem, param, binp, l64, l112)
            rec["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"fail:{e}"
        rec["sec"] = round(time.time() - t0, 1)
        summ[stem] = rec
        summ_p.write_text(json.dumps(summ, indent=2), encoding="utf-8")
        print(f"   -> {rec['status']} ({rec['sec']}s)", flush=True)
    bad = [k for k, v in summ.items() if v["status"] != "ok"]
    print(f"export ok={len(summ) - len(bad)} fail={len(bad)} {bad}")
    return 1 if bad else 0


def bench() -> None:
    cmd = [sys.executable, str(ROOT / "deploy/t527/scripts/run_linux_v4_matrix.py"),
           "--models", str(DST), "--out", str(OUT / "t527"),
           "--threads", "4", "--warmup", "50", "--iters", "1000"]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    rc = export()
    if a.bench and rc == 0:
        bench()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
