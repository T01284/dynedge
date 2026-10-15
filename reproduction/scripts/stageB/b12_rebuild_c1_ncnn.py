#!/usr/bin/env python3
"""B12: rebuild the C1 NCNN artifacts with the two export fixes.

1) 64g students were trained with ReLU but the old bake kept Hardswish:
   re-export FP32 with ReLU (prep_v4_ncnn_baked_pnnx.export_one, fixed).
2) INT8 tables were calibrated on /255 inputs instead of ImageNet mean/std:
   recalibrate every model (Hardswish FP32 graphs are copied unchanged).

Outputs go to deploy/t527/models/<old_dir>_fix (the old dirs stay untouched)
and a summary to results/stageB/c1_fix/rebuild_summary.json.

Usage:
  python scripts/stageB/b12_rebuild_c1_ncnn.py [--only STEM_SUBSTR] [--skip-int8]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_g1_c1_int8_acc import ARMS  # noqa: E402

T = ROOT / "deploy/t527/models"
OUT = ROOT / "results/stageB/c1_fix"
CALIB64_TRAIN = ROOT / "data/calib/ferplus_mcu64/train"
CALIB112 = ROOT / "data/calib/linux112/calib"

_spec = importlib.util.spec_from_file_location(
    "prep_baked", ROOT / "deploy/t527/scripts/prep_v4_ncnn_baked_pnnx.py")
prep = importlib.util.module_from_spec(_spec)
sys.modules["prep_baked"] = prep
assert _spec.loader is not None
_spec.loader.exec_module(prep)

# (arm, stem, src_dir, ckpt_dir, ckpt_tag, relu)
JOBS: list[tuple[str, str, Path, Path, str, bool]] = []
for a in ARMS:
    for base, img, relu in a["stems"]:
        JOBS.append((a["name"], f"{base}_img{img}", a["out_t"], a["ckpt_dir"], a["ckpt_tag"], relu))
for s in ("subnet_ferplus_w0.5_d3322_s96c_seed42_v4_formal_lut_t527_img96",
          "subnet_ferplus_w0.75_d3333_s96c_seed42_v4_formal_lut_t527_img96"):
    JOBS.append(("lut_t527", s, T / "v4_formal_lut_t527_baked", Path(), "", False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summ_p = OUT / "rebuild_summary.json"
    summ = json.loads(summ_p.read_text(encoding="utf-8")) if summ_p.exists() else {}
    lists: dict[str, Path] = {}
    for arm, stem, src, ckpt_dir, tag, relu in JOBS:
        if args.only and args.only not in stem:
            continue
        dst = src.with_name(src.name + "_fix")
        dst.mkdir(parents=True, exist_ok=True)
        prep.OUT_DIR = dst
        if "64" not in lists:
            lists["64"] = prep.ensure_imagelist(CALIB64_TRAIN, dst / "imagelist_calib64_train.txt")
            lists["112"] = prep.ensure_imagelist(CALIB112, dst / "imagelist_real.txt")
        else:
            for k, n in (("64", "imagelist_calib64_train.txt"), ("112", "imagelist_real.txt")):
                if not (dst / n).exists():
                    shutil.copy2(lists[k], dst / n)
        t0 = time.time()
        rec = {"arm": arm, "src": str(src.relative_to(ROOT)), "dst": str(dst.relative_to(ROOT)),
               "relu": relu}
        print(f"== {arm} {stem} relu={relu}", flush=True)
        try:
            if relu:
                prep.CKPT_DIR = ckpt_dir
                _, _, c = prep.export_one(stem, ROOT / "results/stageB/c1_fix/work" / stem,
                                          ckpt_tag=tag, relu=True)
                rec["ncnn_vs_baked_cos"] = c
            else:
                for ext in (".param", ".bin"):
                    shutil.copy2(src / f"{stem}{ext}", dst / f"{stem}{ext}")
            ptxt = (dst / f"{stem}.param").read_text(encoding="utf-8")
            rec["n_hardswish"] = sum(l.startswith("HardSwish") for l in ptxt.splitlines())
            rec["n_relu"] = sum(l.startswith("ReLU") for l in ptxt.splitlines())
            if relu and rec["n_hardswish"]:
                raise RuntimeError(f"relu export still has {rec['n_hardswish']} HardSwish layers")
            if not args.skip_int8:
                prep.build_int8(stem, dst / f"{stem}.param", dst / f"{stem}.bin",
                                lists["64"], lists["112"])
            rec["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"fail:{e}"
        rec["sec"] = round(time.time() - t0, 1)
        summ[stem] = rec
        summ_p.write_text(json.dumps(summ, indent=2), encoding="utf-8")
        print(f"   -> {rec['status']} hs={rec.get('n_hardswish')} relu={rec.get('n_relu')} "
              f"({rec['sec']}s)", flush=True)
    bad = [k for k, v in summ.items() if v["status"] != "ok"]
    print(f"done ok={len(summ) - len(bad)} fail={len(bad)} {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
