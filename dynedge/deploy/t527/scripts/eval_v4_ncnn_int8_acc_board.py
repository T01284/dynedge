#!/usr/bin/env python3
"""T527 board: NCNN FP32↔INT8 (aciq) match on 500-image pool (H1 protocol).

Host preprocesses images → float32 NCHW bins; board fer_ncnn_acc compares
FP32 vs INT8 (host pyncnn FP32 is NaN on these Hardswish graphs).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_t527 import put, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "deploy" / "t527" / "models" / "v4_predistill"
EVAL = ROOT / "data" / "calib" / "linux112" / "eval"
CALIB = ROOT / "data" / "calib" / "linux112" / "calib"
ACC_BIN = ROOT / "deploy" / "ncnn_edge" / "build-linux-aarch64-acc" / "fer_ncnn_acc"
OUT_CSV = ROOT / "results" / "t527_int8_acc_500.csv"
OUT_JSON = ROOT / "results" / "t527_int8_acc_500.json"
OUT_LIST = ROOT / "results" / "t527_v4" / "eval500_imagelist.txt"
BIN_CACHE = ROOT / "results" / "t527_v4" / "eval500_bins"
REMOTE = "/root/fer-dynedge"
REMOTE_V4 = f"{REMOTE}/models/v4_predistill"
REMOTE_ACC = f"{REMOTE}/acc500"
REMOTE_BIN = f"{REMOTE_ACC}/bins"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def hw_of(stem: str) -> int:
    if "img64" in stem or "_s64g_" in stem:
        return 64
    if "img96" in stem or "_s96c_" in stem:
        return 96
    return 112


def folder_label(folder: str) -> str | None:
    name = folder.lower()
    if "_" in name and name.split("_", 1)[0].isdigit():
        name = name.split("_", 1)[1]
    return name


def list_pool(root: Path, source: str) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        parts = p.relative_to(root).parts
        if len(parts) < 2:
            continue
        lab = folder_label(parts[1])
        if lab is None:
            continue
        out.append(
            {
                "path": p,
                "dataset": parts[0],
                "label_name": lab,
                "source": source,
                "rel": str(p.relative_to(ROOT)).replace("\\", "/"),
                "key": f"{parts[0]}/{parts[1]}/{p.name}",
            }
        )
    return out


def build_eval500(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    held = list_pool(EVAL, "eval")
    calib = list_pool(CALIB, "calib")
    used = {it["rel"] for it in held}
    rest = [it for it in calib if it["rel"] not in used]
    random.shuffle(rest)
    by: dict[str, list[dict]] = {}
    for it in rest:
        by.setdefault(f"{it['dataset']}/{it['label_name']}", []).append(it)
    keys = sorted(by)
    topup: list[dict] = []
    i = 0
    need = max(0, n - len(held))
    while len(topup) < need and keys:
        progressed = False
        for k in keys:
            if i < len(by[k]):
                topup.append(by[k][i])
                progressed = True
                if len(topup) >= need:
                    break
        if not progressed:
            break
        i += 1
    pool = held + topup[:need]
    return pool[:n] if len(held) >= n else pool


def prep_bins(pool: list[dict], hws: list[int]) -> dict[int, list[str]]:
    """Write float32 NCHW bins; gray replicate for hw=64 (s64g protocol)."""
    BIN_CACHE.mkdir(parents=True, exist_ok=True)
    lists: dict[int, list[str]] = {hw: [] for hw in hws}
    for hw in hws:
        sub = BIN_CACHE / f"hw{hw}"
        sub.mkdir(parents=True, exist_ok=True)
        for it in pool:
            safe = it["key"].replace("/", "__").replace("\\", "__")
            out = sub / f"{safe}.bin"
            if not out.exists():
                if hw == 64:
                    g = (
                        np.asarray(
                            Image.open(it["path"]).convert("L").resize((hw, hw), Image.BILINEAR),
                            dtype=np.float32,
                        )
                        / 255.0
                    )
                    chw = np.stack([g, g, g], 0).astype(np.float32)
                else:
                    rgb = (
                        np.asarray(
                            Image.open(it["path"]).convert("RGB").resize((hw, hw), Image.BILINEAR),
                            dtype=np.float32,
                        )
                        / 255.0
                    )
                    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32)
                chw.tofile(out)
            lists[hw].append(out.name)
        (BIN_CACHE / f"list_hw{hw}.txt").write_text(
            "\n".join(f"{REMOTE_BIN}/hw{hw}/{n}" for n in lists[hw]) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return lists


def find_acc_bin() -> Path:
    if not ACC_BIN.exists():
        raise FileNotFoundError(f"missing fer_ncnn_acc; build via WSL cmake -> {ACC_BIN}")
    return ACC_BIN


def parse_result(text: str) -> dict:
    m = re.search(r"^RESULT\s+(.*)$", text, re.M)
    if not m:
        raise RuntimeError("no RESULT\n" + text[-2000:])
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def ensure_remote_bins(hws: list[int]) -> None:
    run(f"mkdir -p {REMOTE_BIN} {REMOTE_ACC}")
    put(str(find_acc_bin()), f"{REMOTE}/bin/fer_ncnn_acc")
    run(f"chmod +x {REMOTE}/bin/fer_ncnn_acc")
    for hw in hws:
        run(f"mkdir -p {REMOTE_BIN}/hw{hw}")
        # upload list
        put(str(BIN_CACHE / f"list_hw{hw}.txt"), f"{REMOTE_ACC}/list_hw{hw}.txt")
        # rsync-like: tar upload for speed
        local_tar = BIN_CACHE / f"hw{hw}.tar"
        if not local_tar.exists() or local_tar.stat().st_mtime < (BIN_CACHE / f"hw{hw}").stat().st_mtime:
            import tarfile

            with tarfile.open(local_tar, "w") as tar:
                for p in sorted((BIN_CACHE / f"hw{hw}").glob("*.bin")):
                    tar.add(p, arcname=p.name)
        put(str(local_tar), f"{REMOTE_ACC}/hw{hw}.tar")
        run(f"tar -xf {REMOTE_ACC}/hw{hw}.tar -C {REMOTE_BIN}/hw{hw}")


def eval_stem(stem: str) -> dict:
    hw = hw_of(stem)
    rparam = f"{REMOTE_V4}/{stem}.param"
    rbin = f"{REMOTE_V4}/{stem}.bin"
    rip = f"{REMOTE_V4}/{stem}_int8.param"
    rib = f"{REMOTE_V4}/{stem}_int8.bin"
    # ensure models present
    for local, remote in (
        (MODEL_DIR / f"{stem}.param", rparam),
        (MODEL_DIR / f"{stem}.bin", rbin),
        (MODEL_DIR / f"{stem}_int8.param", rip),
        (MODEL_DIR / f"{stem}_int8.bin", rib),
    ):
        if not local.exists():
            return {"model": stem, "status": f"missing:{local.name}"}
        put(str(local), remote)

    cmd = (
        f"{REMOTE}/bin/fer_ncnn_acc "
        f"--fp32-param {rparam} --fp32-bin {rbin} "
        f"--int8-param {rip} --int8-bin {rib} "
        f"--list {REMOTE_ACC}/list_hw{hw}.txt "
        f"--h {hw} --w {hw} --c 3 --threads 2"
    )
    code, out, err = run(cmd, timeout=3600)
    text = out + err
    if code != 0:
        return {"model": stem, "status": f"fail:{code}", "raw": text[-1500:]}
    d = parse_result(text)
    return {
        "model": stem,
        "status": "ok",
        "dataset": "ferplus" if "ferplus" in stem else "expw",
        "hw": hw,
        "gray": int(hw == 64),
        "method": "aciq",
        "fp32_ref": "ncnn_board",
        "int8_backend": "ncnn_board",
        "n_eval": int(d["n_eval"]),
        "fp32_int8_match": round(float(d["fp32_int8_match"]), 4),
        "logits_cosine_mean": round(float(d["logits_cosine_mean"]), 6),
        "nan_fp": int(d.get("nan_fp", 0)),
        "nan_i8": int(d.get("nan_i8", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-upload-bins", action="store_true")
    ap.add_argument("--prep-only", action="store_true")
    args = ap.parse_args()

    pool = build_eval500(args.n, args.seed)
    OUT_LIST.parent.mkdir(parents=True, exist_ok=True)
    OUT_LIST.write_text("\n".join(it["rel"] for it in pool) + "\n", encoding="utf-8")
    n_h = sum(1 for it in pool if it["source"] == "eval")
    n_c = sum(1 for it in pool if it["source"] == "calib")
    print(f"pool n={len(pool)} heldout={n_h} calib_topup={n_c}", flush=True)

    stems = sorted(p.stem for p in MODEL_DIR.glob("*.param") if "_int8" not in p.name)
    if args.only:
        stems = [s for s in stems if args.only in s]
    hws = sorted({hw_of(s) for s in stems})
    print(f"prep bins hws={hws}", flush=True)
    t0 = time.time()
    prep_bins(pool, hws)
    print(f"bins ready in {time.time()-t0:.1f}s -> {BIN_CACHE}", flush=True)
    if args.prep_only:
        return 0

    if not args.skip_upload_bins:
        print("upload bins + fer_ncnn_acc ...", flush=True)
        ensure_remote_bins(hws)

    rows = []
    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}", flush=True)
        r = eval_stem(stem)
        print(
            f"  status={r.get('status')} match={r.get('fp32_int8_match')} "
            f"cos={r.get('logits_cosine_mean')} n={r.get('n_eval')}",
            flush=True,
        )
        rows.append(r)

    fields = [
        "model",
        "status",
        "dataset",
        "hw",
        "gray",
        "method",
        "fp32_ref",
        "int8_backend",
        "n_eval",
        "fp32_int8_match",
        "logits_cosine_mean",
        "nan_fp",
        "nan_i8",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    OUT_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        mean_m = float(np.mean([r["fp32_int8_match"] for r in ok]))
        print(f"DONE ok={len(ok)}/{len(rows)} mean_match={mean_m:.4f} -> {OUT_CSV}")
    else:
        print(f"DONE no ok -> {OUT_CSV}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
