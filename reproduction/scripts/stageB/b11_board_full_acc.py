#!/usr/bin/env python3
"""Board NCNN FP32 / INT8 top-1 on the full FER+ test split (n=3153), H618 and T527.

Replaces the match@500 proxy (FP32-vs-INT8 agreement on a 500-image pool) with true
test accuracy for the FER+ C1 subnets: FLOPs arm (6), H618-searched LUT arm (6) and
the two T527-only tiers. Host preprocessing reproduces the GPU eval path of
train/train_subnet_distill.py (112 px RGB, /255, ImageNet mean/std, then adapt_input:
channel mean for the gray subnets and F.interpolate bilinear without antialiasing to
the subnet input). The baked NCNN graphs contain no input normalisation, so this has
to happen here. --prep pil keeps the first (PIL luma + antialiased resize) variant.
Each model runs in fer_ncnn_acc --fp32-only mode (which loads any param/bin and dumps
per-image logits), once for the FP32 and once for the INT8 artifact.

Usage:
  python scripts/stageB/b11_board_full_acc.py --prepare-only
  python scripts/stageB/b11_board_full_acc.py --boards t527
  python scripts/stageB/b11_board_full_acc.py --boards h618
  python scripts/stageB/b11_board_full_acc.py --summarize-only
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "data/ferplus112"
PREP = "train"
CACHE = ROOT / "data/board_acc/trainprep"
OUT = ROOT / "results/stageB/board_full_acc"
ACC_BIN = ROOT / "deploy/ncnn_edge/build-linux-aarch64-acc/fer_ncnn_acc"
REMOTE = "/root/fer-dynedge/accfull"
N_TEST = 3153

T = ROOT / "deploy/t527/models"
MODELS: list[tuple[str, str, Path]] = (
    [("flops", s, T / "v4_flops_ctrl_baked") for s in (
        "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_flops_ctrl_img64",
        "subnet_ferplus_w0.35_d2332_s64g_seed42_v4_flops_ctrl_img64",
        "subnet_ferplus_w0.35_d2432_s96c_seed42_v4_flops_ctrl_img96",
        "subnet_ferplus_w0.75_d3322_s64g_seed42_v4_flops_ctrl_img64",
        "subnet_ferplus_w0.75_d3432_s96c_seed42_v4_flops_ctrl_img96",
        "subnet_ferplus_w1_d4342_s112c_seed42_v4_flops_ctrl_img112")]
    + [("lut_h618", s, T / "v4_formal_baked") for s in (
        "subnet_ferplus_w0.35_d3222_s64g_seed42_v4_formal_lut_img64",
        "subnet_ferplus_w0.5_d3332_s64g_seed42_v4_formal_lut_img64",
        "subnet_ferplus_w0.5_d3242_s96c_seed42_v4_formal_lut_img96",
        "subnet_ferplus_w0.5_d3333_s96c_seed42_v4_formal_lut_img96",
        "subnet_ferplus_w0.75_d3333_s112c_seed42_v4_formal_lut_img112",
        "subnet_ferplus_w1_d4342_s112c_seed42_v4_formal_lut_img112")]
    + [("lut_t527", s, T / "v4_formal_lut_t527_baked") for s in (
        "subnet_ferplus_w0.5_d3322_s96c_seed42_v4_formal_lut_t527_img96",
        "subnet_ferplus_w0.75_d3333_s96c_seed42_v4_formal_lut_t527_img96")]
)
EXPW_MODELS: list[tuple[str, str, Path]] = (
    [("flops", s, T / "v4_flops_ctrl_baked") for s in (
        "subnet_expw_w0.35_d2222_s64g_seed42_v4_flops_ctrl_img64",
        "subnet_expw_w0.35_d2342_s64g_seed42_v4_flops_ctrl_img64",
        "subnet_expw_w0.5_d3344_s64g_seed42_v4_flops_ctrl_img64",
        "subnet_expw_w0.5_d3322_s96c_seed42_v4_flops_ctrl_img96",
        "subnet_expw_w0.5_d3333_s112c_seed42_v4_flops_ctrl_img112",
        "subnet_expw_w1_d4342_s112c_seed42_v4_flops_ctrl_img112")]
    + [("lut_h618", s, T / "v4_formal_baked_expw") for s in (
        "subnet_expw_w0.35_d2222_s64g_seed42_v4_formal_lut_img64",
        "subnet_expw_w0.5_d3332_s64g_seed42_v4_formal_lut_img64",
        "subnet_expw_w0.5_d3322_s96c_seed42_v4_formal_lut_img96",
        "subnet_expw_w0.5_d3324_s96c_seed42_v4_formal_lut_img96",
        "subnet_expw_w0.75_d3332_s96c_seed42_v4_formal_lut_img96",
        "subnet_expw_w1_d2222_s96c_seed42_v4_formal_lut_img96",
        "subnet_expw_w0.5_d4322_s112c_seed42_v4_formal_lut_img112",
        "subnet_expw_w1_d4332_s112c_seed42_v4_formal_lut_img112",
        "subnet_expw_w1_d4432_s112c_seed42_v4_formal_lut_img112",
        "subnet_expw_w1_d4442_s112c_seed42_v4_formal_lut_img112",
        "subnet_expw_w1_d4444_s112c_seed42_v4_formal_lut_img112")]
)
IMG_GLOB = "*.png"
COMBOS = {"64g": (64, True), "96c": (96, False), "112c": (112, False)}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
FIELDS = ["board", "arm", "model", "dtype", "combo", "n", "n_nonfinite", "acc", "seconds"]


def combo_of(stem: str) -> str:
    return "64g" if "_s64g_" in stem else ("96c" if "_s96c_" in stem else "112c")


def ssh_module(board: str):
    path = ROOT / f"deploy/{board}/scripts/ssh_{board}.py"
    spec = importlib.util.spec_from_file_location(f"ssh_{board}", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"ssh_{board}"] = m
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


def test_items() -> list[tuple[Path, int]]:
    test = TEST_ROOT / "test"
    if not test.exists() and TEST_ROOT.name == "ferplus112":
        subprocess.run([sys.executable, str(ROOT / "datasets/prepare_ferplus.py"),
                        "--fer2013-csv", str(ROOT / "datasets/fer2013/fer2013.csv"),
                        "--ferplus-csv", str(ROOT / "datasets/FERPlus/fer2013new.csv"),
                        "--out-dir", str(TEST_ROOT)], check=True)
    items = []
    for d in sorted(p for p in test.iterdir() if p.is_dir()):
        label = int(d.name.split("_", 1)[0])
        items += [(f, label) for f in sorted(d.glob(IMG_GLOB))]
    if len(items) != N_TEST:
        raise RuntimeError(f"{test} has {len(items)} images, expected {N_TEST}")
    return items


def preprocess(p: Path, hw: int, gray: bool) -> np.ndarray:
    if PREP == "pil":
        img = Image.open(p).convert("L" if gray else "RGB").resize((hw, hw), Image.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
        chw = np.stack([a, a, a], 0) if gray else a.transpose(2, 0, 1)
        return (chw - MEAN) / STD
    import torch
    import torch.nn.functional as F

    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy((a.transpose(2, 0, 1) - MEAN) / STD)[None]
    if gray:
        x = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    if x.shape[-1] != hw or x.shape[-2] != hw:
        x = F.interpolate(x, size=(hw, hw), mode="bilinear", align_corners=False)
    return x[0].numpy()


def prepare() -> None:
    items = test_items()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "labels.npy", np.array([lab for _, lab in items], dtype=np.int64))
    (CACHE / "names.txt").write_text("\n".join(str(p.relative_to(TEST_ROOT)) for p, _ in items),
                                     encoding="utf-8")
    for combo, (hw, gray) in COMBOS.items():
        tar_path = CACHE / f"{combo}.tar"
        if tar_path.exists():
            continue
        tmp = tar_path.with_suffix(".tar.part")
        with tarfile.open(tmp, "w") as tf:
            for i, (p, _) in enumerate(items):
                chw = preprocess(p, hw, gray)
                data = np.ascontiguousarray(chw, dtype=np.float32).tobytes()
                info = tarfile.TarInfo(f"{combo}/{i:05d}.bin")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        tmp.replace(tar_path)
        print(f"prepared {combo}: {tar_path.stat().st_size / 1e6:.0f} MB", flush=True)


def parse_logits(text: str) -> tuple[np.ndarray, int]:
    preds = np.full(N_TEST, -1, dtype=np.int64)
    bad = 0
    for line in text.splitlines():
        if not line.startswith("LOGITS "):
            continue
        f = dict(tok.split("=", 1) for tok in line.split()[1:4])
        idx = int(f["idx"])
        vals = np.array([float(v) for v in line.rsplit("vals=", 1)[1].split(",")])
        if f["finite"] != "1" or not np.all(np.isfinite(vals)):
            bad += 1
            continue
        preds[idx] = int(vals.argmax())
    return preds, bad


def stream_untar(S, tar_path: Path, remote_dir: str) -> None:
    """Pipe the tar into `tar -x` on the board; the T527 has no room for tar + extracted copy."""
    c = S.connect(S.DEFAULT_HOST)
    try:
        ch = c.get_transport().open_session()
        ch.exec_command(f"cd {remote_dir} && tar -xf -")
        with tar_path.open("rb") as f:
            while chunk := f.read(1 << 20):
                ch.sendall(chunk)
        ch.shutdown_write()
        code = ch.recv_exit_status()
        err = ch.recv_stderr(65536).decode("utf-8", "replace") if ch.recv_stderr_ready() else ""
        if code != 0:
            raise RuntimeError(f"remote untar failed ({code}): {err}")
    finally:
        c.close()


def run_board(board: str, threads: int) -> None:
    S = ssh_module(board)
    labels = np.load(CACHE / "labels.npy")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preds").mkdir(exist_ok=True)
    rows_csv = OUT / f"{board}.csv"
    rows = list(csv.DictReader(rows_csv.open(encoding="utf-8"))) if rows_csv.exists() else []
    done = {(r["model"], r["dtype"]) for r in rows}

    S.run(f"mkdir -p {REMOTE}/models {REMOTE}/bin", S.DEFAULT_HOST)
    S.put(str(ACC_BIN), f"{REMOTE}/bin/fer_ncnn_acc", S.DEFAULT_HOST)
    S.run(f"chmod +x {REMOTE}/bin/fer_ncnn_acc", S.DEFAULT_HOST)
    S.run("for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $f; done",
          S.DEFAULT_HOST)

    for combo, (hw, _) in COMBOS.items():
        todo = [(arm, s, d) for arm, s, d in MODELS if combo_of(s) == combo
                and not {(s, "fp32"), (s, "int8")} <= done]
        if not todo:
            continue
        rdir = f"{REMOTE}/{combo}"
        code, out, _ = S.run(f"ls {rdir} 2>/dev/null | wc -l", S.DEFAULT_HOST)
        if int(out.strip() or 0) != N_TEST:
            S.run(f"rm -rf {rdir}", S.DEFAULT_HOST)
            print(f"[{board}] upload {combo}.tar", flush=True)
            stream_untar(S, CACHE / f"{combo}.tar", REMOTE)
            S.run(f"ls {rdir}/*.bin | sort > {REMOTE}/{combo}.list", S.DEFAULT_HOST, timeout=900)
        for arm, stem, mdir in todo:
            for dtype in ("fp32", "int8"):
                if (stem, dtype) in done:
                    continue
                suf = "" if dtype == "fp32" else "_int8"
                param, binp = mdir / f"{stem}{suf}.param", mdir / f"{stem}{suf}.bin"
                rparam, rbin = f"{REMOTE}/models/{param.name}", f"{REMOTE}/models/{binp.name}"
                S.put(str(param), rparam, S.DEFAULT_HOST)
                S.put(str(binp), rbin, S.DEFAULT_HOST)
                t0 = time.time()
                code, text, err = S.run(
                    f"{REMOTE}/bin/fer_ncnn_acc --fp32-only --fp32-param {rparam} --fp32-bin {rbin} "
                    f"--list {REMOTE}/{combo}.list --h {hw} --w {hw} --c 3 --threads {threads}",
                    S.DEFAULT_HOST, timeout=1800)
                dt = time.time() - t0
                if board == "t527":
                    S.run(f"rm -f {rparam} {rbin}", S.DEFAULT_HOST)
                preds, bad = parse_logits(text)
                ok = preds >= 0
                acc = float((preds[ok] == labels[ok]).mean()) if ok.any() else float("nan")
                np.save(OUT / "preds" / f"{board}_{stem}_{dtype}.npy", preds)
                row = {"board": board, "arm": arm, "model": stem, "dtype": dtype, "combo": combo,
                       "n": int(ok.sum()), "n_nonfinite": bad, "acc": round(acc, 6),
                       "seconds": round(dt, 1)}
                rows.append(row)
                done.add((stem, dtype))
                with rows_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDS)
                    w.writeheader()
                    w.writerows(rows)
                print(f"[{board}] {stem} {dtype} n={row['n']} bad={bad} acc={acc:.4f} ({dt:.0f}s)",
                      flush=True)
        if board == "t527":
            S.run(f"rm -rf {rdir} {REMOTE}/{combo}.list", S.DEFAULT_HOST)


def summarize() -> None:
    import pandas as pd

    gpu = pd.read_csv(ROOT / "results/c1_flops_vs_lut/points.csv")[["model", "test_acc_gpu"]] \
        .drop_duplicates("model").set_index("model").test_acc_gpu.to_dict()
    for arm, s, _ in MODELS:
        if s not in gpu:
            j = ROOT / "results/stageB/subnets_v4_formal_t527" / (
                s.replace("_img96", "").replace("_img112", "").replace("_img64", "") + ".json")
            if j.exists():
                gpu[s] = json.loads(j.read_text(encoding="utf-8"))["test_acc"]
    labels = np.load(CACHE / "labels.npy")
    out = []
    for f in sorted(OUT.glob("*.csv")):
        if f.name.startswith("summary"):
            continue
        d = pd.read_csv(f)
        for (board, model), g in d.groupby(["board", "model"]):
            r = {"board": board, "arm": g.arm.iloc[0], "model": model,
                 "gpu_acc": gpu.get(model, np.nan)}
            for _, x in g.iterrows():
                r[f"{x['dtype']}_acc"] = x["acc"]
            pf = OUT / "preds" / f"{board}_{model}_fp32.npy"
            pi = OUT / "preds" / f"{board}_{model}_int8.npy"
            if pf.exists() and pi.exists():
                a, b = np.load(pf), np.load(pi)
                ok = (a >= 0) & (b >= 0)
                r["fp32_int8_agree"] = round(float((a[ok] == b[ok]).mean()), 6)
            out.append(r)
    s = pd.DataFrame(out)
    if not len(s):
        return
    s["int8_minus_fp32_pp"] = 100 * (s.int8_acc - s.fp32_acc)
    s["fp32_minus_gpu_pp"] = 100 * (s.fp32_acc - s.gpu_acc)
    if {"t527", "h618"} <= set(s.board):
        xb = []
        for m in s.model.unique():
            for dt in ("fp32", "int8"):
                p = [OUT / "preds" / f"{b}_{m}_{dt}.npy" for b in ("t527", "h618")]
                if all(x.exists() for x in p):
                    a, b = np.load(p[0]), np.load(p[1])
                    xb.append({"model": m, "dtype": dt, "cross_board_agree": float((a == b).mean())})
        pd.DataFrame(xb).to_csv(OUT / "summary_cross_board.csv", index=False)
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.round(4).to_string(index=False))
    print(s.groupby("board")[["int8_minus_fp32_pp", "fp32_minus_gpu_pp", "fp32_int8_agree"]]
          .describe().round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--fixed", action="store_true",
                    help="evaluate the B12 rebuilds (*_fix dirs: ReLU 64g, ImageNet-calibrated INT8)")
    ap.add_argument("--dataset", choices=("ferplus", "expw"), default="ferplus",
                    help="expw: data/expw_faithful test split (n=9094), the 17 ExpW C1 subnets")
    ap.add_argument("--prep", choices=("train", "pil"), default="train",
                    help="pil: earlier PIL-luma/antialiased inputs (results under *_pilprep)")
    args = ap.parse_args()
    global OUT, MODELS, TEST_ROOT, CACHE, REMOTE, N_TEST, IMG_GLOB, PREP
    PREP = args.prep
    if PREP == "pil":
        CACHE = ROOT / "data/board_acc"
    if args.dataset == "expw":
        TEST_ROOT = ROOT / "data/expw_faithful/aligned"
        CACHE = CACHE / "expw"
        REMOTE = REMOTE + "_expw"
        N_TEST = 9094
        IMG_GLOB = "*.jpg"
        OUT = OUT / "expw"
        MODELS = EXPW_MODELS
    if args.fixed:
        OUT = OUT / ("fixed" if PREP == "train" else "fixed_pilprep")
        MODELS = [(a, s, d.with_name(d.name + "_fix")) for a, s, d in MODELS]
    if not args.summarize_only:
        prepare()
        if args.prepare_only:
            return 0
        for b in [x for x in args.boards.split(",") if x]:
            run_board(b, args.threads)
    summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
