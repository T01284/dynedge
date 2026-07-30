#!/usr/bin/env python3
"""Orin C3 formal对照 — host NCNN decide + §8.29 TRT segment LUT latency.

Gate: same as §8.27 (host multi-exit NCNN + τ / c3_band / c3_randB / force_final).
Latency: mean(cum_exit_ms[exit]) from results/orin_v4/exit_segment_latency (TRT truncated).
No on-device MLP; no sample-level TRT early-exit runner (engines are per-exit truncated).

Usage:
  python deploy/orin/scripts/run_c3_formal_orin.py
  python deploy/orin/scripts/run_c3_formal_orin.py --n 80
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from pathlib import Path

import ncnn
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.latency_controller import LatencyBudgetController  # noqa: E402

NCNN_DIR = ROOT / "results/t527_v4/t3_early_exit/ncnn"
BIN_DIR = ROOT / "results/t527_v4/eval500_bins/hw112"
CTRL_BAND = ROOT / "results/from_autodl/formal_v4/controller_band"
FEED_JSON = ROOT / "results/orin_v4/exit_segment_latency/summary.json"
OUT_ROOT = ROOT / "results/c3_formal_board/orin"

LAMBDA = 0.2
TAUS = (0.7, 0.85, 0.9)
DEVICE = "orin"


def _fix_posix_path() -> None:
    try:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]
    except Exception:
        pass


def tag_budget(ms: float) -> str:
    return f"{ms:g}"


def load_feed() -> dict:
    return json.loads(FEED_JSON.read_text(encoding="utf-8"))["controller_feed"]


def band_keys(feed: dict) -> dict[str, float]:
    cum = [float(x) for x in feed["cum_exit_ms"]]
    lo, hi = min(cum), max(cum[-1], max(cum))
    return {"tight": round(lo, 2), "mid": round(0.5 * (lo + hi), 2), "loose": round(hi, 2)}


def ctrl_dir(kind: str, budget_ms: float | None) -> Path:
    if kind == "randB":
        # prefer orin-specific; fallback none
        cands = [
            CTRL_BAND / "ferplus_orin_randB_l0p2",
            CTRL_BAND / "ferplus_orin_randB_l0p2_orin",
        ]
        for d in CTRL_BAND.glob("ferplus_orin_randB*"):
            cands.insert(0, d)
        for d in cands:
            if d.is_dir() and list(d.glob("controller_*_best.pt")):
                return d
        raise FileNotFoundError("no orin randB controller under controller_band")
    assert budget_ms is not None
    # tags: ferplus_orin_b1.06ms_l0p2 or with _orin suffix in folder
    exact = CTRL_BAND / f"ferplus_orin_b{tag_budget(budget_ms)}ms_l0p2"
    if exact.is_dir():
        return exact
    for d in sorted(CTRL_BAND.glob(f"ferplus_orin_b{tag_budget(budget_ms)}ms*")):
        if list(d.glob("controller_*_best.pt")):
            return d
    raise FileNotFoundError(f"no orin band controller for B={budget_ms}")


def load_controller(kind: str, budget_ms: float | None, feed: dict) -> tuple[LatencyBudgetController, dict, Path]:
    d = ctrl_dir(kind, budget_ms)
    pts = sorted(d.glob("controller_*_best.pt"))
    pts = sorted(pts, key=lambda p: (0 if "retry" in p.name or "orin" in p.name else 1, p.name))
    pt = pts[0]
    sp = d / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    else:
        summary = {
            "device": DEVICE,
            "budget_ms": budget_ms,
            "lambda_lat": LAMBDA,
            "lat_meta": feed,
            "test": None,
        }
    _fix_posix_path()
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    ctrl = LatencyBudgetController()
    state = ckpt["controller"] if isinstance(ckpt, dict) and "controller" in ckpt else ckpt
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    ctrl.load_state_dict(state, strict=True)
    ctrl.eval()
    return ctrl, summary, pt


def select_bins(n: int) -> list[Path]:
    all_bins = sorted(BIN_DIR.glob("*.bin"))
    fp = [p for p in all_bins if p.name.startswith("ferplus__")]
    picked = fp[:n]
    if len(picked) < n:
        ew = [p for p in all_bins if p.name.startswith("expw__")]
        picked.extend(ew[: n - len(picked)])
    return picked


def parse_gt(name: str) -> tuple[str, int] | None:
    parts = name.split("__")
    if len(parts) < 2:
        return None
    m = re.match(r"(\d+)_", parts[1])
    if not m:
        return None
    return parts[0], int(m.group(1))


def mat_to_logits(out: ncnn.Mat) -> torch.Tensor:
    z = np.array([out[j] for j in range(out.w)], dtype=np.float32)
    return torch.from_numpy(z).unsqueeze(0)


IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_NORM = [1.0 / (0.229 * 255.0), 1.0 / (0.224 * 255.0), 1.0 / (0.225 * 255.0)]


def extract_all(net: ncnn.Net, chw: np.ndarray) -> list[torch.Tensor]:
    rgb = np.ascontiguousarray((np.clip(chw.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8))
    h, w = rgb.shape[:2]
    mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, w, h)
    mat.substract_mean_normalize(IMAGENET_MEAN, IMAGENET_NORM)
    ex = net.create_extractor()
    assert ex.input("input", mat) == 0
    outs = []
    for name in ("out0", "out1", "out2", "out3"):
        ret, out = ex.extract(name)
        assert ret == 0, name
        outs.append(mat_to_logits(out))
    return outs


def decide_c3(ctrl, e2, e3, e4, cum_ms, seg_ms, budget_ms: float) -> int:
    feats = ctrl.build_features((e2, e3, e4), cum_ms=cum_ms, seg_ms=seg_ms, budget_ms=budget_ms)
    for i, g in enumerate(ctrl.forward_gates(feats)):
        if int(g.argmax(dim=-1).item()) == 0:
            return i
    return 3


def decide_tau(e2, e3, e4, tau: float) -> int:
    for i, logits in enumerate((e2, e3, e4)):
        if float(torch.softmax(logits, dim=-1).max()) >= tau:
            return i
    return 3


def share(xs: list[int]) -> list[float]:
    n = max(1, len(xs))
    return [round(sum(1 for x in xs if x == k) / n, 6) for k in range(4)]


def load_ncnn(threads: int) -> ncnn.Net:
    net = ncnn.Net()
    net.opt.num_threads = threads
    for k in ("use_fp16_packed", "use_fp16_storage", "use_fp16_arithmetic", "use_bf16_storage"):
        setattr(net.opt, k, False)
    assert net.load_param(str(NCNN_DIR / "m.ncnn.param")) == 0
    assert net.load_model(str(NCNN_DIR / "m.ncnn.bin")) == 0
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    feed = load_feed()
    bands = band_keys(feed)
    cum = torch.tensor(feed["cum_exit_ms"], dtype=torch.float32)
    seg = torch.tensor(feed["seg_ms"], dtype=torch.float32)
    print("Orin feed cum=", [round(float(x), 4) for x in feed["cum_exit_ms"]], "bands=", bands)

    bins = select_bins(args.n)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    net = load_ncnn(args.threads)

    cache: list[tuple[Path, list[torch.Tensor], tuple[str, int] | None]] = []
    with torch.no_grad():
        for p in bins:
            chw = np.fromfile(p, dtype=np.float32).reshape(3, 112, 112)
            cache.append((p, extract_all(net, chw), parse_gt(p.name)))

    rows: list[dict] = []

    def eval_mode(name: str, exits: list[int], meta: dict) -> None:
        exit_path = OUT_ROOT / f"exits_{name}_n{len(bins)}.txt"
        exit_path.write_text("\n".join(str(x) for x in exits) + "\n", encoding="utf-8")
        correct = total = 0
        agree_final = 0
        for (_p, logits, gt), ex in zip(cache, exits):
            pred = int(logits[ex].argmax(dim=-1).item())
            final_pred = int(logits[3].argmax(dim=-1).item())
            if pred == final_pred:
                agree_final += 1
            if gt and gt[0] == "ferplus":
                total += 1
                if pred == gt[1]:
                    correct += 1
        lut = float(np.mean([cum[e].item() for e in exits]))
        # p50 of per-sample LUT latencies
        lats = sorted(float(cum[e].item()) for e in exits)
        p50 = lats[len(lats) // 2]
        row = {
            "name": name,
            "device": DEVICE,
            "n": len(bins),
            "n_ferplus": total,
            "exit_share": share(exits),
            "acc_ferplus": (correct / total) if total else None,
            "agree_final": agree_final / max(1, len(exits)),
            "p50_ms": round(p50, 4),
            "lut_lat_ms": round(lut, 4),
            "latency_source": "orin_trt_segment_lut_§8.29",
            "exits_file": str(exit_path.relative_to(ROOT)).replace("\\", "/"),
            **meta,
        }
        rows.append(row)
        print(
            f"  {name}: p50={row['p50_ms']} lut={row['lut_lat_ms']} "
            f"exits={row['exit_share']} acc_fp={row['acc_ferplus']}",
            flush=True,
        )

    eval_mode("force_final", [3] * len(cache), {"kind": "force_final"})
    for tau in TAUS:
        exits = [
            decide_tau(logits[0], logits[1], logits[2], tau)
            for _p, logits, _gt in cache
        ]
        eval_mode(f"tau_{tau:g}", exits, {"kind": "tau", "tau": tau})

    for key in ("tight", "mid", "loose"):
        b = bands[key]
        ctrl, summary, pt = load_controller("band", b, feed)
        lat = summary.get("lat_meta") or feed
        cum_t = torch.tensor(lat["cum_exit_ms"], dtype=torch.float32)
        seg_t = torch.tensor(lat["seg_ms"], dtype=torch.float32)
        exits = []
        with torch.no_grad():
            for _p, logits, _gt in cache:
                exits.append(decide_c3(ctrl, logits[0], logits[1], logits[2], cum_t, seg_t, b))
        eval_mode(
            f"c3_band_l0p2_b{tag_budget(b)}ms",
            exits,
            {
                "kind": "c3_band",
                "lambda_lat": LAMBDA,
                "budget_ms": b,
                "band_key": key,
                "controller": str(pt.relative_to(ROOT)).replace("\\", "/"),
                "autodl_test": summary.get("test"),
            },
        )

    bmid = bands["mid"]
    ctrl, summary, pt = load_controller("randB", None, feed)
    lat = summary.get("lat_meta") or feed
    cum_t = torch.tensor(lat["cum_exit_ms"], dtype=torch.float32)
    seg_t = torch.tensor(lat["seg_ms"], dtype=torch.float32)
    exits = []
    with torch.no_grad():
        for _p, logits, _gt in cache:
            exits.append(decide_c3(ctrl, logits[0], logits[1], logits[2], cum_t, seg_t, bmid))
    eval_mode(
        f"c3_randB_l0p2_b{tag_budget(bmid)}ms",
        exits,
        {
            "kind": "c3_randB",
            "lambda_lat": LAMBDA,
            "budget_ms": bmid,
            "controller": str(pt.relative_to(ROOT)).replace("\\", "/"),
            "autodl_test": summary.get("test"),
        },
    )

    ff = next(r for r in rows if r["kind"] == "force_final")
    for r in rows:
        r["speedup_vs_ff"] = round(ff["p50_ms"] / r["p50_ms"], 4) if r["p50_ms"] else None

    summary = {
        "n": args.n,
        "threads": args.threads,
        "protocol": "§8.21 / Orin §8.29 LUT",
        "controller_feed": feed,
        "bands": bands,
        "table": rows,
        "note": "p50_ms = median per-sample TRT truncated-engine cum from §8.29; host gate only",
    }
    (OUT_ROOT / "c3_formal_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    csv_path = OUT_ROOT / "c3_formal_table.csv"
    fields = [
        "device",
        "name",
        "kind",
        "p50_ms",
        "lut_lat_ms",
        "speedup_vs_ff",
        "acc_ferplus",
        "n_ferplus",
        "agree_final",
        "exit_share",
        "budget_ms",
        "latency_source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["exit_share"] = json.dumps(r["exit_share"])
            w.writerow(rr)
    print(f"\n-> {OUT_ROOT / 'c3_formal_summary.json'}")
    print(f"-> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
