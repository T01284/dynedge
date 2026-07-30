#!/usr/bin/env python3
"""§8.21 C3 formal board对照 — H618 + T527.

Host: multi-exit NCNN logits + v4_band_retry / τ / force_final → exits + FER+ GT acc.
Board: fer_early_exit wall-clock (P50/P99, exit share) via --exits / --tau / --force-final.

Default rows (per device):
  force_final
  tau∈{0.7,0.85,0.9}  (online on board)
  c3_band λ=0.2 @ tight / mid / loose budgets
  c3_randB λ=0.2 @ mid budget (features)

Usage:
  python deploy/t527/scripts/run_c3_formal_board.py
  python deploy/t527/scripts/run_c3_formal_board.py --devices h618 --n 80
  python deploy/t527/scripts/run_c3_formal_board.py --host-only   # exits+acc, no SSH
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import tarfile
import tempfile
from pathlib import Path

import ncnn
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy/t527/scripts"))
sys.path.insert(0, str(ROOT / "deploy/h618/scripts"))

from models.latency_controller import LatencyBudgetController  # noqa: E402

NCNN_DIR = ROOT / "results/t527_v4/t3_early_exit/ncnn"
BIN_DIR = ROOT / "results/t527_v4/eval500_bins/hw112"
CTRL_BAND = ROOT / "results/from_autodl/formal_v4/controller_band"
EE_BIN = ROOT / "deploy/ncnn_edge/build-linux-aarch64-acc/fer_early_exit"
OUT_ROOT = ROOT / "results/c3_formal_board"

# In-band budgets from controller_feed (SUMMARY §8.25.1 / §8.18–8.19)
BAND = {
    "h618": {"tight": 16.9, "mid": 22.42, "loose": 27.93},
    "t527": {"tight": 11.93, "mid": 15.99, "loose": 20.05},
}
LAMBDA = 0.2
TAUS = (0.7, 0.85, 0.9)


def _fix_posix_path() -> None:
    try:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]
    except Exception:
        pass


def tag_budget(ms: float) -> str:
    # directory names use compact forms: 16.9 / 22.42 / 11.93 / 15.99
    s = f"{ms:g}"
    return s


def ctrl_dir(device: str, kind: str, budget_ms: float | None) -> Path:
    if kind == "randB":
        return CTRL_BAND / f"ferplus_{device}_randB_l0p2"
    assert budget_ms is not None
    return CTRL_BAND / f"ferplus_{device}_b{tag_budget(budget_ms)}ms_l0p2"


def load_controller(device: str, kind: str, budget_ms: float | None) -> tuple[LatencyBudgetController, dict, Path]:
    d = ctrl_dir(device, kind, budget_ms)
    pts = sorted(d.glob("controller_*_best.pt"))
    if not pts:
        raise FileNotFoundError(d)
    # prefer *retry* if present
    pts = sorted(pts, key=lambda p: (0 if "retry" in p.name else 1, p.name))
    pt = pts[0]
    sp = d / "summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    else:
        # some band points only shipped the .pt; lat_meta from board feed
        feed = json.loads(
            (ROOT / f"results/{device}_v4/exit_segment_latency/summary.json").read_text(
                encoding="utf-8"
            )
        )["controller_feed"]
        summary = {
            "device": device,
            "budget_ms": budget_ms,
            "lambda_lat": LAMBDA,
            "lat_meta": feed,
            "test": None,
            "note": "summary.json missing; lat_meta from exit_segment_latency",
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


def select_bins(n: int, prefer_ferplus: bool) -> list[Path]:
    all_bins = sorted(BIN_DIR.glob("*.bin"))
    if not prefer_ferplus:
        return all_bins[:n]
    fp = [p for p in all_bins if p.name.startswith("ferplus__")]
    ew = [p for p in all_bins if p.name.startswith("expw__")]
    # Prefer all FER+ then fill with ExpW to reach n (wall-clock n; acc only on FER+)
    picked = fp[:n]
    if len(picked) < n:
        picked.extend(ew[: n - len(picked)])
    return picked


def parse_gt(name: str) -> tuple[str, int] | None:
    # ferplus__00_neutral__xxx.bin
    parts = name.split("__")
    if len(parts) < 2:
        return None
    ds = parts[0]
    m = re.match(r"(\d+)_", parts[1])
    if not m:
        return None
    return ds, int(m.group(1))


def mat_to_logits(out: ncnn.Mat) -> torch.Tensor:
    z = np.array([out[j] for j in range(out.w)], dtype=np.float32)
    return torch.from_numpy(z).unsqueeze(0)


# ImageNet mean/norm for uint8 pixels (must match train/datasets.common Normalize).
# Prior bug: mean=0,norm=1/255 → host n=155 FF acc 0.787; with ImageNet → 0.955.
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


def host_decide_device(
    device: str,
    bins: list[Path],
    threads: int,
) -> dict:
    out = OUT_ROOT / device
    out.mkdir(parents=True, exist_ok=True)
    net = load_ncnn(threads)

    # preload all logits once
    cache: list[tuple[Path, list[torch.Tensor], tuple[str, int] | None]] = []
    with torch.no_grad():
        for p in bins:
            chw = np.fromfile(p, dtype=np.float32).reshape(3, 112, 112)
            logits = extract_all(net, chw)
            cache.append((p, logits, parse_gt(p.name)))

    rows = []

    def eval_mode(name: str, exits: list[int], meta: dict) -> dict:
        exit_path = out / f"exits_{name}_n{len(bins)}.txt"
        exit_path.write_text("\n".join(str(x) for x in exits) + "\n", encoding="utf-8")
        # GT acc on ferplus only
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
        row = {
            "name": name,
            "device": device,
            "n": len(bins),
            "n_ferplus": total,
            "exit_share": share(exits),
            "acc_ferplus": (correct / total) if total else None,
            "agree_final": agree_final / max(1, len(exits)),
            "exits_file": str(exit_path.relative_to(ROOT)).replace("\\", "/"),
            **meta,
        }
        rows.append(row)
        print(
            f"  HOST {device}/{name}: exits={row['exit_share']} "
            f"acc_fp={row['acc_ferplus']} agree={row['agree_final']:.3f}",
            flush=True,
        )
        return row

    # force_final
    eval_mode("force_final", [3] * len(cache), {"kind": "force_final"})

    # tau online (host-side exits for acc; board will use --tau)
    for tau in TAUS:
        exits = []
        with torch.no_grad():
            for _p, logits, _gt in cache:
                exits.append(decide_tau(logits[0], logits[1], logits[2], tau))
        eval_mode(f"tau_{tau:g}", exits, {"kind": "tau", "tau": tau})

    # band λ=0.2
    for key in ("tight", "mid", "loose"):
        b = BAND[device][key]
        ctrl, summary, pt = load_controller(device, "band", b)
        lat = summary["lat_meta"]
        cum = torch.tensor(lat["cum_exit_ms"], dtype=torch.float32)
        seg = torch.tensor(lat["seg_ms"], dtype=torch.float32)
        exits = []
        with torch.no_grad():
            for _p, logits, _gt in cache:
                exits.append(decide_c3(ctrl, logits[0], logits[1], logits[2], cum, seg, b))
        lut_lat = float(np.mean([cum[e].item() for e in exits]))
        eval_mode(
            f"c3_band_l0p2_b{tag_budget(b)}ms",
            exits,
            {
                "kind": "c3_band",
                "lambda_lat": LAMBDA,
                "budget_ms": b,
                "band_key": key,
                "controller": str(pt.relative_to(ROOT)).replace("\\", "/"),
                "lut_lat_ms": lut_lat,
                "autodl_test": summary.get("test"),
            },
        )

    # randB λ=0.2 @ mid budget features
    bmid = BAND[device]["mid"]
    ctrl, summary, pt = load_controller(device, "randB", None)
    lat = summary.get("lat_meta") or json.loads(
        (ROOT / f"results/{device}_v4/exit_segment_latency/summary.json").read_text(encoding="utf-8")
    )["controller_feed"]
    cum = torch.tensor(lat["cum_exit_ms"], dtype=torch.float32)
    seg = torch.tensor(lat["seg_ms"], dtype=torch.float32)
    exits = []
    with torch.no_grad():
        for _p, logits, _gt in cache:
            exits.append(decide_c3(ctrl, logits[0], logits[1], logits[2], cum, seg, bmid))
    lut_lat = float(np.mean([cum[e].item() for e in exits]))
    eval_mode(
        f"c3_randB_l0p2_b{tag_budget(bmid)}ms",
        exits,
        {
            "kind": "c3_randB",
            "lambda_lat": LAMBDA,
            "budget_ms": bmid,
            "controller": str(pt.relative_to(ROOT)).replace("\\", "/"),
            "lut_lat_ms": lut_lat,
            "autodl_test": summary.get("test"),
        },
    )

    payload = {"device": device, "n": len(bins), "rows": rows}
    host_json = out / "host_summary.json"
    host_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Keep sized snapshot so n=80 formal and n=155 full-pool do not clobber each other.
    (out / f"host_summary_n{len(bins)}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return {"device": device, "rows": rows, "bins": [p.name for p in bins]}


def parse_result(text: str) -> dict:
    m = re.search(r"^RESULT .+$", text, re.M)
    if not m:
        return {"raw_tail": text[-500:]}
    row = {"raw": m.group(0)}
    for p in m.group(0).split()[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            row[k] = v
    return row


def board_run(device: str, bins: list[Path], host_rows: list[dict], threads: int) -> dict:
    if device == "h618":
        from ssh_h618 import put, run  # type: ignore

        remote = "/root/c3_formal"
    else:
        from ssh_t527 import put, run  # type: ignore

        remote = "/root/c3_formal"

    out = OUT_ROOT / device
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "bins.tar"
        with tarfile.open(tar_path, "w") as tar:
            for p in bins:
                tar.add(p, arcname=p.name)
        run(f"rm -rf {remote}; mkdir -p {remote}/bins {remote}/models", timeout=60)
        put(str(EE_BIN), f"{remote}/fer_early_exit")
        put(str(NCNN_DIR / "m.ncnn.param"), f"{remote}/models/m.ncnn.param")
        put(str(NCNN_DIR / "m.ncnn.bin"), f"{remote}/models/m.ncnn.bin")
        put(str(tar_path), f"{remote}/bins.tar")
        # push all exits files used for forced modes
        for row in host_rows:
            if row["kind"] in ("force_final", "tau"):
                continue
            local = ROOT / row["exits_file"]
            put(str(local), f"{remote}/{Path(row['exits_file']).name}")

    code, o, e = run(
        f"cd {remote} && tar xf bins.tar -C bins && "
        f"for f in bins/*; do echo {remote}/$f; done > list.txt && "
        f"chmod +x fer_early_exit && wc -l list.txt",
        timeout=180,
    )
    print(o, e, "prep", code, flush=True)
    if code != 0:
        raise RuntimeError(f"prep failed on {device}: {e}")

    common = (
        f"./fer_early_exit --param models/m.ncnn.param --bin models/m.ncnn.bin "
        f"--list list.txt --h 112 --w 112 --c 3 --threads {threads} "
        f"--out0 out0 --out1 out1 --out2 out2 --out3 out3"
    )

    board_rows = []
    # force_final + tau online on board
    jobs = [("force_final", " --force-final", "force_final")]
    for tau in TAUS:
        jobs.append((f"tau_{tau:g}", f" --tau {tau}", "tau"))
    # learned forced exits
    for row in host_rows:
        if row["kind"] in ("c3_band", "c3_randB"):
            ename = Path(row["exits_file"]).name
            jobs.append((row["name"], f" --exits {ename} --tau 0.85", row["kind"]))

    for name, extra, kind in jobs:
        print(f"  BOARD {device}/{name} ...", flush=True)
        code, o, e = run(f"cd {remote} && {common}{extra}", timeout=900)
        text = (o or "") + "\n" + (e or "")
        (out / f"board_{name}.txt").write_text(text, encoding="utf-8")
        parsed = parse_result(o or "")
        if code != 0:
            raise RuntimeError(f"board {device}/{name} rc={code}\n{text[-1500:]}")
        # merge host acc if same name
        host = next((r for r in host_rows if r["name"] == name), None)
        brow = {
            "name": name,
            "kind": kind,
            "device": device,
            "board": parsed,
            "p50_ms": float(parsed["p50_ms"]) if "p50_ms" in parsed else None,
            "p99_ms": float(parsed["p99_ms"]) if "p99_ms" in parsed else None,
            "host": host,
        }
        board_rows.append(brow)
        print(
            f"  OK {device}/{name} p50={brow['p50_ms']} p99={brow['p99_ms']}",
            flush=True,
        )

    ff = next(r["p50_ms"] for r in board_rows if r["name"] == "force_final")
    for r in board_rows:
        if r["p50_ms"] and ff:
            r["speedup_vs_ff"] = round(ff / r["p50_ms"], 4)

    summary = {
        "status": "ok",
        "device": device,
        "threads": threads,
        "n": len(bins),
        "protocol": "§8.21 C3 formal: host gate + board wall-clock",
        "rows": board_rows,
    }
    (out / "board_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="h618,t527")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--host-only", action="store_true")
    ap.add_argument("--no-prefer-ferplus", action="store_true")
    args = ap.parse_args()

    if not EE_BIN.exists():
        raise SystemExit(f"missing {EE_BIN}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    bins = select_bins(args.n, prefer_ferplus=not args.no_prefer_ferplus)
    print(f"bins={len(bins)} ferplus={sum(1 for p in bins if p.name.startswith('ferplus__'))}", flush=True)

    all_out = {}
    for device in devices:
        print(f"\n===== HOST DECIDE {device} =====", flush=True)
        host = host_decide_device(device, bins, args.threads)
        if args.host_only:
            all_out[device] = {"host": host}
            continue
        print(f"\n===== BOARD {device} =====", flush=True)
        board = board_run(device, bins, host["rows"], args.threads)
        all_out[device] = {"host": host, "board": board}

    summary_path = OUT_ROOT / "c3_formal_summary.json"
    # compact table
    table = []
    for device, blob in all_out.items():
        board = blob.get("board")
        if not board:
            continue
        for r in board["rows"]:
            h = r.get("host") or {}
            table.append(
                {
                    "device": device,
                    "name": r["name"],
                    "kind": r["kind"],
                    "p50_ms": r.get("p50_ms"),
                    "p99_ms": r.get("p99_ms"),
                    "speedup_vs_ff": r.get("speedup_vs_ff"),
                    "exit_share": (h.get("exit_share") if h else None),
                    "acc_ferplus": (h.get("acc_ferplus") if h else None),
                    "n_ferplus": (h.get("n_ferplus") if h else None),
                }
            )
    payload = {"n": args.n, "threads": args.threads, "table": table, "devices": all_out}
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}", flush=True)
    print("\n=== TABLE ===", flush=True)
    for r in table:
        print(
            f"{r['device']:5s} {r['name']:28s} p50={r['p50_ms']}  "
            f"acc={r['acc_ferplus']}  xFF={r['speedup_vs_ff']}  exits={r['exit_share']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
