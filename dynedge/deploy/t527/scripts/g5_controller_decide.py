#!/usr/bin/env python3
"""G5 host-side controller decisions on T3 multi-exit NCNN logits.

Pipeline (通管道, not paper-final budgets):
  1) Load T527 bake multi-exit ncnn (same artifact as T3)
  2) Load C3 controller ckpt + LUT cum/seg from summary.json
  3) Hard-gate per image → exits.txt (0=e2 .. 3=final)
  4) Also emit T3 τ=0.85 exits for same images (对照)

Board wall-clock: push exits.txt into fer_early_exit --exits (see run_g5_board_early_exit.py).
"""
from __future__ import annotations

import argparse
import json
import pathlib
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
CTRL_ROOT = ROOT / "results/from_autodl/formal_v4/controller"


def _fix_posix_path() -> None:
    try:
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]
    except Exception:
        pass


def load_controller(device: str, budget_ms: float) -> tuple[LatencyBudgetController, dict]:
    tag = f"ferplus_{device}_b{int(budget_ms)}ms"
    summary_path = CTRL_ROOT / tag / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pt = CTRL_ROOT / tag / f"controller_{tag}_seed42_v4_best.pt"
    if not pt.exists():
        # tolerate alternate naming from AutoDL path
        cands = list((CTRL_ROOT / tag).glob("controller_*_best.pt"))
        if not cands:
            raise FileNotFoundError(pt)
        pt = cands[0]
    _fix_posix_path()
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    ctrl = LatencyBudgetController()
    state = ckpt["controller"] if isinstance(ckpt, dict) and "controller" in ckpt else ckpt
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    ctrl.load_state_dict(state, strict=True)
    ctrl.eval()
    return ctrl, summary


def mat_to_logits(out: ncnn.Mat) -> torch.Tensor:
    z = np.array([out[j] for j in range(out.w)], dtype=np.float32)
    return torch.from_numpy(z).unsqueeze(0)


def extract_all_logits(
    net: ncnn.Net, chw: np.ndarray, out_names: list[str]
) -> list[torch.Tensor]:
    rgb = np.ascontiguousarray(
        (np.clip(chw.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    )
    h, w = rgb.shape[:2]
    mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, w, h)
    # ImageNet uint8 mean/norm (match datasets.common); was wrongly 0 + 1/255
    mat.substract_mean_normalize(
        [123.675, 116.28, 103.53],
        [1.0 / (0.229 * 255.0), 1.0 / (0.224 * 255.0), 1.0 / (0.225 * 255.0)],
    )
    ex = net.create_extractor()
    assert ex.input("input", mat) == 0
    logits = []
    for name in out_names:
        ret, out = ex.extract(name)
        assert ret == 0, name
        logits.append(mat_to_logits(out))
    return logits


def decide_g5(
    ctrl: LatencyBudgetController,
    e2: torch.Tensor,
    e3: torch.Tensor,
    e4: torch.Tensor,
    cum_ms: torch.Tensor,
    seg_ms: torch.Tensor,
    budget_ms: float,
) -> int:
    feats = ctrl.build_features((e2, e3, e4), cum_ms=cum_ms, seg_ms=seg_ms, budget_ms=budget_ms)
    gate_logits = ctrl.forward_gates(feats)
    decided = False
    chosen = 3
    for i, g in enumerate(gate_logits):
        if int(g.argmax(dim=-1).item()) == 0:
            chosen = i
            decided = True
            break
    return chosen if decided else 3


def decide_t3(e2: torch.Tensor, e3: torch.Tensor, e4: torch.Tensor, tau: float) -> int:
    for i, logits in enumerate((e2, e3, e4)):
        p = torch.softmax(logits, dim=-1)
        if float(p.max()) >= tau:
            return i
    return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("t527", "h618"), default="t527")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--budget-ms", type=float, default=5.0)
    ap.add_argument("--tau", type=float, default=0.85, help="T3 threshold for对照")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    out = ROOT / f"results/{args.device}_v4/g5_early_exit"
    out.mkdir(parents=True, exist_ok=True)
    param = NCNN_DIR / "m.ncnn.param"
    binp = NCNN_DIR / "m.ncnn.bin"
    if not param.exists() or not binp.exists():
        raise SystemExit(f"missing T3 ncnn multi-exit under {NCNN_DIR}")

    ctrl, summary = load_controller(args.device, args.budget_ms)
    lat = summary["lat_meta"]
    cum_ms = torch.tensor(lat["cum_exit_ms"], dtype=torch.float32)
    # seg_ms length 5 in summary (stem..final); controller uses seg[i+1] at gate i
    seg_ms = torch.tensor(lat["seg_ms"], dtype=torch.float32)

    net = ncnn.Net()
    net.opt.num_threads = args.threads
    for k in (
        "use_fp16_packed",
        "use_fp16_storage",
        "use_fp16_arithmetic",
        "use_bf16_storage",
    ):
        setattr(net.opt, k, False)
    assert net.load_param(str(param)) == 0 and net.load_model(str(binp)) == 0
    out_names = ["out0", "out1", "out2", "out3"]

    bins = sorted(BIN_DIR.glob("*.bin"))[: args.n]
    g5_exits: list[int] = []
    t3_exits: list[int] = []
    with torch.no_grad():
        for p in bins:
            chw = np.fromfile(p, dtype=np.float32).reshape(3, 112, 112)
            e2, e3, e4, ef = extract_all_logits(net, chw, out_names)
            _ = ef  # final unused for gating
            g5_exits.append(
                decide_g5(ctrl, e2, e3, e4, cum_ms, seg_ms, args.budget_ms)
            )
            t3_exits.append(decide_t3(e2, e3, e4, args.tau))

    def share(xs: list[int]) -> list[float]:
        n = max(1, len(xs))
        return [sum(1 for x in xs if x == k) / n for k in range(4)]

    g5_path = out / f"exits_g5_{args.device}_b{int(args.budget_ms)}ms_n{args.n}.txt"
    t3_path = out / f"exits_t3_tau{args.tau:.2f}_n{args.n}.txt"
    g5_path.write_text("\n".join(str(x) for x in g5_exits) + "\n", encoding="utf-8")
    t3_path.write_text("\n".join(str(x) for x in t3_exits) + "\n", encoding="utf-8")

    # LUT expected latency under G5 decisions
    lut_lat = float(np.mean([cum_ms[e].item() for e in g5_exits]))
    meta = {
        "status": "host_decided",
        "note": "G5 hard-gate on T3 multi-exit NCNN logits; board uses --exits for wall clock",
        "device": args.device,
        "n": len(bins),
        "budget_ms": args.budget_ms,
        "controller": summary.get("best_path"),
        "lut_cum_exit_ms": lat["cum_exit_ms"],
        "g5_exit_share": share(g5_exits),
        "g5_lut_mean_ms": lut_lat,
        "t3_tau": args.tau,
        "t3_exit_share": share(t3_exits),
        "autodl_test": summary.get("test"),
        "exits_g5": str(g5_path.relative_to(ROOT)).replace("\\", "/"),
        "exits_t3": str(t3_path.relative_to(ROOT)).replace("\\", "/"),
        "bins": [p.name for p in bins],
    }
    meta_path = out / f"host_decide_b{int(args.budget_ms)}ms.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
