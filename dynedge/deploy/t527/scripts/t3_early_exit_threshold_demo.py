#!/usr/bin/env python3
"""T3 threshold early-exit demo (no G5 learned controller).

Bake multi-exit StaticSubnet → pnnx → confidence-threshold gating on host/board.
Outputs exit hit rates + wall latency P50/P99. Label clearly as threshold ablation,
NOT the full C3 Gumbel+LUT controller.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import ncnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.elastic_mbv3 import (  # noqa: E402
    ARCH_SPECS,
    ElasticFERSuperNet,
    ExitHead,
    SubnetConfig,
    make_divisible,
)
from models.slimmable_bn import set_model_width, width_key  # noqa: E402
from tools.export_subnet_onnx import remap_v1_stem_keys  # noqa: E402

_bake_spec = importlib.util.spec_from_file_location(
    "t1b_bake", ROOT / "deploy/t527/scripts/t1b_bake_static_ncnn.py"
)
_bake = importlib.util.module_from_spec(_bake_spec)
assert _bake_spec.loader is not None
_bake_spec.loader.exec_module(_bake)

CKPT = (
    ROOT
    / "results/from_autodl/subnets_v4_predistill"
    / "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_predistill_best.pt"
)
OUT = ROOT / "results/t527_v4/t3_early_exit"
BIN_DIR = ROOT / "results/t527_v4/eval500_bins/hw112"


class BakedMultiExit(nn.Module):
    """Bake slimmable subnet with exit2/exit3/exit4 + final (4 logits)."""

    def __init__(self, slim: ElasticFERSuperNet, cfg: SubnetConfig):
        super().__init__()
        set_model_width(slim, cfg.width)
        w = float(cfg.width)
        spec = ARCH_SPECS[slim.arch]
        stem_c = make_divisible(slim.stem_ch * w)
        layers: list[nn.Module] = [_bake.bake_conv_bn_act(slim.stem[0], 3)]
        if spec["stem_extra"]:
            layers.append(_bake.bake_conv_bn_act(slim.stem[1], stem_c))
        self.stem = nn.Sequential(*layers)

        stages = nn.ModuleList()
        in_ch = stem_c
        stage_outs: list[int] = []
        for si, (out_ch_max, stride, expand) in enumerate(slim.STAGE_SPECS):
            out_ch = make_divisible(out_ch_max * w)
            blocks = nn.ModuleList()
            for bi in range(cfg.depths[si]):
                s = stride if bi == 0 else 1
                inp = in_ch if bi == 0 else out_ch
                hid = make_divisible(inp * expand)
                blocks.append(_bake.bake_ir_se(slim.stages[si][bi], inp, out_ch, hid))
            stages.append(blocks)
            in_ch = out_ch
            stage_outs.append(out_ch)
        self.stages = stages
        self.depths = cfg.depths

        # Exit heads after stages 1/2/3 (0-indexed) — copy from SlimmableExitHead
        self.exit2 = self._bake_exit(slim.exit2, stage_outs[1], w, slim.num_classes)
        self.exit3 = self._bake_exit(slim.exit3, stage_outs[2], w, slim.num_classes)
        self.exit4 = self._bake_exit(slim.exit4, stage_outs[3], w, slim.num_classes)

        mid = make_divisible(slim.head_mid * w)
        self.final_pw = _bake.bake_conv_bn_act(slim.final_pw, stage_outs[3])
        self.fc = nn.Linear(mid, slim.num_classes)
        self.fc.weight.data.copy_(slim.final_fc.weight.data[:, :mid].contiguous())
        self.fc.bias.data.copy_(slim.final_fc.bias.data.contiguous())

    @staticmethod
    def _bake_exit(src, in_ch: int, width: float, num_classes: int) -> ExitHead:
        # SlimmableExitHead → plain ExitHead with sliced weights
        mid = src.mid_ch[float(width)]
        dst = ExitHead(in_ch, num_classes)
        # ExitHead: pool, flatten, Linear(in,mid), Hardswish, Linear(mid,C)
        # But models.ExitHead uses mid=max(64,in//2) and two Linears
        # Rebuild to match sliced dims
        dst.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, mid),
            nn.Hardswish(inplace=True),
            nn.Linear(mid, num_classes),
        )
        dst.net[2].weight.data.copy_(src.fc1.weight.data[:mid, :in_ch].contiguous())
        dst.net[2].bias.data.copy_(src.fc1.bias.data[:mid].contiguous())
        dst.net[4].weight.data.copy_(src.fc2.weight.data[:, :mid].contiguous())
        dst.net[4].bias.data.copy_(src.fc2.bias.data.contiguous())
        return dst

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        for bi in range(self.depths[0]):
            x = self.stages[0][bi](x)
        for bi in range(self.depths[1]):
            x = self.stages[1][bi](x)
        e2 = self.exit2(x)
        for bi in range(self.depths[2]):
            x = self.stages[2][bi](x)
        e3 = self.exit3(x)
        for bi in range(self.depths[3]):
            x = self.stages[3][bi](x)
        e4 = self.exit4(x)
        y = self.final_pw(x)
        y = F.adaptive_avg_pool2d(y, 1).flatten(1)
        e_final = self.fc(y)
        return e2, e3, e4, e_final


def conf_max(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1).max(dim=-1).values


def run_threshold_sweep(model: nn.Module, imgs: list[np.ndarray], thresholds: list[float]) -> dict:
    model.eval()
    rows = []
    with torch.no_grad():
        packed = []
        for chw in imgs:
            e2, e3, e4, ef = model(torch.from_numpy(chw[None]))
            packed.append(
                (
                    conf_max(e2).item(),
                    conf_max(e3).item(),
                    conf_max(e4).item(),
                    int(ef.argmax(-1)),
                    int(e2.argmax(-1)),
                    int(e3.argmax(-1)),
                    int(e4.argmax(-1)),
                )
            )
    for tau in thresholds:
        hits = {2: 0, 3: 0, 4: 0, "final": 0}
        match_final = 0
        for c2, c3, c4, pf, p2, p3, p4 in packed:
            if c2 >= tau:
                hits[2] += 1
                pred = p2
            elif c3 >= tau:
                hits[3] += 1
                pred = p3
            elif c4 >= tau:
                hits[4] += 1
                pred = p4
            else:
                hits["final"] += 1
                pred = pf
            match_final += int(pred == pf)
        n = len(packed)
        rows.append(
            {
                "tau": tau,
                "exit2_rate": hits[2] / n,
                "exit3_rate": hits[3] / n,
                "exit4_rate": hits[4] / n,
                "final_rate": hits["final"] / n,
                "agree_final": match_final / n,
            }
        )
    return {"n": len(packed), "sweep": rows}


def export_pnnx(model: nn.Module, out: Path, img: int = 112) -> tuple[Path, Path]:
    import pnnx

    out.mkdir(parents=True, exist_ok=True)
    pt = str(out / "m.pt")
    param = out / "m.ncnn.param"
    binp = out / "m.ncnn.bin"
    xt = torch.randn(1, 3, img, img)
    try:
        pnnx.export(
            model,
            pt,
            inputs=xt,
            ncnnparam=str(param),
            ncnnbin=str(binp),
            fp16=False,
        )
    except Exception as e:
        print(f"pnnx post-err: {type(e).__name__}: {e}")
    if not param.exists():
        raise RuntimeError("pnnx failed")
    # Rename first input to input; keep multiple outputs as out0..out3
    text = param.read_text(encoding="utf-8")
    text = text.replace(" in0", " input")
    lines = []
    for line in text.splitlines():
        if line.startswith("Input"):
            line = "Input                    input                     0 1 input"
        lines.append(line)
    param.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return param, binp


def bench_ncnn_threshold(
    param: Path,
    binp: Path,
    rgbs: list[np.ndarray],
    tau: float,
    out_names: list[str],
    threads: int = 4,
    force_final: bool = False,
) -> dict:
    net = ncnn.Net()
    net.opt.num_threads = threads
    for k in (
        "use_fp16_packed",
        "use_fp16_storage",
        "use_fp16_arithmetic",
        "use_bf16_storage",
    ):
        setattr(net.opt, k, False)
    assert net.load_param(str(param)) == 0 and net.load_model(str(binp)) == 0
    hits = {0: 0, 1: 0, 2: 0, 3: 0}
    times = []
    for rgb in rgbs:
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        mat = ncnn.Mat.from_pixels(
            np.ascontiguousarray(rgb), ncnn.Mat.PixelType.PIXEL_RGB, w, h
        )
        mat.substract_mean_normalize(
            [123.675, 116.28, 103.53],
            [1.0 / (0.229 * 255.0), 1.0 / (0.224 * 255.0), 1.0 / (0.225 * 255.0)],
        )
        ex = net.create_extractor()
        assert ex.input("input", mat) == 0
        exited = False
        if not force_final:
            for i, name in enumerate(out_names[:-1]):
                ret, out = ex.extract(name)
                if ret != 0:
                    continue
                z = np.array([out[j] for j in range(out.w)], dtype=np.float32)
                e = np.exp(z - z.max())
                p = e / e.sum()
                if float(p.max()) >= tau:
                    hits[i] += 1
                    exited = True
                    break
        if not exited:
            ret, out = ex.extract(out_names[-1])
            assert ret == 0
            hits[3] += 1
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    n = len(rgbs)
    return {
        "tau": tau,
        "threads": threads,
        "force_final": int(force_final),
        "n": n,
        "exit_rates": {f"e{i}": hits[i] / n for i in range(4)},
        "lat_ms_p50": float(np.percentile(arr, 50)),
        "lat_ms_p99": float(np.percentile(arr, 99)),
        "lat_ms_mean": float(arr.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tau", type=float, default=0.85)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = remap_v1_stem_keys(ckpt["model"])
    slim = ElasticFERSuperNet(num_classes=8, arch="v3_wide", slimmable=True)
    slim.load_state_dict(state, strict=True)
    slim.eval()
    cfg = SubnetConfig(width=0.5, depths=(3, 3, 3, 3), img_size=112, gray=False)
    model = BakedMultiExit(slim, cfg).eval()

    bins = sorted(BIN_DIR.glob("*.bin"))[: args.n]
    chws = [np.fromfile(p, dtype=np.float32).reshape(3, 112, 112) for p in bins]
    rgbs = [
        np.ascontiguousarray((np.clip(c.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8))
        for c in chws
    ]

    sweep = run_threshold_sweep(model, chws, [0.5, 0.7, 0.85, 0.9, 0.95, 0.99])
    (OUT / "torch_threshold_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    print("torch sweep", json.dumps(sweep, indent=2))

    print("exporting multi-exit pnnx...")
    param, binp = export_pnnx(model, OUT / "ncnn")
    text = param.read_text(encoding="utf-8")
    # Prefer explicit multi-exit names from pnnx
    out_names = []
    for cand in ("out0", "out1", "out2", "out3"):
        if re.search(rf"\b{cand}\b", text):
            out_names.append(cand)
    if len(out_names) < 4:
        outs = []
        for line in text.splitlines():
            if line.startswith("InnerProduct"):
                parts = line.split()
                try:
                    nb, nt = int(parts[2]), int(parts[3])
                    tops = parts[4 + nb : 4 + nb + nt]
                    # keep only 8-class classifiers (0=8)
                    if "0=8" in line:
                        outs.extend(tops)
                except Exception:
                    pass
        uniq = []
        for o in outs:
            if o not in uniq:
                uniq.append(o)
        out_names = uniq[-4:] if len(uniq) >= 4 else uniq
    print("ncnn outputs", out_names)
    if len(out_names) < 4:
        print("WARN: expected 4 outputs; pnnx may have fused exits — torch sweep still valid")
        meta = {
            "status": "partial",
            "note": "threshold ablation host-side; ncnn multi-out incomplete",
            "torch_sweep": sweep,
            "out_names": out_names,
        }
        (OUT / "t3_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return 0

    threads = 4  # formal matrix protocol
    base = bench_ncnn_threshold(
        param, binp, rgbs, args.tau, out_names, threads=threads, force_final=True
    )
    print("ncnn force_final @4t", base)
    bench = bench_ncnn_threshold(
        param, binp, rgbs, args.tau, out_names, threads=threads, force_final=False
    )
    print("ncnn early_exit @4t", bench)
    speedup = (
        base["lat_ms_p50"] / bench["lat_ms_p50"] if bench["lat_ms_p50"] > 0 else float("nan")
    )
    meta = {
        "status": "ok_threshold_demo",
        "note": "NOT G5 learned controller; fixed-confidence threshold ablation (paper 4.4)",
        "model": CKPT.name,
        "threads": threads,
        "torch_sweep": sweep,
        "ncnn_force_final": base,
        "ncnn_bench": bench,
        "speedup_p50_vs_force_final": round(speedup, 4),
        "out_names": out_names,
    }
    (OUT / "t3_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE speedup_p50={speedup:.3f}x vs force_final -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
