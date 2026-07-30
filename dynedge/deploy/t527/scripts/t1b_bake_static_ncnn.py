#!/usr/bin/env python3
"""T1b: bake slimmable v4 → plain static net → ONNX → onnx2ncnn (no optimize) → cos check.

Root cause so far: onnx2ncnn of slimmable/Unsqueeze graph writes NaN axes; pnnx of
SlimmableBatchNorm ModuleDict also drifts. Baking to ConvBNAct/InvertedResidualSE
removes dynamic slice + per-width BN dict before conversion.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import ncnn
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxsim import simplify

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.elastic_mbv3 import (  # noqa: E402
    ARCH_SPECS,
    ConvBNAct,
    ElasticFERSuperNet,
    InvertedResidualSE,
    SqueezeExcite,
    SubnetConfig,
    make_divisible,
)
from models.slimmable_bn import set_model_width, width_key  # noqa: E402
from tools.export_subnet_onnx import StaticSubnet, remap_v1_stem_keys  # noqa: E402

WSL_TOOLS = "/home/wang/opt/ncnn-x86-tools/bin"
DEFAULT_CKPT = (
    ROOT
    / "results/from_autodl/subnets_v4_predistill"
    / "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_best.pt"
)
DEFAULT_ONNX_GOLD = (
    ROOT
    / "results/h618_onnx_v4"
    / "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64.onnx"
)
OUT = ROOT / "results/t527_v4/t1b_baked"


def wsl_path(p: Path) -> str:
    s = str(p.resolve())
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:].replace(chr(92), '/')}"
    return s


def run_wsl(args: list[str]) -> None:
    if sys.platform.startswith("linux"):
        r = subprocess.run(args, capture_output=True, text=True)
    else:
        cmd = ["wsl", "-e", "bash", "-lc", " ".join(shlex.quote(a) for a in args)]
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout or "")[-2000:] + (r.stderr or "")[-2000:])


def copy_bn(dst: nn.BatchNorm2d, src: nn.BatchNorm2d) -> None:
    dst.weight.data.copy_(src.weight.data)
    dst.bias.data.copy_(src.bias.data)
    dst.running_mean.copy_(src.running_mean)
    dst.running_var.copy_(src.running_var)
    dst.eps = src.eps


def bake_conv_bn_act(src, in_ch: int) -> ConvBNAct:
    out_ch = int(src.bn.channels_for())
    k = int(src.conv.kernel_size[0])
    s = int(src.conv.stride[0])
    has_act = not isinstance(src.act, nn.Identity)
    dst = ConvBNAct(in_ch, out_ch, k=k, s=s, act=has_act)
    dst.conv.weight.data.copy_(src.conv.weight.data[:out_ch, :in_ch].contiguous())
    bn = src.bn.bns[width_key(src.bn.active_width)]
    copy_bn(dst.bn, bn)
    return dst


def bake_se(src, ch: int) -> SqueezeExcite:
    mid = max(8, ch // src.reduction)
    dst = SqueezeExcite(ch, reduction=src.reduction)
    # SqueezeExcite builds mid = max(8, ch // reduction) — must match
    assert dst.net[2].in_features == ch and dst.net[2].out_features == mid
    dst.net[2].weight.data.copy_(src.fc1.weight.data[:mid, :ch].contiguous())
    dst.net[2].bias.data.copy_(src.fc1.bias.data[:mid].contiguous())
    dst.net[4].weight.data.copy_(src.fc2.weight.data[:ch, :mid].contiguous())
    dst.net[4].bias.data.copy_(src.fc2.bias.data[:ch].contiguous())
    return dst


def bake_ir_se(src, in_ch: int, out_ch: int, hid_ch: int) -> InvertedResidualSE:
    dst = InvertedResidualSE(in_ch, out_ch, src.stride, src.expand_ratio, use_se=True)
    # expand (optional 1x1) + dw + bn + act live in expand_dw
    if src.expand is not None:
        assert isinstance(dst.expand_dw[0], ConvBNAct)
        baked_exp = bake_conv_bn_act(src.expand, in_ch)
        dst.expand_dw[0].conv.weight.data.copy_(baked_exp.conv.weight.data)
        copy_bn(dst.expand_dw[0].bn, baked_exp.bn)
        dw_idx = 1
    else:
        dw_idx = 0
    # dw conv / bn / act
    dw = dst.expand_dw[dw_idx]
    assert isinstance(dw, nn.Conv2d)
    dw.weight.data.copy_(src.dw.weight.data[:hid_ch].contiguous())
    copy_bn(dst.expand_dw[dw_idx + 1], src.dw_bn.bns[width_key(src.dw_bn.active_width)])
    # SE
    assert not isinstance(src.se, nn.Identity)
    dst.se = bake_se(src.se, hid_ch)
    # project
    dst.project[0].weight.data.copy_(src.project.weight.data[:out_ch, :hid_ch].contiguous())
    copy_bn(dst.project[1], src.project_bn.bns[width_key(src.project_bn.active_width)])
    return dst


class BakedFER(nn.Module):
    def __init__(self, slim: ElasticFERSuperNet, cfg: SubnetConfig):
        super().__init__()
        set_model_width(slim, cfg.width)
        w = float(cfg.width)
        spec = ARCH_SPECS[slim.arch]
        stem_c = make_divisible(slim.stem_ch * w)
        layers: list[nn.Module] = [bake_conv_bn_act(slim.stem[0], 3)]
        if spec["stem_extra"]:
            layers.append(bake_conv_bn_act(slim.stem[1], stem_c))
        self.stem = nn.Sequential(*layers)

        stages = nn.ModuleList()
        in_ch = stem_c
        for si, (out_ch_max, stride, expand) in enumerate(slim.STAGE_SPECS):
            out_ch = make_divisible(out_ch_max * w)
            blocks = nn.ModuleList()
            for bi in range(cfg.depths[si]):
                s = stride if bi == 0 else 1
                inp = in_ch if bi == 0 else out_ch
                hid = make_divisible(inp * expand)
                src = slim.stages[si][bi]
                blocks.append(bake_ir_se(src, inp, out_ch, hid))
            stages.append(blocks)
            in_ch = out_ch
        self.stages = stages
        self.depths = cfg.depths

        last = in_ch
        mid = make_divisible(slim.head_mid * w)
        self.final_pw = bake_conv_bn_act(slim.final_pw, last)
        self.fc = nn.Linear(mid, slim.num_classes)
        self.fc.weight.data.copy_(slim.final_fc.weight.data[:, :mid].contiguous())
        self.fc.bias.data.copy_(slim.final_fc.bias.data.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for si, blocks in enumerate(self.stages):
            for bi in range(self.depths[si]):
                x = blocks[bi](x)
        x = self.final_pw(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def patch_nan_axes(bin_path: Path) -> int:
    a = np.fromfile(bin_path, dtype=np.float32).copy()
    n = 0
    for i in range(min(256, a.size)):
        if np.isnan(a[i]):
            a[i] = -1.0
            n += 1
    if n:
        a.tofile(bin_path)
    return n


def ncnn_logits(param: Path, binp: Path, rgb: np.ndarray, in_name: str = "input") -> np.ndarray:
    net = ncnn.Net()
    net.opt.num_threads = 1
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.opt.use_bf16_storage = False
    assert net.load_param(str(param)) == 0
    assert net.load_model(str(binp)) == 0
    h, w = rgb.shape[:2]
    mat = ncnn.Mat.from_pixels(
        np.ascontiguousarray(rgb), ncnn.Mat.PixelType.PIXEL_RGB, w, h
    )
    mat.substract_mean_normalize([0, 0, 0], [1 / 255.0] * 3)
    ex = net.create_extractor()
    text = param.read_text(encoding="utf-8")
    if in_name not in text:
        for line in text.splitlines():
            if line.startswith("Input"):
                in_name = line.split()[3] if len(line.split()) > 3 else line.split()[-1]
                break
    assert ex.input(in_name, mat) == 0
    for cand in ("output", "out0", "logits"):
        ret, out = ex.extract(cand)
        if ret == 0:
            return np.array([out[i] for i in range(out.w)], dtype=np.float32)
    # last blob name from param
    out_name = None
    for line in reversed(text.splitlines()):
        parts = line.split()
        if len(parts) >= 4 and parts[0] not in {"7767517", "Input"}:
            try:
                nb, nt = int(parts[2]), int(parts[3])
                tops = parts[4 + nb : 4 + nb + nt]
                if tops:
                    out_name = tops[-1]
                    break
            except Exception:
                continue
    assert out_name is not None
    ret, out = ex.extract(out_name)
    assert ret == 0
    return np.array([out[i] for i in range(out.w)], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--gold-onnx", type=Path, default=DEFAULT_ONNX_GOLD)
    ap.add_argument("--width", type=float, default=0.35)
    ap.add_argument("--depths", type=str, default="2,2,2,2")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--gray", action="store_true", default=True)
    ap.add_argument("--skip-optimize", action="store_true", default=True)
    args = ap.parse_args()

    depths = tuple(int(x) for x in args.depths.split(","))
    assert len(depths) == 4
    cfg = SubnetConfig(width=args.width, depths=depths, img_size=args.img_size, gray=True)

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    slim = ElasticFERSuperNet(num_classes=8, arch="v3_wide", slimmable=True)
    slim.load_state_dict(remap_v1_stem_keys(ckpt["model"]), strict=True)
    slim.eval()
    wrapped = StaticSubnet(slim, cfg).eval()
    baked = BakedFER(slim, cfg).eval()

    bin_dir = ROOT / "results/t527_v4/eval500_bins" / f"hw{args.img_size}"
    chw = np.fromfile(next(bin_dir.glob("*.bin")), dtype=np.float32).reshape(
        3, args.img_size, args.img_size
    )
    xt = torch.from_numpy(chw[None])
    with torch.no_grad():
        zs = wrapped(xt).numpy().reshape(-1)
        zb = baked(xt).numpy().reshape(-1)
    print(f"baked vs slim cos={cos(zb, zs):.8f} maxabs={float(np.abs(zb - zs).max()):.3e}")
    if cos(zb, zs) < 0.999999:
        print("FAIL bake numerical match")
        return 1

    onnx_path = OUT / "baked.onnx"
    torch.onnx.export(
        baked,
        xt,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo=False,
    )
    sim, ok = simplify(onnx.load(str(onnx_path)), skip_fuse_bn=False)
    print(f"onnxsim ok={ok}")
    sim_path = OUT / "baked_sim.onnx"
    onnx.save(sim, str(sim_path))

    sess = ort.InferenceSession(str(sim_path), providers=["CPUExecutionProvider"])
    zo = np.asarray(sess.run(None, {"input": chw[None]})[0], dtype=np.float32).reshape(-1)
    print(f"sim-onnx vs baked cos={cos(zo, zb):.8f}")

    if args.gold_onnx.exists():
        g = ort.InferenceSession(str(args.gold_onnx), providers=["CPUExecutionProvider"])
        name = g.get_inputs()[0].name
        zg = np.asarray(g.run(None, {name: chw[None]})[0], dtype=np.float32).reshape(-1)
        print(f"gold-onnx vs baked cos={cos(zg, zb):.8f}")

    param = OUT / "model.param"
    binp = OUT / "model.bin"
    print("onnx2ncnn...")
    run_wsl([f"{WSL_TOOLS}/onnx2ncnn", wsl_path(sim_path), wsl_path(param), wsl_path(binp)])
    n_patch = patch_nan_axes(binp)
    a = np.fromfile(binp, dtype=np.float32)
    print(f"raw ncnn bin nan={np.isnan(a).sum()} inf={np.isinf(a).sum()} patched_axes={n_patch}")

    if not args.skip_optimize:
        run_wsl(
            [
                f"{WSL_TOOLS}/ncnnoptimize",
                wsl_path(param),
                wsl_path(binp),
                wsl_path(param),
                wsl_path(binp),
                "0",
            ]
        )
        a = np.fromfile(binp, dtype=np.float32)
        print(f"opt ncnn bin nan={np.isnan(a).sum()} inf={np.isinf(a).sum()}")

    rgb = np.ascontiguousarray(
        (np.clip(chw.transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    )
    zn = ncnn_logits(param, binp, rgb)
    finite = bool(np.isfinite(zn).all())
    c = cos(zn, zb) if finite else float("nan")
    print(f"ncnn finite={finite} vs_baked cos={c:.8f}")
    print("ncnn ", np.round(zn, 4))
    print("baked", np.round(zb, 4))
    ok_pass = finite and c > 0.999
    print("PASS" if ok_pass else "FAIL")
    return 0 if ok_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
