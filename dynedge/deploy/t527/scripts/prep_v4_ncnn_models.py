#!/usr/bin/env python3
"""Host-side: convert v4 ONNX → ncnn fp32/int8 for T527 (board has no onnx2ncnn).

Uses WSL ncnn tools under ~/opt/ncnn-x86-tools/bin.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ONNX_DIR = ROOT / "results" / "h618_onnx_v4"
OUT_DIR = ROOT / "deploy" / "t527" / "models" / "v4_predistill"
CALIB112 = ROOT / "data" / "calib" / "linux112" / "calib"
CALIB64 = ROOT / "data" / "calib" / "ferplus_mcu64"
WSL_TOOLS = "<home>/opt/ncnn-x86-tools/bin"


def in_wsl_or_linux() -> bool:
    return sys.platform.startswith("linux")


def wsl_path(p: Path) -> str:
    s = str(p.resolve())
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:].replace(chr(92), '/')}"
    return s


def run_tools(args: list[str]) -> subprocess.CompletedProcess:
    """Run ncnn host tools. Prefer native exec when already inside WSL/Linux."""
    if in_wsl_or_linux():
        return subprocess.run(args, capture_output=True, text=True)
    cmd = ["wsl", "-e", "bash", "-lc", " ".join(shlex.quote(a) for a in args)]
    return subprocess.run(cmd, capture_output=True, text=True)


def shape_hwc(stem: str) -> tuple[int, int, int]:
    if "img64" in stem or "_s64g_" in stem:
        return 64, 64, 3
    if "img96" in stem or "_s96c_" in stem:
        return 96, 96, 3
    return 112, 112, 3


def ensure_imagelist(calib_root: Path, out: Path, limit: int = 0) -> Path:
    files = sorted(
        [
            p
            for p in calib_root.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    if limit:
        files = files[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    # Paths must be WSL-visible for ncnn2table running in WSL.
    out.write_text("\n".join(wsl_path(p) for p in files) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    onnx_list = sorted(ONNX_DIR.glob("*.onnx"))
    if args.limit:
        onnx_list = onnx_list[: args.limit]
    if not onnx_list:
        print(f"ERR no onnx under {ONNX_DIR}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    list112 = ensure_imagelist(CALIB112, OUT_DIR / "imagelist_real.txt")
    list64 = ensure_imagelist(CALIB64, OUT_DIR / "imagelist_mcu64.txt", limit=200)

    ok = fail = 0
    for onnx in onnx_list:
        stem = onnx.stem
        param = OUT_DIR / f"{stem}.param"
        binp = OUT_DIR / f"{stem}.bin"
        iparam = OUT_DIR / f"{stem}_int8.param"
        ibin = OUT_DIR / f"{stem}_int8.bin"
        table = OUT_DIR / f"{stem}.table"
        h, w, c = shape_hwc(stem)
        shape = f"[{h},{w},{c}]"
        calib = list64 if h == 64 else list112

        print(f"== {stem} {h}x{w} ==", flush=True)
        if not param.exists() or not binp.exists():
            r = run_tools(
                [
                    f"{WSL_TOOLS}/onnx2ncnn",
                    wsl_path(onnx),
                    wsl_path(param),
                    wsl_path(binp),
                ]
            )
            if r.returncode != 0 or not param.exists():
                print(r.stdout[-500:], r.stderr[-800:], file=sys.stderr)
                print(f"FAIL onnx2ncnn {stem}", flush=True)
                fail += 1
                continue
            run_tools(
                [
                    f"{WSL_TOOLS}/ncnnoptimize",
                    wsl_path(param),
                    wsl_path(binp),
                    wsl_path(param),
                    wsl_path(binp),
                    "0",
                ]
            )

        if not args.skip_int8 and (not iparam.exists() or not ibin.exists()):
            table.unlink(missing_ok=True)
            # Host ncnn2table (a4d2ea1): method=kl segfaults in build_histogram;
            # method=aciq works. H618 board used kl — note in SUMMARY.
            r = run_tools(
                [
                    f"{WSL_TOOLS}/ncnn2table",
                    wsl_path(param),
                    wsl_path(binp),
                    wsl_path(calib),
                    wsl_path(table),
                    "mean=[0,0,0]",
                    "norm=[0.00392157,0.00392157,0.00392157]",
                    f"shape={shape}",
                    "pixel=RGB",
                    "thread=1",
                    "method=aciq",
                ]
            )
            if r.returncode != 0 or not table.exists() or table.stat().st_size == 0:
                print((r.stdout or "")[-400:], (r.stderr or "")[-600:], file=sys.stderr)
                print(f"FAIL table {stem}", flush=True)
                fail += 1
                continue
            # Replace non-finite / absurd scales with per-layer max finite
            # (blanket inf→1e6 destroys INT8 accuracy).
            import math

            lines = table.read_text(encoding="utf-8", errors="replace").splitlines()
            n_fix = 0
            out_lines: list[str] = []
            for line in lines:
                if not line.strip():
                    out_lines.append(line)
                    continue
                parts = line.split()
                name, vals = parts[0], parts[1:]
                nums: list[float] = []
                for v in vals:
                    try:
                        nums.append(float(v))
                    except ValueError:
                        nums.append(float("nan"))
                finite = [abs(x) for x in nums if math.isfinite(x) and 0 < abs(x) <= 1e5]
                fill = max(finite) if finite else 1.0
                new_vals: list[str] = []
                for x in nums:
                    if (not math.isfinite(x)) or abs(x) > 1e5:
                        new_vals.append(f"{fill:.6f}")
                        n_fix += 1
                    else:
                        new_vals.append(f"{x:.6f}")
                out_lines.append(name + " " + " ".join(new_vals) + " ")
            if n_fix:
                table.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
                print(f"WARN patched {n_fix} nonfinite/huge scales in {table.name}", flush=True)
            r2 = run_tools(
                [
                    f"{WSL_TOOLS}/ncnn2int8",
                    wsl_path(param),
                    wsl_path(binp),
                    wsl_path(iparam),
                    wsl_path(ibin),
                    wsl_path(table),
                ]
            )
            if r2.returncode != 0 or not iparam.exists():
                print((r2.stdout or "")[-400:], (r2.stderr or "")[-600:], file=sys.stderr)
                print(f"FAIL int8 {stem}", flush=True)
                fail += 1
                continue
        print(f"OK {stem}", flush=True)
        ok += 1

    print(f"DONE ok={ok} fail={fail} -> {OUT_DIR}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
