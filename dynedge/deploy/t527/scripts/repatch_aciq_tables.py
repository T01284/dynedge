#!/usr/bin/env python3
"""Re-patch aciq tables: replace nonfinite/huge scales with per-layer max finite, then ncnn2int8."""
from __future__ import annotations

import math
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "deploy" / "t527" / "models" / "v4_predistill"
WSL_TOOLS = "<home>/opt/ncnn-x86-tools/bin"


def wsl_path(p: Path) -> str:
    s = str(p.resolve())
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:].replace(chr(92), '/')}"
    return s


def run_tools(args: list[str]) -> subprocess.CompletedProcess:
    if sys.platform.startswith("linux"):
        return subprocess.run(args, capture_output=True, text=True)
    cmd = ["wsl", "-e", "bash", "-lc", " ".join(shlex.quote(a) for a in args)]
    return subprocess.run(cmd, capture_output=True, text=True)


def patch_table(path: Path) -> int:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n_fix = 0
    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        parts = line.split()
        name, vals = parts[0], parts[1:]
        nums: list[float] = []
        for v in vals:
            try:
                nums.append(float(v))
            except ValueError:
                nums.append(float("nan"))
        finite = [abs(x) for x in nums if math.isfinite(x) and abs(x) > 0]
        fill = max(finite) if finite else 1.0
        new_vals: list[str] = []
        for x in nums:
            if (not math.isfinite(x)) or abs(x) > 1e5:
                new_vals.append(f"{fill:.6f}")
                n_fix += 1
            else:
                new_vals.append(f"{x:.6f}")
        out.append(name + " " + " ".join(new_vals) + " ")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return n_fix


def main() -> int:
    ok = fail = 0
    for param in sorted(MODEL.glob("*.param")):
        if "_int8" in param.name:
            continue
        stem = param.stem
        binp = MODEL / f"{stem}.bin"
        table = MODEL / f"{stem}.table"
        iparam = MODEL / f"{stem}_int8.param"
        ibin = MODEL / f"{stem}_int8.bin"
        if not table.exists():
            print(f"SKIP no table {stem}", flush=True)
            fail += 1
            continue
        n = patch_table(table)
        iparam.unlink(missing_ok=True)
        ibin.unlink(missing_ok=True)
        r = run_tools(
            [
                f"{WSL_TOOLS}/ncnn2int8",
                wsl_path(param),
                wsl_path(binp),
                wsl_path(iparam),
                wsl_path(ibin),
                wsl_path(table),
            ]
        )
        if r.returncode != 0 or not iparam.exists():
            print(f"FAIL {stem} {(r.stderr or r.stdout or '')[-300:]}", flush=True)
            fail += 1
        else:
            print(f"OK {stem} patched={n}", flush=True)
            ok += 1
    print(f"DONE ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
