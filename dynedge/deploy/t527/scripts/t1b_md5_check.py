#!/usr/bin/env python3
"""T1b: md5 board vs host model files."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_t527 import run  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
STEM = "subnet_ferplus_w0.35_d2222_s64g_seed42_v4_predistill_img64"
MODEL = ROOT / "deploy" / "t527" / "models" / "v4_predistill"
REMOTE_DIR = "/root/fer-dynedge/models/v4_predistill"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    names = [f"{STEM}.param", f"{STEM}.bin", f"{STEM}_int8.param", f"{STEM}_int8.bin"]
    ok = True
    for name in names:
        local = MODEL / name
        code, out, _ = run(f"md5sum {REMOTE_DIR}/{name}")
        remote_md5 = (out.split() or [""])[0]
        local_md5 = md5(local)
        match = remote_md5 == local_md5 and code == 0
        ok = ok and match
        print(f"{name}: local={local_md5} remote={remote_md5} {'OK' if match else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
