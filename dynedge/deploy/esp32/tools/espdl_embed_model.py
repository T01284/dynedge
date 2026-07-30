#!/usr/bin/env python3
"""Regenerate firmware/main/model_espdl_data.c from a chosen .espdl."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--espdl", type=Path, required=True)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "deploy/mcu_espdl/firmware/main/model_espdl_data.c",
    )
    args = ap.parse_args()
    data = args.espdl.read_bytes()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#include <stdint.h>\n#include <stddef.h>\n")
        f.write(f"const size_t model_espdl_len = {len(data)};\n")
        f.write("const uint8_t model_espdl[] __attribute__((aligned(16))) = {\n")
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n")
    print(f"wrote {args.out} ({len(data)} bytes from {args.espdl.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
