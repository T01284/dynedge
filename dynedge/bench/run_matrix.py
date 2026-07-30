#!/usr/bin/env python3
"""SSH dispatch / run / collect stub for cross-board latency matrices.

Per-board matrix entrypoints live under ``dynedge/deploy/*/scripts/``.
This module is the stable CLI name referenced in the paper supplementary
material; it forwards to those scripts after loading ``config.local.yaml``.

Examples
--------
python dynedge/bench/run_matrix.py --board h618 --list
python dynedge/bench/run_matrix.py --board t527 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BOARD_ENTRY = {
    "h618": "dynedge/deploy/h618/scripts",
    "t527": "dynedge/deploy/t527/scripts",
    "orin": "dynedge/deploy/orin/scripts",
    "esp32": "dynedge/deploy/esp32/tools",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--board", choices=sorted(BOARD_ENTRY), required=True)
    ap.add_argument("--list", action="store_true", help="list available scripts for the board")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--config",
        default="config.local.yaml",
        help="local board/path config (gitignored); copy from dynedge/configs/config.example.yaml",
    )
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    scripts = root / BOARD_ENTRY[args.board]
    if not scripts.is_dir():
        print(f"[run_matrix] missing scripts dir: {scripts}", file=sys.stderr)
        return 2

    files = sorted(p.name for p in scripts.iterdir() if p.suffix in {".py", ".sh"})
    print(f"[run_matrix] board={args.board} config={args.config}")
    print(f"[run_matrix] scripts dir: {scripts}")
    if args.list or args.dry-run:
        for n in files:
            print(f"  - {n}")
        print(
            "Wire SSH host/user from config.local.yaml, then invoke the matching "
            "run_*_matrix / run_latency_matrix / board_check script."
        )
        return 0

    print(
        "No default matrix attached yet — pick a script from --list and run it "
        "with board endpoints from config.local.yaml.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
