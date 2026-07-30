#!/usr/bin/env python3
"""TCP client for fer-dynedge ESP-DL CMD protocol (WiFi power-bench path).

Examples:
  python tools/espdl_tcp_client.py --host BOARD_IP_REDACTED INFO
  python tools/espdl_tcp_client.py --host BOARD_IP_REDACTED BENCH
  python tools/espdl_tcp_client.py --host BOARD_IP_REDACTED CHECK --until CHECK_DONE
"""
from __future__ import annotations

import argparse
import socket
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True, help="board STA IPv4")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--until",
        default="",
        help="stop reading when a line starts with this (e.g. BENCH_DONE / CHECK_DONE)",
    )
    ap.add_argument("cmd", nargs="+", help="command words, e.g. CMD BENCH or BENCH")
    args = ap.parse_args()

    line = " ".join(args.cmd).strip()
    if not line.upper().startswith("CMD ") and line.upper() not in {
        "INFO",
        "CHECK",
        "BENCH",
        "BENCH_SYN",
        "INFER",
        "IP",
    }:
        # allow LOGITS N
        pass
    if not line.upper().startswith("CMD "):
        # Prefer full form for consistency
        if line.upper().startswith("LOGITS"):
            line = "CMD " + line
        elif line.upper() in {"INFO", "CHECK", "BENCH", "BENCH_SYN", "INFER", "IP"}:
            line = "CMD " + line

    until = args.until
    if not until:
        u = line.upper()
        if "BENCH" in u:
            until = "BENCH_DONE"
        elif "CHECK" in u:
            until = "CHECK_DONE"
        elif "INFO" in u or u.endswith(" IP") or u == "CMD IP":
            until = "READY" if "INFO" in u else "IP "

    with socket.create_connection((args.host, args.port), timeout=10.0) as sock:
        sock.settimeout(args.timeout)
        # Drain boot banner briefly
        sock.settimeout(1.0)
        t0 = time.time()
        buf = b""
        while time.time() - t0 < 1.5:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            except socket.timeout:
                break
        if buf:
            sys.stdout.write(buf.decode("utf-8", "replace"))
            sys.stdout.flush()

        sock.settimeout(args.timeout)
        sock.sendall((line + "\n").encode("utf-8"))

        buf = b""
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                print("[timeout]", file=sys.stderr)
                return 2
            if not chunk:
                break
            buf += chunk
            text = buf.decode("utf-8", "replace")
            while "\n" in text:
                one, text = text.split("\n", 1)
                buf = text.encode("utf-8")
                print(one)
                if until and one.startswith(until.rstrip()):
                    return 0
                # INFO prints READY
                if until == "READY" and one.startswith("READY"):
                    return 0
                if until.startswith("IP") and one.startswith("IP "):
                    return 0
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
