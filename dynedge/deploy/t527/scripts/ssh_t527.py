#!/usr/bin/env python3
"""SSH helper for MYIR T527 Buildroot (no serial)."""
from __future__ import annotations

import argparse
import os
import sys

import paramiko
from scp import SCPClient

DEFAULT_HOST = os.environ.get("T527_HOST", "BOARD_IP_REDACTED")
DEFAULT_USER = os.environ.get("T527_USER", "root")
DEFAULT_PASS = os.environ.get("T527_PASS", "root")


def connect(host: str = DEFAULT_HOST, user: str = DEFAULT_USER, password: <REDACTED> = DEFAULT_PASS):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        host,
        username=user,
        password=<REDACTED>
        timeout=15,
        allow_agent=False,
        look_for_keys=False,
    )
    return c


def run(cmd: str, host: str = DEFAULT_HOST, timeout: int = 600) -> tuple[int, str, str]:
    c = connect(host)
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        return code, out, err
    finally:
        c.close()


def put(local: str, remote: str, host: str = DEFAULT_HOST) -> None:
    c = connect(host)
    try:
        with SCPClient(c.get_transport()) as scp:
            scp.put(local, remote, recursive=os.path.isdir(local))
    finally:
        c.close()


def get(remote: str, local: str, host: str = DEFAULT_HOST) -> None:
    c = connect(host)
    try:
        with SCPClient(c.get_transport()) as scp:
            scp.get(remote, local, recursive=True)
    finally:
        c.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", help="remote shell command")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    args = ap.parse_args()
    if args.put:
        put(args.put[0], args.put[1], args.host)
        print(f"PUT {args.put[0]} -> {args.put[1]}")
        return 0
    if args.get:
        get(args.get[0], args.get[1], args.host)
        print(f"GET {args.get[0]} -> {args.get[1]}")
        return 0
    if not args.cmd:
        ap.error("need cmd or --put/--get")
    code, out, err = run(args.cmd, args.host)
    sys.stdout.write(out)
    sys.stderr.write(err)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
