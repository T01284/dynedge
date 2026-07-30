#!/usr/bin/env python3
"""Portable P5S1 voltage/current capture CLI for fer-dynedge power runs.

Examples:
  python -m bench.p5s1.capture list
  python -m bench.p5s1.capture info --port SERIAL_PORT
  python -m bench.p5s1.capture capture --port SERIAL_PORT --device esp32s3 \\
      --rate 100 --duration 30 --out results/power/esp32s3_probe.csv
  python -m bench.p5s1.capture capture --port SERIAL_PORT --device orin --rate 100 \\
      --interactive --out results/power/orin_run.csv
  python -m bench.p5s1.capture summarize --csv results/power/orin_run.csv \\
      --n-infer 1000 --device orin --out results/power/orin_run_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench.p5s1.energy import load_capture_csv, summarize_capture, write_summary_json
from bench.p5s1.meter import DEFAULT_BAUD, Meter, list_serial_ports
from bench.p5s1.protocol import (
    PAPER_I_RANGE_A,
    PAPER_I_RESOLUTION_MA,
    assert_golden_vectors,
)


CSV_FIELDS = [
    "host_s",
    "t_ms",
    "voltage_v",
    "current_a",
    "power_w",
    "current_ma",
    "power_mw",
    "mark",
]


def cmd_list(_: argparse.Namespace) -> int:
    ports = list_serial_ports()
    if not ports:
        print("no serial ports found")
        return 1
    for p in ports:
        print(p)
    return 0


def cmd_selftest(_: argparse.Namespace) -> int:
    assert_golden_vectors()
    print("golden vectors OK (START/STOP/SET_RATE_10)")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    with Meter(args.port, baud=args.baud) as m:
        m.start()
        if args.rate:
            m.set_sample_rate(args.rate)
        info = m.wait_info(timeout_s=args.timeout)
        print(json.dumps(info.__dict__, indent=2, ensure_ascii=False))
        m.stop()
    return 0


def _interactive_marker(stop_evt: threading.Event, marks: list[tuple[float, str]]) -> None:
    """Read stdin lines: start / stop / idle_start / idle_stop / quit."""
    print(
        "interactive marks: type start | stop | idle_start | idle_stop | quit",
        flush=True,
    )
    while not stop_evt.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            time.sleep(0.05)
            continue
        cmd = line.strip().lower()
        if not cmd:
            continue
        if cmd in {"q", "quit", "exit"}:
            stop_evt.set()
            break
        if cmd in {"start", "stop", "idle_start", "idle_stop"}:
            marks.append((time.monotonic(), cmd))
            print(f"# mark {cmd} @ {marks[-1][0]:.3f}", flush=True)
        else:
            print(f"# unknown mark '{cmd}'", flush=True)


def cmd_capture(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stop_evt = threading.Event()
    marks: list[tuple[float, str]] = []
    marker_th = None
    if args.interactive:
        marker_th = threading.Thread(
            target=_interactive_marker, args=(stop_evt, marks), daemon=True
        )
        marker_th.start()

    meta = {
        "device": args.device,
        "label": args.label,
        "port": args.port,
        "baud": args.baud,
        "rate_hz": args.rate,
        "started_host_s": time.time(),
    }

    n_written = 0
    last_print = 0.0
    with Meter(args.port, baud=args.baud) as m, open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        m.start()
        # Protocol note: lock link with START first, then configure rate.
        if args.rate:
            time.sleep(0.05)
            m.set_sample_rate(args.rate)
        try:
            info = m.wait_info(timeout_s=args.timeout)
            meta["meter_model"] = info.model
            meta["meter_version"] = info.version
            meta["v_range_v"] = info.voltage_range_v
            meta["i_range_a"] = info.current_range_a
            meta["paper_i_range_a"] = PAPER_I_RANGE_A
            meta["paper_i_resolution_ma"] = PAPER_I_RESOLUTION_MA
            meta["digits"] = info.digits
            meta["sample_rate_hz"] = info.sample_rate_hz
            print(
                f"# {info.model} {info.version}  "
                f"Vrange={info.voltage_range_v}V  Irange={info.current_range_a}A  "
                f"rate={info.sample_rate_hz}Hz  "
                f"(paper ±{PAPER_I_RANGE_A}A / {PAPER_I_RESOLUTION_MA}mA)",
                flush=True,
            )
            if abs(info.current_range_a - PAPER_I_RANGE_A) > 0.05:
                print(
                    f"# WARN: Irange={info.current_range_a}A ≠ paper ±{PAPER_I_RANGE_A}A",
                    flush=True,
                )
        except TimeoutError as e:
            print(f"# WARN: {e} (continuing; DATA may still arrive)", flush=True)

        t_start = time.monotonic()
        duration = args.duration
        try:
            for host_t, s in m.iter_samples(
                duration_s=None if args.interactive else duration
            ):
                if stop_evt.is_set():
                    break
                if duration is not None and not args.interactive:
                    if (host_t - t_start) >= duration:
                        break
                mark = ""
                while marks and marks[0][0] <= host_t:
                    mark = marks.pop(0)[1]
                row = {
                    "host_s": f"{host_t:.6f}",
                    "t_ms": s.t_ms,
                    "voltage_v": f"{s.voltage_v:.9f}",
                    "current_a": f"{s.current_a:.12f}",
                    "power_w": f"{s.power_w:.12f}",
                    "current_ma": f"{s.current_a * 1e3:.6f}",
                    "power_mw": f"{s.power_w * 1e3:.6f}",
                    "mark": mark,
                }
                writer.writerow(row)
                n_written += 1
                if args.live and (host_t - last_print) >= args.live_interval:
                    print(
                        f"t={s.t_ms}ms  V={s.voltage_v:.4f}V  "
                        f"I={s.current_a * 1e3:.3f}mA  P={s.power_w * 1e3:.3f}mW",
                        flush=True,
                    )
                    last_print = host_t
                if args.max_samples and n_written >= args.max_samples:
                    break
        except KeyboardInterrupt:
            print("\n# interrupted", flush=True)
        finally:
            stop_evt.set()
            m.stop()

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta["n_samples"] = n_written
    meta["ended_host_s"] = time.time()
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"# wrote {n_written} samples → {out}")
    print(f"# meta → {meta_path}")
    if marker_th is not None:
        marker_th.join(timeout=0.2)
    return 0 if n_written > 0 else 2


def _parse_window(spec: str | None) -> tuple[float, float] | None:
    if not spec:
        return None
    a, b = spec.split(",")
    return float(a), float(b)


def cmd_summarize(args: argparse.Namespace) -> int:
    rows = load_capture_csv(args.csv)
    # Auto-detect mark windows from CSV if present.
    active_host = _parse_window(args.active_host)
    idle_host = _parse_window(args.idle_host)
    active_t = _parse_window(args.active_t_ms)
    idle_t = _parse_window(args.idle_t_ms)

    if active_host is None and active_t is None:
        # Derive from mark column: start..stop
        starts = [float(r["host_s"]) for r in rows if r.get("mark") == "start"]
        stops = [float(r["host_s"]) for r in rows if r.get("mark") == "stop"]
        if starts and stops and stops[-1] > starts[0]:
            active_host = (starts[0], stops[-1])
    if idle_host is None and idle_t is None:
        i0 = [float(r["host_s"]) for r in rows if r.get("mark") == "idle_start"]
        i1 = [float(r["host_s"]) for r in rows if r.get("mark") == "idle_stop"]
        if i0 and i1 and i1[-1] > i0[0]:
            idle_host = (i0[0], i1[-1])

    meta_rate = args.rate
    meta_path = Path(args.csv).with_suffix(Path(args.csv).suffix + ".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_rate = int(meta.get("sample_rate_hz") or meta.get("rate_hz") or meta_rate)
        device = args.device or meta.get("device", "")
        label = args.label or meta.get("label", "")
    else:
        device = args.device
        label = args.label

    summary = summarize_capture(
        rows,
        device=device,
        label=label,
        n_infer=args.n_infer,
        sample_rate_hz=meta_rate,
        active_t_ms=active_t,
        idle_t_ms=idle_t,
        active_host_s=active_host,
        idle_host_s=idle_host,
        notes=args.notes,
    )
    text = json.dumps(
        {
            "device": summary.device,
            "label": summary.label,
            "n_infer": summary.n_infer,
            "active_duration_s": summary.active.duration_s,
            "v_mean": summary.active.v_mean,
            "i_mean_ma": summary.active.i_mean_ma,
            "p_mean_w": summary.active.p_mean_w,
            "energy_mj_gross": summary.energy_mj_gross,
            "energy_mj_per_infer_gross": summary.energy_mj_per_infer_gross,
            "energy_mj_net": summary.energy_mj_net,
            "energy_mj_per_infer_net": summary.energy_mj_per_infer_net,
            "idle_p_mean_w": summary.idle.p_mean_w if summary.idle else None,
        },
        indent=2,
        ensure_ascii=False,
    )
    print(text)
    if args.out:
        write_summary_json(summary, args.out)
        print(f"# summary → {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="P5S1 portable power capture")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list serial ports")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("selftest", help="verify CRC golden vectors")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("info", help="START and print device INFO")
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--rate", type=int, choices=[1, 10, 100], default=None)
    p.add_argument("--timeout", type=float, default=3.0)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("capture", help="stream samples to CSV")
    p.add_argument("--port", required=True)
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--rate", type=int, choices=[1, 10, 100], default=100)
    p.add_argument("--duration", type=float, default=30.0, help="seconds (ignored with --interactive)")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--device", default="", help="orin|t527|h618|esp32s3|esp32p4")
    p.add_argument("--label", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=float, default=3.0)
    p.add_argument("--live", action="store_true")
    p.add_argument("--live-interval", type=float, default=0.5)
    p.add_argument(
        "--interactive",
        action="store_true",
        help="mark start/stop/idle_* from stdin; Ctrl+C or 'quit' to end",
    )
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("summarize", help="integrate CSV → mJ/inference")
    p.add_argument("--csv", required=True)
    p.add_argument("--n-infer", type=int, required=True)
    p.add_argument("--device", default="")
    p.add_argument("--label", default="")
    p.add_argument("--rate", type=int, default=100)
    p.add_argument("--active-host", default=None, help="host_s0,host_s1")
    p.add_argument("--idle-host", default=None, help="host_s0,host_s1")
    p.add_argument("--active-t-ms", default=None, help="t_ms0,t_ms1")
    p.add_argument("--idle-t-ms", default=None, help="t_ms0,t_ms1")
    p.add_argument("--notes", default="")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_summarize)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
