"""Energy helpers for P5S1 capture CSVs.

Integrates P(t)=V*I with device timestamps (trapezoidal). Supports idle
baseline subtraction for mJ/inference reporting.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass
class WindowStats:
    n_samples: int
    duration_s: float
    v_mean: float
    i_mean: float
    i_mean_ma: float
    p_mean_w: float
    energy_j: float
    energy_mj: float

    @property
    def energy_mj_per_infer(self) -> Optional[float]:
        return None  # filled by summarize when n_infer given


@dataclass
class EnergySummary:
    device: str
    label: str
    n_infer: int
    sample_rate_hz: int
    active: WindowStats
    idle: Optional[WindowStats]
    energy_mj_gross: float
    energy_mj_net: Optional[float]
    energy_mj_per_infer_gross: float
    energy_mj_per_infer_net: Optional[float]
    notes: str = ""


def _rows_in_window(
    rows: Sequence[dict],
    t0_ms: Optional[float],
    t1_ms: Optional[float],
    host0: Optional[float] = None,
    host1: Optional[float] = None,
) -> list[dict]:
    out = []
    for r in rows:
        if t0_ms is not None and t1_ms is not None:
            t = float(r["t_ms"])
            if t < t0_ms or t > t1_ms:
                continue
        elif host0 is not None and host1 is not None:
            h = float(r["host_s"])
            if h < host0 or h > host1:
                continue
        out.append(r)
    return out


def integrate_power(rows: Sequence[dict]) -> WindowStats:
    if len(rows) < 2:
        if not rows:
            return WindowStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        r = rows[0]
        v, i = float(r["voltage_v"]), float(r["current_a"])
        p = v * i
        return WindowStats(1, 0.0, v, i, i * 1e3, p, 0.0, 0.0)

    # Prefer device time; fall back to host monotonic.
    use_host = False
    try:
        ts = [float(r["t_ms"]) / 1000.0 for r in rows]
        if ts[-1] <= ts[0]:
            use_host = True
    except (KeyError, ValueError):
        use_host = True
    if use_host:
        ts = [float(r["host_s"]) for r in rows]

    vs = [float(r["voltage_v"]) for r in rows]
    is_ = [float(r["current_a"]) for r in rows]
    ps = [v * i for v, i in zip(vs, is_)]

    energy = 0.0
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        if dt <= 0:
            continue
        energy += 0.5 * (ps[k] + ps[k - 1]) * dt

    duration = max(0.0, ts[-1] - ts[0])
    return WindowStats(
        n_samples=len(rows),
        duration_s=duration,
        v_mean=sum(vs) / len(vs),
        i_mean=sum(is_) / len(is_),
        i_mean_ma=(sum(is_) / len(is_)) * 1e3,
        p_mean_w=sum(ps) / len(ps),
        energy_j=energy,
        energy_mj=energy * 1000.0,
    )


def load_capture_csv(path: Path | str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_capture(
    rows: Sequence[dict],
    *,
    device: str = "",
    label: str = "",
    n_infer: int = 1,
    sample_rate_hz: int = 100,
    active_t_ms: Optional[tuple[float, float]] = None,
    idle_t_ms: Optional[tuple[float, float]] = None,
    active_host_s: Optional[tuple[float, float]] = None,
    idle_host_s: Optional[tuple[float, float]] = None,
    notes: str = "",
) -> EnergySummary:
    if active_t_ms is not None:
        active_rows = _rows_in_window(rows, active_t_ms[0], active_t_ms[1])
    elif active_host_s is not None:
        active_rows = _rows_in_window(
            rows, None, None, active_host_s[0], active_host_s[1]
        )
    else:
        active_rows = list(rows)

    active = integrate_power(active_rows)
    idle_stats = None
    energy_net = None
    per_infer_net = None
    if idle_t_ms is not None or idle_host_s is not None:
        if idle_t_ms is not None:
            idle_rows = _rows_in_window(rows, idle_t_ms[0], idle_t_ms[1])
        else:
            idle_rows = _rows_in_window(
                rows, None, None, idle_host_s[0], idle_host_s[1]  # type: ignore[index]
            )
        idle_stats = integrate_power(idle_rows)
        # Scale idle energy to active duration: E_net = E_act - P_idle * T_act
        if idle_stats.duration_s > 0 and active.duration_s > 0:
            p_idle = idle_stats.energy_j / idle_stats.duration_s
            energy_net = (active.energy_j - p_idle * active.duration_s) * 1000.0
            per_infer_net = energy_net / max(1, n_infer)

    n = max(1, n_infer)
    return EnergySummary(
        device=device,
        label=label,
        n_infer=n_infer,
        sample_rate_hz=sample_rate_hz,
        active=active,
        idle=idle_stats,
        energy_mj_gross=active.energy_mj,
        energy_mj_net=energy_net,
        energy_mj_per_infer_gross=active.energy_mj / n,
        energy_mj_per_infer_net=per_infer_net,
        notes=notes,
    )


def write_summary_json(summary: EnergySummary, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
