#!/usr/bin/env python3
"""Expand v3_wide/v4 SubnetConfig into operator rows for LUT mining / prediction.

Uses ARCH_SPECS['v3_wide'] from models.elastic_mbv3 (stem=32, stages 48/72/144/288, SE).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.elastic_mbv3 import ARCH_SPECS, make_divisible  # noqa: E402


def scaled(ch: int, width: float) -> int:
    return make_divisible(ch * width)


def expand_subnet_ops(
    *,
    width: float,
    depths: tuple[int, int, int, int],
    img_size: int,
    dtype: str,
    backend: str,
    arch: str = "v3_wide",
    include_se: bool = True,
    include_add: bool = True,
    include_gap: bool = True,
) -> list[dict]:
    spec = ARCH_SPECS[arch]
    stem_ch = scaled(int(spec["stem"]), width)
    head_mid = scaled(int(spec["head_mid"]), width)
    rows: list[dict] = []

    def add(row: dict) -> None:
        rows.append(
            {
                **row,
                "width": width,
                "backend": backend,
                "dtype": dtype,
                "img_size": img_size,
            }
        )

    h = img_size
    # stem s=2
    add(
        {
            "op": "conv",
            "k": 3,
            "s": 2,
            "cin": 3,
            "cout": stem_ch,
            "h": h,
            "w": h,
            "groups": 1,
            "note": "stem",
        }
    )
    h = h // 2
    if spec.get("stem_extra"):
        add(
            {
                "op": "conv",
                "k": 3,
                "s": 1,
                "cin": stem_ch,
                "cout": stem_ch,
                "h": h,
                "w": h,
                "groups": 1,
                "note": "stem_extra",
            }
        )

    in_ch = stem_ch
    for si, (out_base, stride, expand) in enumerate(spec["stages"]):
        out_ch = scaled(int(out_base), width)
        for bi in range(int(depths[si])):
            s = int(stride) if bi == 0 else 1
            mid = make_divisible(in_ch * int(expand))
            oh = h // s
            add(
                {
                    "op": "conv",
                    "k": 1,
                    "s": 1,
                    "cin": in_ch,
                    "cout": mid,
                    "h": h,
                    "w": h,
                    "groups": 1,
                    "note": f"ir_pw_exp_s{si}_b{bi}",
                }
            )
            add(
                {
                    "op": "dwconv",
                    "k": 3,
                    "s": s,
                    "cin": mid,
                    "cout": mid,
                    "h": h,
                    "w": h,
                    "groups": mid,
                    "note": f"ir_dw_s{si}_b{bi}",
                }
            )
            if include_se and spec.get("use_se"):
                se_mid = max(8, mid // 4)
                add(
                    {
                        "op": "gemm",
                        "m": 1,
                        "k": mid,
                        "n": se_mid,
                        "note": f"se_down_s{si}_b{bi}",
                    }
                )
                add(
                    {
                        "op": "gemm",
                        "m": 1,
                        "k": se_mid,
                        "n": mid,
                        "note": f"se_up_s{si}_b{bi}",
                    }
                )
            add(
                {
                    "op": "conv",
                    "k": 1,
                    "s": 1,
                    "cin": mid,
                    "cout": out_ch,
                    "h": oh,
                    "w": oh,
                    "groups": 1,
                    "note": f"ir_pw_proj_s{si}_b{bi}",
                }
            )
            if include_add and s == 1 and in_ch == out_ch:
                add(
                    {
                        "op": "add",
                        "cin": out_ch,
                        "h": oh,
                        "w": oh,
                        "note": f"ir_res_s{si}_b{bi}",
                    }
                )
            h = oh
            in_ch = out_ch

    add(
        {
            "op": "conv",
            "k": 1,
            "s": 1,
            "cin": in_ch,
            "cout": head_mid,
            "h": h,
            "w": h,
            "groups": 1,
            "note": "head_pw",
        }
    )
    if include_gap:
        add(
            {
                "op": "gap",
                "cin": head_mid,
                "h": h,
                "w": h,
                "note": "gap",
            }
        )
    add(
        {
            "op": "gemm",
            "m": 1,
            "k": head_mid,
            "n": 8,
            "note": "fc",
        }
    )
    return rows


def op_key(o: dict) -> tuple:
    def norm(v):
        if v is None or v == "":
            return None
        return v

    return (
        o.get("op"),
        norm(o.get("cin")),
        norm(o.get("cout")),
        norm(o.get("k")),
        norm(o.get("s")),
        norm(o.get("h")),
        norm(o.get("w")),
        norm(o.get("m")),
        norm(o.get("n")),
        norm(o.get("groups")),
        norm(o.get("dtype")),
        norm(o.get("backend")),
    )


def dedupe_ops(rows: Iterable[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in rows:
        k = op_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out
