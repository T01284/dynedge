#!/usr/bin/env python3
"""T527: force each exit → measured cum_exit_ms for C3 retrain feed.

Schema exit_segment_latency_v1 — same shape as train_controller lat_meta, plus
board p50/p99. Align H618 when it lands the same schema.

NPU column: VIPLite NBG is single-exit e2e only (≈ force_final); no multi-exit NBG.
"""
from __future__ import annotations

import csv
import json
import re
import tarfile
import tempfile
from pathlib import Path

from ssh_t527 import put, run

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/t527_v4/exit_segment_latency"
T3_NCNN = ROOT / "results/t527_v4/t3_early_exit/ncnn"
BIN = ROOT / "deploy/ncnn_edge/build-linux-aarch64-acc/fer_early_exit"
CTRL = ROOT / "results/from_autodl/formal_v4/controller/ferplus_t527_b5ms/summary.json"
NPU_HS = ROOT / "results/t527_v4/npu_matrix_summary.csv"
NPU_RELU = ROOT / "results/t527_v4/npu_matrix_summary_relu.csv"
THREADS = 4
N = 80
SUBNET = "ferplus_w0.5_d3333_s112"


def parse_result(text: str) -> dict:
    m = re.search(r"^RESULT .+$", text, re.M)
    if not m:
        return {"raw": text}
    row = {"raw": m.group(0)}
    for p in m.group(0).split()[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            row[k] = v
    return row


def npu_ms(path: Path, needle: str) -> float | None:
    if not path.exists():
        return None
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if needle in r["model"]:
            return round(float(r["latency_us_p50"]) / 1000.0, 3)
    return None


def derive_seg_ms(cum: list[float]) -> list[float]:
    """Map 4 measured cums → 5 segs for controller features.

    Put entire first cum into seg[0]+seg[1] with seg[0]=0 so:
      cum[i] == sum(seg[: i+2]) for i in 0..3 when seg has len 5.
    next_seg at gate i uses seg[i+1]: gate0→seg1=cum0, gate1→seg2=Δ01, ...
    (stem folded into first decision cum; matches measured wall-clock).
    """
    c0, c1, c2, c3 = cum
    return [0.0, c0, c1 - c0, c2 - c1, c3 - c2]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bins = sorted((ROOT / "results/t527_v4/eval500_bins/hw112").glob("*.bin"))[:N]
    if len(bins) < N:
        raise SystemExit(f"need {N} bins, got {len(bins)}")
    if not BIN.exists():
        raise SystemExit(f"missing {BIN}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        tar_path = td_path / "bins.tar"
        with tarfile.open(tar_path, "w") as tar:
            for p in bins:
                tar.add(p, arcname=p.name)
        for ei in range(4):
            (td_path / f"exits_e{ei}.txt").write_text(
                "\n".join([str(ei)] * N) + "\n", encoding="utf-8"
            )

        run("rm -rf /root/seg_ee; mkdir -p /root/seg_ee/bins /root/seg_ee/models")
        put(str(BIN), "/root/seg_ee/fer_early_exit")
        put(str(T3_NCNN / "m.ncnn.param"), "/root/seg_ee/models/m.ncnn.param")
        put(str(T3_NCNN / "m.ncnn.bin"), "/root/seg_ee/models/m.ncnn.bin")
        put(str(tar_path), "/root/seg_ee/bins.tar")
        for ei in range(4):
            put(str(td_path / f"exits_e{ei}.txt"), f"/root/seg_ee/exits_e{ei}.txt")

    code, out, err = run(
        "cd /root/seg_ee && tar xf bins.tar -C bins && "
        "for f in bins/*; do echo /root/seg_ee/$f; done > list80.txt && "
        "chmod +x fer_early_exit && wc -l list80.txt exits_e*.txt",
        timeout=120,
    )
    print(out, err, "prep", code)
    if code != 0:
        return code

    common = (
        "./fer_early_exit --param models/m.ncnn.param --bin models/m.ncnn.bin "
        f"--list list80.txt --h 112 --w 112 --c 3 --threads {THREADS} "
        "--out0 out0 --out1 out1 --out2 out2 --out3 out3"
    )

    board: dict[str, dict] = {}
    for ei, name in enumerate(("exit2", "exit3", "exit4", "final")):
        extra = f" --exits exits_e{ei}.txt" if ei < 3 else " --force-final"
        # exit indices 0,1,2 via --exits; final via --force-final (same as e3 all-3s)
        if ei == 3:
            # also run constant-3 exits for parity
            code, o, e = run(f"cd /root/seg_ee && {common} --exits exits_e3.txt", timeout=300)
        else:
            code, o, e = run(f"cd /root/seg_ee && {common}{extra}", timeout=300)
        text = (o or "") + "\n" + (e or "")
        (OUT / f"board_force_{name}.txt").write_text(text, encoding="utf-8")
        row = parse_result(o or "")
        board[name] = row
        print(name, row.get("raw", text[-200:]), "code", code)
        if code != 0:
            return code

    cum_p50 = [float(board[k]["p50_ms"]) for k in ("exit2", "exit3", "exit4", "final")]
    cum_p99 = [float(board[k]["p99_ms"]) for k in ("exit2", "exit3", "exit4", "final")]
    seg = derive_seg_ms(cum_p50)

    lut = {}
    if CTRL.exists():
        lut = json.loads(CTRL.read_text(encoding="utf-8")).get("lat_meta", {})

    npu_mid_hs = npu_ms(NPU_HS, "w0.5_d3333_s112c_seed42_v4_predistill_img112")
    npu_mid_relu = npu_ms(NPU_RELU, "w0.5_d3333_s112c_seed42_v4_relu_npu_img112")

    payload = {
        "schema": "exit_segment_latency_v1",
        "device": "t527",
        "subnet": SUBNET,
        "threads": THREADS,
        "n": N,
        "img": 112,
        "cpu": {
            "backend": "ncnn_fp32_multi_exit",
            "dtype": "fp32",
            "cum_exit_ms": cum_p50,
            "cum_exit_ms_p99": cum_p99,
            "seg_ms": seg,
            "full_ms": cum_p50[-1],
            "source": "board_forced_exits@4t_n80",
            "board_raw": {k: board[k].get("raw") for k in board},
        },
        "lut_ref": {
            "cum_exit_ms": lut.get("cum_exit_ms"),
            "seg_ms": lut.get("seg_ms"),
            "full_ms": lut.get("full_ms"),
            "source": lut.get("source"),
            "note": "G5 AutoDL LUT prior; replace with cpu.* for band-aware retrain",
        },
        "npu": {
            "backend": "viplite_uint8",
            "note": (
                "No multi-exit NBG; only full-network e2e (≈ CPU final). "
                "Controller retrain on T527-CPU uses cpu.cum/seg; NPU column is e2e contrast."
            ),
            "e2e_ms_p50_hardswish": npu_mid_hs,
            "e2e_ms_p50_relu": npu_mid_relu,
            "model_hardswish": "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_predistill_img112",
            "model_relu": "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_relu_npu_img112",
        },
        "controller_feed": {
            "device": "t527",
            "dtype": "fp32",
            "seg_ms": seg,
            "cum_exit_ms": cum_p50,
            "full_ms": cum_p50[-1],
            "source": "results/t527_v4/exit_segment_latency/summary.json#cpu",
        },
    }
    out_json = OUT / "summary.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # flat CSV for tables
    with (OUT / "cum_exit_ms.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["device", "backend", "exit", "p50_ms", "p99_ms", "lut_ms"],
        )
        w.writeheader()
        lut_c = lut.get("cum_exit_ms") or [None] * 4
        for i, name in enumerate(("exit2", "exit3", "exit4", "final")):
            w.writerow(
                {
                    "device": "t527",
                    "backend": "ncnn_cpu",
                    "exit": name,
                    "p50_ms": cum_p50[i],
                    "p99_ms": cum_p99[i],
                    "lut_ms": lut_c[i] if i < len(lut_c) else "",
                }
            )
        w.writerow(
            {
                "device": "t527",
                "backend": "viplite_npu_hs",
                "exit": "final_e2e",
                "p50_ms": npu_mid_hs if npu_mid_hs is not None else "",
                "p99_ms": "",
                "lut_ms": "",
            }
        )
        w.writerow(
            {
                "device": "t527",
                "backend": "viplite_npu_relu",
                "exit": "final_e2e",
                "p50_ms": npu_mid_relu if npu_mid_relu is not None else "",
                "p99_ms": "",
                "lut_ms": "",
            }
        )
    print(f"OK -> {out_json}")
    print("cum_p50", cum_p50, "seg", [round(x, 3) for x in seg])
    print("lut_cum", lut.get("cum_exit_ms"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
