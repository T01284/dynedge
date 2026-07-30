#!/usr/bin/env python3
"""T527 key-op microbench (A55 SDOT vs H618 reference shapes)."""
from __future__ import annotations

import csv
import json
import re
import struct
import tarfile
import tempfile
from pathlib import Path

from ssh_t527 import put, run

REPO = Path(__file__).resolve().parents[3]
LOCAL_BENCH = REPO / "deploy" / "ncnn_edge" / "build-linux-aarch64" / "fer_op_bench"
REMOTE_ROOT = "/root/fer-dynedge"
REMOTE_BENCH = f"{REMOTE_ROOT}/bin/fer_op_bench"
REMOTE_WORK = f"{REMOTE_ROOT}/lut_ops_smoke"
OUT_CSV = REPO / "results" / "t527_linux_op_smoke.csv"
OUT_LOG = REPO / "results" / "t527_linux_op_smoke.log"

# Representative shapes from MobileNetV3-Small @112 (and SE/FC).
OPS = [
    # stem 3x3 s2
    {"op": "conv", "cin": 3, "cout": 16, "k": 3, "s": 2, "h": 112, "w": 112},
    # early dw
    {"op": "dwconv", "cin": 16, "cout": 16, "k": 3, "s": 2, "h": 56, "w": 56},
    {"op": "conv", "cin": 16, "cout": 16, "k": 1, "s": 1, "h": 28, "w": 28},
    # mid IR
    {"op": "conv", "cin": 24, "cout": 88, "k": 1, "s": 1, "h": 28, "w": 28},
    {"op": "dwconv", "cin": 88, "cout": 88, "k": 5, "s": 2, "h": 28, "w": 28},
    {"op": "conv", "cin": 88, "cout": 24, "k": 1, "s": 1, "h": 14, "w": 14},
    {"op": "conv", "cin": 40, "cout": 120, "k": 1, "s": 1, "h": 14, "w": 14},
    {"op": "dwconv", "cin": 120, "cout": 120, "k": 5, "s": 1, "h": 14, "w": 14},
    {"op": "conv", "cin": 120, "cout": 40, "k": 1, "s": 1, "h": 14, "w": 14},
    {"op": "conv", "cin": 48, "cout": 144, "k": 1, "s": 1, "h": 14, "w": 14},
    {"op": "dwconv", "cin": 144, "cout": 144, "k": 5, "s": 1, "h": 14, "w": 14},
    {"op": "conv", "cin": 96, "cout": 576, "k": 1, "s": 1, "h": 7, "w": 7},
    {"op": "dwconv", "cin": 576, "cout": 576, "k": 5, "s": 1, "h": 7, "w": 7},
    {"op": "conv", "cin": 576, "cout": 96, "k": 1, "s": 1, "h": 7, "w": 7},
    # SE / classifier
    {"op": "gemm", "k": 96, "n": 24},
    {"op": "gemm", "k": 576, "n": 144},
    {"op": "gemm", "k": 576, "n": 1024},
    {"op": "gemm", "k": 1024, "n": 7},
    {"op": "add", "cin": 24, "h": 14, "w": 14},
    {"op": "add", "cin": 96, "h": 7, "w": 7},
    {"op": "gap", "cin": 576, "h": 7, "w": 7},
]


def _write_f32_layer(bin_f, weights, bias=None):
    bin_f.write(struct.pack("<i", 0))
    for v in weights:
        bin_f.write(struct.pack("<f", float(v)))
    if bias is not None:
        for v in bias:
            bin_f.write(struct.pack("<f", float(v)))


def write_conv(param, bin_path, *, cin, cout, k, s, groups, h, w):
    pad = k // 2
    if groups > 1 and groups == cin == cout:
        n_w = cout * k * k
        layer = (
            f"ConvolutionDepthWise conv0 1 1 input out "
            f"0={cout} 1={k} 11={k} 2=1 12=1 3={s} 13={s} "
            f"4={pad} 14={pad} 15={pad} 16={pad} 5=1 6={n_w} 7={groups}"
        )
    else:
        n_w = cin * cout * k * k // max(groups, 1)
        layer = (
            f"Convolution      conv0 1 1 input out "
            f"0={cout} 1={k} 11={k} 2=1 12=1 3={s} 13={s} "
            f"4={pad} 14={pad} 15={pad} 16={pad} 5=1 6={n_w}"
        )
    param.write_text(
        "\n".join(
            [
                "7767517",
                "2 2",
                f"Input input 0 1 input 0={w} 1={h} 2={cin}",
                layer,
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    with bin_path.open("wb") as f:
        _write_f32_layer(f, [0.01] * n_w, [0.0] * cout)


def write_gemm(param, bin_path, *, k, n):
    param.write_text(
        "\n".join(
            [
                "7767517",
                "2 2",
                f"Input input 0 1 input 0={k} 1=1 2=1",
                f"InnerProduct fc0 1 1 input out 0={n} 1=1 2={k * n}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    with bin_path.open("wb") as f:
        _write_f32_layer(f, [0.01] * (k * n), [0.0] * n)


def write_add(param, bin_path, *, cin, h, w):
    param.write_text(
        "\n".join(
            [
                "7767517",
                "3 3",
                f"Input input 0 1 input 0={w} 1={h} 2={cin}",
                "Split split0 1 2 input a b",
                "Eltwise add0 2 1 a b out 0=1",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    bin_path.write_bytes(b"")


def write_gap(param, bin_path, *, cin, h, w):
    param.write_text(
        "\n".join(
            [
                "7767517",
                "2 2",
                f"Input input 0 1 input 0={w} 1={h} 2={cin}",
                "Pooling gap0 1 1 input out 0=1 4=1",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    bin_path.write_bytes(b"")


def pack_ops(work: Path) -> list[dict]:
    meta = []
    for i, o in enumerate(OPS):
        name = f"{i:02d}_{o['op']}"
        p = work / f"{name}.param"
        b = work / f"{name}.bin"
        if o["op"] in {"conv", "dwconv"}:
            cin, cout = o["cin"], o["cout"]
            groups = cin if o["op"] == "dwconv" else 1
            write_conv(
                p, b, cin=cin, cout=cout, k=o["k"], s=o["s"], groups=groups, h=o["h"], w=o["w"]
            )
            ih, iw, ic = o["h"], o["w"], cin
        elif o["op"] == "gemm":
            write_gemm(p, b, k=o["k"], n=o["n"])
            ih, iw, ic = 1, o["k"], 1
        elif o["op"] == "add":
            write_add(p, b, cin=o["cin"], h=o["h"], w=o["w"])
            ih, iw, ic = o["h"], o["w"], o["cin"]
        else:
            write_gap(p, b, cin=o["cin"], h=o["h"], w=o["w"])
            ih, iw, ic = o["h"], o["w"], o["cin"]
        meta.append({**o, "name": name, "ih": ih, "iw": iw, "ic": ic})
    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def main() -> None:
    if not LOCAL_BENCH.exists():
        raise SystemExit(f"missing {LOCAL_BENCH}")

    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "pack"
        work.mkdir()
        meta = pack_ops(work)
        tar_path = Path(td) / "pack.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(work, arcname="pack")

        run(f"mkdir -p {REMOTE_ROOT}/bin", timeout=30)
        put(str(LOCAL_BENCH), REMOTE_BENCH)
        run(f"chmod +x {REMOTE_BENCH}", timeout=30)
        put(str(tar_path), f"{REMOTE_ROOT}/lut_ops_smoke.tar")

    mine_local = Path(__file__).with_name("_remote_mine_ops_smoke.py")
    mine_local.write_text(
        """#!/usr/bin/env python3
import json, subprocess
from pathlib import Path
work = Path("/root/fer-dynedge/lut_ops_smoke")
bench = "/root/fer-dynedge/bin/fer_op_bench"
meta = json.loads((work / "meta.json").read_text())
rows = []
for i, o in enumerate(meta):
    name = o["name"]
    cmd = [
        bench, "--device", "t527",
        "--param", str(work / f"{name}.param"),
        "--bin", str(work / f"{name}.bin"),
        "--h", str(o["ih"]), "--w", str(o["iw"]), "--c", str(o["ic"]),
        "--threads", "4", "--warmup", "20", "--iters", "200",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    text = r.stdout + r.stderr
    lat = None
    for line in text.splitlines():
        if "latency_us_p50=" in line:
            for part in line.split():
                if part.startswith("latency_us_p50="):
                    lat = float(part.split("=", 1)[1])
    ok = 1 if lat is not None else 0
    rows.append({**o, "p50_us": lat, "ok": ok})
    print(f"[{i+1}/{len(meta)}] {name} p50={lat} ok={ok}", flush=True)
(work / "results.json").write_text(json.dumps(rows, indent=2))
print("DONE", sum(r["ok"] for r in rows), "/", len(rows))
""",
        encoding="utf-8",
        newline="\n",
    )
    put(str(mine_local), f"{REMOTE_ROOT}/_remote_mine_ops_smoke.py")
    # BusyBox tar: avoid long-opt quirks; extract then flatten one level.
    code, out, err = run(
        f"rm -rf {REMOTE_WORK} {REMOTE_WORK}_raw; mkdir -p {REMOTE_WORK}_raw {REMOTE_WORK}; "
        f"tar -xf {REMOTE_ROOT}/lut_ops_smoke.tar -C {REMOTE_WORK}_raw; "
        f"cp -a {REMOTE_WORK}_raw/pack/. {REMOTE_WORK}/; "
        f"ls -la {REMOTE_WORK}; "
        f"python3 {REMOTE_ROOT}/_remote_mine_ops_smoke.py",
        timeout=900,
    )
    text = out + err
    OUT_LOG.write_text(text, encoding="utf-8")
    if code != 0:
        raise SystemExit(f"remote failed code={code}\n{text[-3000:]}")

    # fetch results
    code2, out2, err2 = run(f"cat {REMOTE_WORK}/results.json", timeout=60)
    rows = json.loads(out2)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "device",
            "backend",
            "name",
            "op",
            "dtype",
            "cin",
            "cout",
            "k",
            "s",
            "h",
            "w",
            "n",
            "threads",
            "latency_us_p50",
            "ok",
            "note",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "device": "t527",
                    "backend": "ncnn_cpu",
                    "name": r["name"],
                    "op": r["op"],
                    "dtype": "fp32",
                    "cin": r.get("cin", ""),
                    "cout": r.get("cout", ""),
                    "k": r.get("k", ""),
                    "s": r.get("s", ""),
                    "h": r.get("h", r.get("ih", "")),
                    "w": r.get("w", r.get("iw", "")),
                    "n": r.get("n", ""),
                    "threads": 4,
                    "latency_us_p50": r.get("p50_us", ""),
                    "ok": r.get("ok", 0),
                    "note": "MNV3-representative smoke; A55@1416MHz",
                }
            )
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"OK {ok}/{len(rows)} -> {OUT_CSV}")


if __name__ == "__main__":
    main()
