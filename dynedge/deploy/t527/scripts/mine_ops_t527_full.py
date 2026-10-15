#!/usr/bin/env python3
"""Full T527 LUT mine for h618_ncnn-shaped ops (A55 vs A53 §4.5).

Packs graphs on host, uploads uncompressed tar (BusyBox-safe), runs fer_op_bench
on board. INT8: host-side ncnn2table/ncnn2int8 when available; else fp32_proxy.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lut"))
sys.path.insert(0, str(ROOT / "deploy" / "t527" / "scripts"))
from topo_ops import op_key  # noqa: E402
from ssh_t527 import put, run  # noqa: E402

LOCAL_BENCH = ROOT / "deploy" / "ncnn_edge" / "build-linux-aarch64" / "fer_op_bench"
REMOTE_ROOT = "/root/fer-dynedge"
REMOTE_BENCH = f"{REMOTE_ROOT}/bin/fer_op_bench"
REMOTE_WORK = f"{REMOTE_ROOT}/lut_ops_full"
OUT_CSV = ROOT / "lut" / "ops_t527_v4_full.csv"
OUT_LOG = ROOT / "results" / "t527_linux_lut_mine.log"
PID_FILE = ROOT / "lut" / "_t527_mine_pid.txt"
META_FILE = ROOT / "lut" / "_t527_mine_meta.json"


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
    # BinaryOp Add with duplicated input via MemoryData constant 0 + skip?
    # Use two Inputs of same shape fed identically by bench — not supported.
    # Prefer in-place friendly: Convolution 1x1 identity + Eltwise needs 2 blobs.
    # H618 used Split+Eltwise; fer_op_bench single input fails on some ncnn builds.
    # Workaround: measure BinaryOp via Bias-like noop — use Eltwise with MemoryData zero.
    param.write_text(
        "\n".join(
            [
                "7767517",
                "3 3",
                f"Input input 0 1 input 0={w} 1={h} 2={cin}",
                f"MemoryData zeros 0 1 zeros 0={w} 1={h} 2={cin}",
                "Eltwise add0 2 1 input zeros out 0=1",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    # MemoryData stores raw floats in bin (no convolution flag header).
    n = cin * h * w
    with bin_path.open("wb") as f:
        for _ in range(n):
            f.write(struct.pack("<f", 0.0))


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


def _tool_cmd(name: str) -> list[str] | None:
    native = shutil.which(name)
    if native:
        return [native]
    wsl_bin = f"<home>/opt/ncnn-x86-tools/bin/{name}"
    # Prefer WSL tools when running from Windows host.
    try:
        r = subprocess.run(
            ["wsl", "-e", "test", "-x", wsl_bin],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return ["wsl", "-e", wsl_bin]
    except OSError:
        pass
    return None


def _to_wsl_path(p: Path) -> str:
    s = str(p.resolve())
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return s


def try_quantize_int8(
    param: Path, bin_path: Path, ih: int, iw: int, ic: int, work: Path, name: str
) -> tuple[Path, Path, str]:
    t_table = _tool_cmd("ncnn2table")
    t_int8 = _tool_cmd("ncnn2int8")
    if not t_table or not t_int8:
        return param, bin_path, "fp32_proxy_no_tools"
    calib_img = work / f"c_{ih}x{iw}.jpg"
    calib_txt = work / f"c_{ih}x{iw}.txt"
    if not calib_img.exists():
        from PIL import Image
        import numpy as np

        arr = (np.random.rand(ih, iw, 3) * 255).astype("uint8")
        Image.fromarray(arr, "RGB").save(calib_img)
        # Paths inside calib list must be visible to the tool process.
        if t_table[0] == "wsl":
            calib_txt.write_text(_to_wsl_path(calib_img) + "\n", encoding="utf-8")
        else:
            calib_txt.write_text(str(calib_img) + "\n", encoding="utf-8")
    table = work / f"{name}.table"
    ip = work / f"{name}_int8.param"
    ib = work / f"{name}_int8.bin"

    def args_for(tool: list[str], argv: list[str]) -> list[str]:
        if tool[0] == "wsl":
            return tool + [_to_wsl_path(Path(a)) if ("\\" in a or (len(a) > 1 and a[1] == ":")) else a for a in argv]
        return tool + argv

    # Rebuild argv carefully: only convert Path-like file args
    table_argv = [
        str(param),
        str(bin_path),
        str(calib_txt),
        str(table),
        "mean=[0,0,0]",
        "norm=[0.00392157,0.00392157,0.00392157]",
        f"shape=[{ih},{iw},{ic}]",
        "pixel=RGB",
        "thread=2",
        "method=kl",
    ]
    if t_table[0] == "wsl":
        table_cmd = t_table + [
            _to_wsl_path(param),
            _to_wsl_path(bin_path),
            _to_wsl_path(calib_txt),
            _to_wsl_path(table),
            "mean=[0,0,0]",
            "norm=[0.00392157,0.00392157,0.00392157]",
            f"shape=[{ih},{iw},{ic}]",
            "pixel=RGB",
            "thread=2",
            "method=kl",
        ]
    else:
        table_cmd = t_table + table_argv
    r = subprocess.run(table_cmd, capture_output=True, text=True)
    if r.returncode != 0 or not table.exists() or table.stat().st_size == 0:
        return param, bin_path, "table_fail_fp32proxy"
    if t_int8[0] == "wsl":
        int8_cmd = t_int8 + [
            _to_wsl_path(param),
            _to_wsl_path(bin_path),
            _to_wsl_path(ip),
            _to_wsl_path(ib),
            _to_wsl_path(table),
        ]
    else:
        int8_cmd = t_int8 + [str(param), str(bin_path), str(ip), str(ib), str(table)]
    r2 = subprocess.run(int8_cmd, capture_output=True, text=True)
    if r2.returncode != 0 or not ip.exists():
        return param, bin_path, "int8_fail_fp32proxy"
    return ip, ib, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum", type=Path, default=ROOT / "lut" / "enum_v4_full.json")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--dtype", default="both", choices=["fp32", "int8", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--detach", action="store_true", help="nohup on board and return")
    args = ap.parse_args()

    if not LOCAL_BENCH.exists():
        raise SystemExit(f"missing {LOCAL_BENCH}")

    dtypes = ["fp32", "int8"] if args.dtype == "both" else [args.dtype]
    enum = json.loads(args.enum.read_text(encoding="utf-8"))
    ops = []
    seen = set()
    for o in enum["ops"]:
        if o.get("backend") != "h618_ncnn":
            continue
        if o.get("dtype") not in dtypes:
            continue
        if o.get("op") not in {"conv", "dwconv", "gemm", "add", "gap"}:
            continue
        k = op_key(o)
        if k in seen:
            continue
        seen.add(k)
        ops.append(o)
    if args.limit:
        ops = ops[: args.limit]
    print(f"ops={len(ops)} dtypes={dtypes}", flush=True)

    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "pack"
        work.mkdir()
        meta = []
        for i, o in enumerate(ops):
            name = f"{i:04d}_{o['op']}_{o['dtype']}"
            p = work / f"{name}.param"
            b = work / f"{name}.bin"
            note = ""
            if o["op"] in {"conv", "dwconv"}:
                cin, cout = int(o["cin"]), int(o["cout"])
                kk, s = int(o["k"]), int(o["s"])
                groups = int(o.get("groups") or (cin if o["op"] == "dwconv" else 1))
                h = int(o["h"])
                w = int(o.get("w") or h)
                write_conv(p, b, cin=cin, cout=cout, k=kk, s=s, groups=groups, h=h, w=w)
                ih, iw, ic = h, w, cin
            elif o["op"] == "gemm":
                kk, nn = int(o["k"]), int(o["n"])
                write_gemm(p, b, k=kk, n=nn)
                ih, iw, ic = 1, kk, 1
            elif o["op"] == "add":
                cin = int(o["cin"])
                h = int(o["h"])
                w = int(o.get("w") or h)
                write_add(p, b, cin=cin, h=h, w=w)
                ih, iw, ic = h, w, cin
            else:
                cin = int(o["cin"])
                h = int(o["h"])
                w = int(o.get("w") or h)
                write_gap(p, b, cin=cin, h=h, w=w)
                ih, iw, ic = h, w, cin

            use_p, use_b = p, b
            if o["dtype"] == "int8" and o["op"] in {"conv", "dwconv"}:
                use_p, use_b, note = try_quantize_int8(p, b, ih, iw, ic, work, name)
                if use_p != p:
                    # rename to canonical names for remote loop
                    (work / f"{name}.param").write_bytes(use_p.read_bytes())
                    (work / f"{name}.bin").write_bytes(use_b.read_bytes())
            elif o["dtype"] == "int8":
                note = "fp32_proxy"

            meta.append({**o, "name": name, "ih": ih, "iw": iw, "ic": ic, "note": note})
            if (i + 1) % 50 == 0:
                print(f"packed {i+1}/{len(ops)}", flush=True)

        (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        tar_path = Path(td) / "pack.tar"
        with tarfile.open(tar_path, "w") as tar:
            tar.add(work, arcname="pack")
        print(f"tar {tar_path.stat().st_size/1e6:.1f}MB", flush=True)

        run(f"mkdir -p {REMOTE_ROOT}/bin", timeout=30)
        put(str(LOCAL_BENCH), REMOTE_BENCH)
        run(f"chmod +x {REMOTE_BENCH}", timeout=30)
        print("upload pack...", flush=True)
        put(str(tar_path), f"{REMOTE_ROOT}/lut_ops_full.tar")

    mine_py = f"""#!/usr/bin/env python3
import json, subprocess
from pathlib import Path
work = Path("{REMOTE_WORK}")
bench = "{REMOTE_BENCH}"
meta = json.loads((work/"meta.json").read_text())
out = work/"results.csv"
out.write_text("name,p50_us,ok,note\\n")
ok=fail=0
for i,o in enumerate(meta):
    name=o["name"]
    note=o.get("note") or ""
    cmd=[bench,"--device","t527","--param",str(work/f"{{name}}.param"),"--bin",str(work/f"{{name}}.bin"),
         "--h",str(o["ih"]),"--w",str(o["iw"]),"--c",str(o["ic"]),
         "--threads","4","--warmup","10","--iters","80"]
    try:
        r=subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        fail+=1
        with out.open("a") as f: f.write(f"{{name}},,0,exc\\n")
        print(f"[{{i+1}}/{{len(meta)}}] EXC {{name}} {{e}}", flush=True)
        continue
    lat=None
    for line in (r.stdout or "").splitlines():
        if "latency_us_p50=" in line:
            for part in line.split():
                if part.startswith("latency_us_p50="):
                    lat=float(part.split("=",1)[1])
    if lat is None:
        fail+=1
        with out.open("a") as f: f.write(f"{{name}},,0,{{note or 'bench_fail'}}\\n")
        print(f"[{{i+1}}/{{len(meta)}}] FAIL {{name}} {{note}}", flush=True)
    else:
        ok+=1
        with out.open("a") as f: f.write(f"{{name}},{{lat}},1,{{note}}\\n")
        print(f"[{{i+1}}/{{len(meta)}}] OK {{name}} p50={{lat}} {{note}}", flush=True)
print(f"DONE ok={{ok}} fail={{fail}}", flush=True)
"""
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", newline="\n") as f:
        f.write(mine_py)
        mine_local = f.name
    put(mine_local, f"{REMOTE_ROOT}/mine_all.py")

    run(
        f"rm -rf {REMOTE_WORK} {REMOTE_WORK}_raw; mkdir -p {REMOTE_WORK}_raw {REMOTE_WORK}; "
        f"tar -xf {REMOTE_ROOT}/lut_ops_full.tar -C {REMOTE_WORK}_raw; "
        f"cp -a {REMOTE_WORK}_raw/pack/. {REMOTE_WORK}/; "
        f"ls {REMOTE_WORK} | wc -l",
        timeout=120,
    )
    META_FILE.write_text(json.dumps(meta), encoding="utf-8")

    if args.detach:
        code, out, err = run(
            f"nohup python3 {REMOTE_ROOT}/mine_all.py > {REMOTE_WORK}/mine.log 2>&1 & echo $!",
            timeout=30,
        )
        pid = (out + err).strip().splitlines()[-1]
        PID_FILE.write_text(pid + "\n" + REMOTE_WORK + "\n", encoding="utf-8")
        print(f"started pid={pid}; poll: python deploy/t527/scripts/pull_t527_ops_mine.py")
        return 0

    # foreground
    code, out, err = run(f"python3 {REMOTE_ROOT}/mine_all.py", timeout=7200)
    OUT_LOG.write_text(out + err, encoding="utf-8")
    print(out[-2000:] if len(out) > 2000 else out)
    # pull results
    code2, csv_txt, err2 = run(f"cat {REMOTE_WORK}/results.csv", timeout=60)
    merge_results(csv_txt, meta, args.out)
    return 0 if code == 0 else code


def merge_results(csv_txt: str, meta: list, out_path: Path) -> None:
    by_name = {}
    for line in csv_txt.splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        by_name[parts[0]] = parts
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
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
            "groups",
            "width",
            "img_size",
            "threads",
            "latency_us_p50",
            "ok",
            "note",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for o in meta:
            parts = by_name.get(o["name"])
            lat = parts[1] if parts and len(parts) > 1 else ""
            ok = parts[2] if parts and len(parts) > 2 else "0"
            note = parts[3] if parts and len(parts) > 3 else o.get("note", "")
            w.writerow(
                {
                    "device": "t527",
                    "backend": "ncnn_cpu",
                    "name": o["name"],
                    "op": o["op"],
                    "dtype": o["dtype"],
                    "cin": o.get("cin", ""),
                    "cout": o.get("cout", ""),
                    "k": o.get("k", ""),
                    "s": o.get("s", ""),
                    "h": o.get("h", ""),
                    "w": o.get("w", o.get("h", "")),
                    "n": o.get("n", ""),
                    "groups": o.get("groups", ""),
                    "width": o.get("width", ""),
                    "img_size": o.get("img_size", ""),
                    "threads": 4,
                    "latency_us_p50": lat,
                    "ok": ok,
                    "note": note,
                }
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
