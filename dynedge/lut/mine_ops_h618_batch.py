#!/usr/bin/env python3
"""Fast H618 LUT miner: pack all ops, one SSH upload, remote bash loop."""
from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
import tarfile
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lut"))
from topo_ops import op_key  # noqa: E402

HOST, PASS = "BOARD_IP_REDACTED", "1234"
REMOTE_ROOT = "/root/fer-dynedge"
REMOTE_WORK = f"{REMOTE_ROOT}/lut_ops_full"
REMOTE_BENCH = f"{REMOTE_ROOT}/deploy/ncnn_edge/build-opbench/fer_op_bench"


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
                f"InnerProduct fc0 1 1 input out 0={n} 1=1 2={k*n}",
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enum", type=Path, default=ROOT / "lut/enum_v4_full.json")
    ap.add_argument("--out", type=Path, default=ROOT / "lut/ops_h618_v4_full.csv")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "int8", "both"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

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

    work = ROOT / "lut" / "_ops_full_pack"
    if work.exists():
        import shutil

        shutil.rmtree(work)
    work.mkdir(parents=True)
    meta = []
    for i, o in enumerate(ops):
        name = f"{i:04d}_{o['op']}_{o['dtype']}"
        p = work / f"{name}.param"
        b = work / f"{name}.bin"
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
        meta.append({**o, "name": name, "ih": ih, "iw": iw, "ic": ic})

    (work / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    tar_path = ROOT / "lut" / "_ops_full_pack.tgz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(work, arcname="pack")
    print(f"packed {tar_path} ({tar_path.stat().st_size/1e6:.1f}MB)", flush=True)

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="root", password=<REDACTED> timeout=30, allow_agent=False, look_for_keys=False)
    sftp = c.open_sftp()
    rem_tar = f"{REMOTE_ROOT}/lut_ops_full_pack.tgz"
    print("upload tar...", flush=True)
    sftp.put(str(tar_path), rem_tar)
    sftp.close()

    remote_sh = f"""
set -e
export PATH=/opt/ncnn/install/bin:$PATH
BENCH={REMOTE_BENCH}
WORK={REMOTE_WORK}
rm -rf "$WORK"
mkdir -p "$WORK"
tar -xzf {rem_tar} -C "$WORK" --strip-components=1
OUT="$WORK/results.csv"
echo 'name,p50_us,ok,note' > "$OUT"
python3 - <<'PY'
import json, subprocess, os
from pathlib import Path
work = Path("{REMOTE_WORK}")
bench = "{REMOTE_BENCH}"
meta = json.loads((work/"meta.json").read_text())
out = work/"results.csv"
# already has header
fp32_cache = {{}}  # shape key -> p50 for int8 fallback / int8 quant base
ok=fail=0
for i,o in enumerate(meta):
    name=o["name"]
    param=str(work/f"{{name}}.param")
    binp=str(work/f"{{name}}.bin")
    ih,iw,ic=int(o["ih"]),int(o["iw"]),int(o["ic"])
    dtype=o["dtype"]
    use_param, use_bin = param, binp
    note=""
    if dtype=="int8" and o["op"] in {{"conv","dwconv"}}:
        # quantize
        table=str(work/f"{{name}}.table")
        ip=str(work/f"{{name}}_int8.param")
        ib=str(work/f"{{name}}_int8.bin")
        calib_img=str(work/f"c_{{ih}}x{{iw}}.jpg")
        calib_txt=str(work/f"c_{{ih}}x{{iw}}.txt")
        if not Path(calib_img).exists():
            try:
                from PIL import Image
                import numpy as np
                arr=(np.random.rand(ih,iw,3)*255).astype("uint8")
                Image.fromarray(arr,"RGB").save(calib_img)
            except Exception:
                import glob,shutil
                xs=glob.glob("/root/fer-dynedge/data/calib/**/*.jpg", recursive=True)
                shutil.copy(xs[0], calib_img)
            Path(calib_txt).write_text(calib_img+"\\n")
        import subprocess as sp
        r=sp.run(["ncnn2table", param, binp, calib_txt, table,
                  f"mean=[0,0,0]", f"norm=[0.00392157,0.00392157,0.00392157]",
                  f"shape=[{{ih}},{{iw}},{{ic}}]", "pixel=RGB", "thread=2", "method=kl"],
                 capture_output=True, text=True)
        if r.returncode!=0 or not Path(table).exists() or Path(table).stat().st_size==0:
            note="table_fail"
        else:
            r2=sp.run(["ncnn2int8", param, binp, ip, ib, table], capture_output=True, text=True)
            if r2.returncode!=0 or not Path(ip).exists():
                note="int8_fail"
            else:
                use_param, use_bin = ip, ib
    elif dtype=="int8":
        note="fp32_proxy"  # gemm/add/gap: measure fp32 graph as proxy
    cmd=[bench,"--device","h618","--param",use_param,"--bin",use_bin,
         "--h",str(ih),"--w",str(iw),"--c",str(ic),"--threads","4","--warmup","10","--iters","80"]
    r=subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    lat=None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT") and "latency_us_p50=" in line:
            for part in line.split():
                if part.startswith("latency_us_p50="):
                    lat=float(part.split("=",1)[1])
    if lat is None:
        fail+=1
        with out.open("a") as f:
            f.write(f"{{name}},,0,{{note or 'bench_fail'}}\\n")
        print(f"[{{i+1}}/{{len(meta)}}] FAIL {{name}} {{note}}", flush=True)
    else:
        ok+=1
        with out.open("a") as f:
            f.write(f"{{name}},{{lat}},1,{{note}}\\n")
        print(f"[{{i+1}}/{{len(meta)}}] OK {{name}} p50={{lat}} {{note}}", flush=True)
print(f"DONE ok={{ok}} fail={{fail}}")
PY
"""
    print("remote mine starting...", flush=True)
    # run in background on board with nohup so we can poll
    stdin, stdout, stderr = c.exec_command(
        f"nohup bash -lc {json.dumps(remote_sh)} > {REMOTE_WORK}/mine.log 2>&1 & echo $!"
    )
    # The above quoting is fragile. Write script via sftp instead.
    c.close()

    # rewrite: put mine.sh and run
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="root", password=<REDACTED> timeout=30, allow_agent=False, look_for_keys=False)
    sftp = c.open_sftp()
    # extract first
    stdin, stdout, stderr = c.exec_command(
        f"rm -rf {REMOTE_WORK}; mkdir -p {REMOTE_WORK}; tar -xzf {rem_tar} -C {REMOTE_WORK} --strip-components=1; ls {REMOTE_WORK} | wc -l"
    )
    print(stdout.read().decode(), stderr.read().decode())

    mine_py = f"""#!/usr/bin/env python3
import json, subprocess, glob, shutil
from pathlib import Path
work = Path("{REMOTE_WORK}")
bench = "{REMOTE_BENCH}"
meta = json.loads((work/"meta.json").read_text())
out = work/"results.csv"
out.write_text("name,p50_us,ok,note\\n")
ok=fail=0
for i,o in enumerate(meta):
    name=o["name"]
    param=str(work/f"{{name}}.param")
    binp=str(work/f"{{name}}.bin")
    ih,iw,ic=int(o["ih"]),int(o["iw"]),int(o["ic"])
    dtype=o["dtype"]
    use_param, use_bin = param, binp
    note=""
    if dtype=="int8" and o["op"] in ("conv","dwconv"):
        table=str(work/f"{{name}}.table")
        ip=str(work/f"{{name}}_int8.param")
        ib=str(work/f"{{name}}_int8.bin")
        calib_img=str(work/f"c_{{ih}}x{{iw}}.jpg")
        calib_txt=str(work/f"c_{{ih}}x{{iw}}.txt")
        if not Path(calib_img).exists():
            try:
                from PIL import Image
                import numpy as np
                arr=(np.random.rand(ih,iw,3)*255).astype("uint8")
                Image.fromarray(arr,"RGB").save(calib_img)
            except Exception:
                xs=glob.glob("/root/fer-dynedge/data/calib/**/*.jpg", recursive=True)
                shutil.copy(xs[0], calib_img)
            Path(calib_txt).write_text(calib_img+"\\n")
        r=subprocess.run(["ncnn2table", param, binp, calib_txt, table,
            "mean=[0,0,0]", "norm=[0.00392157,0.00392157,0.00392157]",
            f"shape=[{{ih}},{{iw}},{{ic}}]", "pixel=RGB", "thread=2", "method=kl"],
            capture_output=True, text=True)
        if r.returncode!=0 or not Path(table).exists() or Path(table).stat().st_size==0:
            note="table_fail_fp32proxy"
        else:
            r2=subprocess.run(["ncnn2int8", param, binp, ip, ib, table], capture_output=True, text=True)
            if r2.returncode!=0 or not Path(ip).exists():
                note="int8_fail_fp32proxy"
            else:
                use_param, use_bin = ip, ib
    elif dtype=="int8":
        note="fp32_proxy"
    try:
        r=subprocess.run([bench,"--device","h618","--param",use_param,"--bin",use_bin,
            "--h",str(ih),"--w",str(iw),"--c",str(ic),"--threads","4","--warmup","10","--iters","80"],
            capture_output=True, text=True, timeout=120)
    except Exception as e:
        fail+=1
        with out.open("a") as f: f.write(f"{{name}},,0,exc\\n")
        print(f"[{{i+1}}/{{len(meta)}}] EXC {{name}} {{e}}", flush=True)
        continue
    lat=None
    for line in (r.stdout or "").splitlines():
        if line.startswith("RESULT") and "latency_us_p50=" in line:
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
    with sftp.file(f"{REMOTE_WORK}/mine_all.py", "w") as f:
        f.write(mine_py)
    sftp.chmod(f"{REMOTE_WORK}/mine_all.py", 0o755)
    sftp.close()

    # launch detached
    stdin, stdout, stderr = c.exec_command(
        f"nohup python3 {REMOTE_WORK}/mine_all.py > {REMOTE_WORK}/mine.log 2>&1 & echo $!"
    )
    pid = stdout.read().decode().strip()
    print(f"started pid={pid}", flush=True)
    (ROOT / "lut" / "_h618_mine_pid.txt").write_text(pid + "\n" + REMOTE_WORK, encoding="utf-8")
    # save local meta for merge
    (ROOT / "lut" / "_h618_mine_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    c.close()
    print("poll with: python lut/pull_h618_ops_mine.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
