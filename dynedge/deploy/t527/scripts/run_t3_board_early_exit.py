#!/usr/bin/env python3
"""Push T3 early-exit demo to T527 and run (threshold + force-final baseline).

Formal thread protocol: --threads 4 (same as v4 latency matrix).
"""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

from ssh_t527 import put, run

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/t527_v4/t3_early_exit"
THREADS = 4


def main() -> int:
    bins = sorted((ROOT / "results/t527_v4/eval500_bins/hw112").glob("*.bin"))[:80]
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "bins.tar"
        with tarfile.open(tar_path, "w") as tar:
            for p in bins:
                tar.add(p, arcname=p.name)
        print("tar bytes", tar_path.stat().st_size)
        run("rm -rf /root/t3_ee; mkdir -p /root/t3_ee/bins /root/t3_ee/models")
        put(
            str(ROOT / "deploy/ncnn_edge/build-linux-aarch64-acc/fer_early_exit"),
            "/root/t3_ee/fer_early_exit",
        )
        put(str(OUT / "ncnn/m.ncnn.param"), "/root/t3_ee/models/m.ncnn.param")
        put(str(OUT / "ncnn/m.ncnn.bin"), "/root/t3_ee/models/m.ncnn.bin")
        put(str(tar_path), "/root/t3_ee/bins.tar")

    code, out, err = run(
        "cd /root/t3_ee && tar xf bins.tar -C bins && ls bins | wc -l",
        timeout=120,
    )
    print(out, err, "extract_code", code)
    if code != 0:
        return code

    prep = (
        "cd /root/t3_ee && for f in bins/*; do echo /root/t3_ee/$f; done > list80.txt && "
        "wc -l list80.txt && chmod +x fer_early_exit && "
    )
    common = (
        "./fer_early_exit --param models/m.ncnn.param --bin models/m.ncnn.bin "
        f"--list list80.txt --h 112 --w 112 --c 3 --tau 0.85 --threads {THREADS} "
        "--out0 out0 --out1 out1 --out2 out2 --out3 out3"
    )

    code, out_ff, err_ff = run(prep + common + " --force-final", timeout=300)
    print("FORCE_FINAL", out_ff)
    if err_ff:
        print(err_ff[-800:])
    (OUT / "t3_board_force_final.txt").write_text(
        (out_ff or "") + "\n" + (err_ff or ""), encoding="utf-8"
    )
    if code != 0:
        return code

    code, out_ee, err_ee = run(prep + common, timeout=300)
    print("EARLY_EXIT", out_ee)
    if err_ee:
        print(err_ee[-800:])
    (OUT / "t3_board_result.txt").write_text(
        (out_ee or "") + "\n" + (err_ee or ""), encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
