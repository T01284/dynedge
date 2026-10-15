#!/usr/bin/env python3
"""Push the stage-B code, feeds, checkpoint and raw FER+ CSVs to the Orin.

Large files are skipped when the remote copy already has the same size.
"""
from __future__ import annotations

import argparse
import posixpath
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/orin/scripts"))
from ssh_orin import DEFAULT_HOST, connect  # noqa: E402
from scp import SCPClient  # noqa: E402

REMOTE = "<gpu-workdir>/fer-dynedge"

CODE = [
    "models/__init__.py",
    "models/elastic_mbv3.py",
    "models/slimmable_bn.py",
    "models/latency_controller.py",
    "datasets/__init__.py",
    "datasets/common.py",
    "datasets/prepare_ferplus.py",
    "train/search_subnets_lut.py",
    "train/search_subnets_flops_ctrl.py",
    "scripts/stageB/b10_search.py",
    "scripts/stageB/orin_trt_sweep.py",
    "scripts/stageB/orin_b10_o1.sh",
    "scripts/stageB/orin_b10_o4.sh",
    "train/train_controller.py",
    "lut/topo_ops.py",
    "lut/ops_t527_v4_full.csv",
    "lut/ops_h618_v4_full.csv",
    "results/h618_v4/exit_segment_latency/summary.json",
    "results/t527_v4/exit_segment_latency/summary.json",
    "scripts/stageB/b1_cache_logits.py",
    "scripts/stageB/orin_pipeline.sh",
    "scripts/stageB/orin_launch.sh",
    "scripts/stageB/orin_status.sh",
    "scripts/stageB/orin_expw.sh",
    "scripts/stageB/orin_expw_faithful.sh",
    "scripts/stageB/b0_expw_consistency.py",
    "scripts/stageB/orin_fam2.sh",
    "scripts/stageB/orin_fam2_multi.py",
    "scripts/stageB/orin_fam2_multi.sh",
    "scripts/stageB/orin_b10_multi.sh",
    "scripts/stageB/orin_b10_o4_multi.sh",
    "scripts/stageB/orin_b10_o4b.sh",
    "scripts/stageB/orin_b10_retest.sh",
    "results/stageB/orin_sweep/o3m_cost_table_int8.json",
    "results/stageB/fam2/orin_order.json",
    "results/from_autodl/formal_v4/search_lut/search_expw_h618.json",
    "results/from_autodl/formal_v4/search_lut/search_expw_t527.json",
]
BIG = {
    "results/from_autodl/supernet/supernet_ferplus_seed42_v4_best.pt": "results/supernet/supernet_ferplus_seed42_v4_best.pt",
    "datasets/fer2013/fer2013.csv": "raw/fer2013.csv",
    "datasets/FERPlus/fer2013new.csv": "raw/fer2013new.csv",
    "results/from_autodl/supernet/supernet_expw_seed42_v4_best.pt": "results/supernet/supernet_expw_seed42_v4_best.pt",
    "data/expw_rebuild.tar": "raw/expw_rebuild.tar",
    "data/expw_faithful.tar": "raw/expw_faithful.tar",
    "data/orin_sweep_onnx.tar": "raw/orin_sweep_onnx.tar",
    "data/orin_space_onnx.tar": "raw/orin_space_onnx.tar",
    "data/fam2_onnx.tar": "raw/fam2_onnx.tar",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--code-only", action="store_true")
    ap.add_argument("files", nargs="*", help="extra repo-relative files to push")
    args = ap.parse_args()

    pairs = [(f, f) for f in CODE + args.files]
    if not args.code_only:
        pairs += list(BIG.items())
    c = connect(args.host)
    try:
        dirs = sorted({posixpath.dirname(posixpath.join(REMOTE, r)) for _, r in pairs})
        c.exec_command("mkdir -p " + " ".join(dirs))[1].channel.recv_exit_status()
        sftp = c.open_sftp()
        with SCPClient(c.get_transport()) as scp:
            for local, remote in pairs:
                lp = ROOT / local
                rp = posixpath.join(REMOTE, remote)
                if not lp.is_file():
                    print(f"missing locally, skip {local}")
                    continue
                if local in BIG:
                    try:
                        if sftp.stat(rp).st_size == lp.stat().st_size:
                            print(f"same size, skip {remote}")
                            continue
                    except FileNotFoundError:
                        pass
                scp.put(str(lp), rp)
                print(f"PUT {local} -> {remote} ({lp.stat().st_size / 1e6:.1f} MB)")
        sftp.close()
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
