#!/usr/bin/env python3
"""Export T3 multi-exit bake to standard ONNX (opset17) for Orin TensorRT.

pnnx ONNX uses nn.Linear/prim::* and is NOT TRT-importable.
This script rebuilds BakedMultiExit and torch.onnx.exports 4 outputs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402
from tools.export_subnet_onnx import remap_v1_stem_keys  # noqa: E402

# reuse BakedMultiExit from t3 script
_spec = importlib.util.spec_from_file_location(
    "t3ee", ROOT / "deploy/t527/scripts/t3_early_exit_threshold_demo.py"
)
_t3 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_t3)

CKPT = (
    ROOT
    / "results/from_autodl/subnets_v4_predistill"
    / "subnet_ferplus_w0.5_d3333_s112c_seed42_v4_predistill_best.pt"
)
OUT = ROOT / "deploy/orin/models/multi_exit"
ONNX = OUT / "ferplus_w05_d3333_s112c_multiexit.onnx"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = remap_v1_stem_keys(ckpt["model"])
    num_classes = int(state["final_fc.weight"].shape[0])
    slim = ElasticFERSuperNet(num_classes=num_classes, arch="v3_wide", slimmable=True)
    slim.load_state_dict(state, strict=True)
    slim.eval()
    cfg = SubnetConfig(width=0.5, depths=(3, 3, 3, 3), img_size=112, gray=False)
    model = _t3.BakedMultiExit(slim, cfg).eval()

    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        outs = model(dummy)
    assert len(outs) == 4
    print("shapes", [tuple(o.shape) for o in outs])

    torch.onnx.export(
        model,
        dummy,
        str(ONNX),
        input_names=["input"],
        output_names=["out0", "out1", "out2", "out3"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    meta = {
        "ckpt": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
        "onnx": str(ONNX.relative_to(ROOT)).replace("\\", "/"),
        "width": 0.5,
        "depths": [3, 3, 3, 3],
        "img": 112,
        "outputs": ["out0", "out1", "out2", "out3"],
        "note": "standard ONNX for Orin TRT; same topology as T3 multi-exit bake",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", ONNX, "bytes", ONNX.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
