#!/usr/bin/env python3
"""Dump float32 HWC inputs for the first N host_ref samples (board replay)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main() -> None:
    ref_path = ROOT / "deploy/mcu_espdl/models/d2222_ferplus_64_esp32p4_host_ref.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    outdir = ROOT / "deploy/mcu_espdl/assets/ref_inputs"
    outdir.mkdir(parents=True, exist_ok=True)
    meta = []
    for r in ref["samples"][:16]:
        img = Image.open(r["path"]).convert("L").convert("RGB").resize((64, 64), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        name = f"sample_{r['index']:02d}.f32"
        arr.astype(np.float32).tofile(outdir / name)
        meta.append(
            {
                "index": r["index"],
                "path": r["path"],
                "int8_pred": r["int8_pred"],
                "int8_logits": r["int8_logits"],
                "bin": name,
            }
        )
    (outdir / "manifest.json").write_text(
        json.dumps(
            {"img_size": 64, "n": len(meta), "layout": "HWC_float32", "samples": meta},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(meta)} bins -> {outdir}")


if __name__ == "__main__":
    main()
