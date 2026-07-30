#!/usr/bin/env python3
"""Pack first N gray→RGB uint8 64x64 images as C arrays for board logits check."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref",
        type=Path,
        default=ROOT / "deploy/mcu_espdl/models/d2222_ferplus_64_esp32p4_host_ref.json",
    )
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    ref = json.loads(args.ref.read_text(encoding="utf-8"))
    n = min(args.n, len(ref["samples"]))
    out_c = ROOT / "deploy/mcu_espdl/firmware/main/ref_images.c"
    out_h = ROOT / "deploy/mcu_espdl/firmware/main/ref_images.h"
    blobs = []
    meta = []
    for r in ref["samples"][:n]:
        path = ROOT / r["path"]
        img = Image.open(path).convert("L").convert("RGB").resize((64, 64), Image.BILINEAR)
        arr = bytes(img.tobytes())  # HWC RGB888
        blobs.append(arr)
        meta.append(
            {
                "index": r["index"],
                "int8_pred": r["int8_pred"],
                "int8_logits": r["int8_logits"],
                "path": r["path"],
            }
        )

    with out_h.open("w", encoding="utf-8", newline="\n") as f:
        f.write("#pragma once\n#include <stddef.h>\n#include <stdint.h>\n")
        f.write(f"#define REF_IMG_N {n}\n")
        f.write("#define REF_IMG_H 64\n#define REF_IMG_W 64\n#define REF_IMG_C 3\n")
        f.write("extern const uint8_t ref_images[REF_IMG_N][REF_IMG_H * REF_IMG_W * REF_IMG_C];\n")
        f.write("extern const int ref_host_pred[REF_IMG_N];\n")

    with out_c.open("w", encoding="utf-8", newline="\n") as f:
        f.write('#include "ref_images.h"\n')
        f.write("const uint8_t ref_images[REF_IMG_N][REF_IMG_H * REF_IMG_W * REF_IMG_C] = {\n")
        for bi, blob in enumerate(blobs):
            f.write(f"  {{ /* sample {bi} */\n")
            for i in range(0, len(blob), 16):
                chunk = blob[i : i + 16]
                f.write("    " + ", ".join(str(b) for b in chunk) + ",\n")
            f.write("  },\n")
        f.write("};\n")
        preds = ", ".join(str(m["int8_pred"]) for m in meta)
        f.write(f"const int ref_host_pred[REF_IMG_N] = {{{preds}}};\n")

    (ROOT / "deploy/mcu_espdl/assets/ref_images_meta.json").write_text(
        json.dumps({"n": n, "ref": str(args.ref), "samples": meta}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_c} ({n} images from {args.ref.name})")


if __name__ == "__main__":
    main()
