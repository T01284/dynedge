#!/usr/bin/env python3
"""Re-run host FP32 vs INT8 logits reference (assumes .espdl already built).

Re-quantizes lightly to recover PPQ graph in-memory, then dumps host_ref.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.espdl_ptq_d2222 import FolderImages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--calib-root", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--target", type=str, default="esp32p4")
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--gray-to-3ch", action="store_true")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", type=str, default="d2222_ferplus_64")
    ap.add_argument("--test-limit", type=int, default=32)
    args = ap.parse_args()

    from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_onnx
    from esp_ppq.executor.torch import TorchExecutor
    import onnxruntime as ort

    calib_ds = FolderImages(args.calib_root, args.img_size, args.gray_to_3ch, 128)
    calib_loader = DataLoader(calib_ds, batch_size=8, shuffle=False)

    def collate_fn(batch):
        if isinstance(batch, torch.Tensor):
            return batch
        return torch.stack(list(batch), dim=0)

    espdl_path = args.out_dir / f"{args.name}_{args.target}.espdl"
    graph = espdl_quantize_onnx(
        onnx_import_file=str(args.onnx),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=16,
        input_shape=[1, 3, args.img_size, args.img_size],
        target=args.target,
        num_of_bits=8,
        collate_fn=collate_fn,
        setting=QuantizationSettingFactory.espdl_setting(),
        device="cpu",
        error_report=False,
        skip_export=False,
        export_test_values=False,
        verbose=0,
    )

    test_ds = FolderImages(args.test_root, args.img_size, args.gray_to_3ch, args.test_limit)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    executor = TorchExecutor(graph=graph, device="cpu")

    rows = []
    for i in range(len(test_ds)):
        x = test_ds[i].unsqueeze(0)
        path = test_ds.path_at(i)
        fp32 = sess.run(None, {"input": x.numpy()})[0][0]
        q_out = executor.forward(inputs=[x])[0]
        if isinstance(q_out, torch.Tensor):
            q = q_out.detach().cpu().numpy().reshape(-1)
        else:
            q = np.asarray(q_out).reshape(-1)
        # also dump int8-ish rounded for board compare helper
        rows.append(
            {
                "index": i,
                "path": path,
                "fp32_logits": fp32.tolist(),
                "int8_logits": q.tolist(),
                "fp32_pred": int(fp32.argmax()),
                "int8_pred": int(q.argmax()),
                "cosine": float(
                    np.dot(fp32, q) / (np.linalg.norm(fp32) * np.linalg.norm(q) + 1e-12)
                ),
            }
        )

    match = sum(1 for r in rows if r["fp32_pred"] == r["int8_pred"])
    mean_cos = float(np.mean([r["cosine"] for r in rows]))
    ref = {
        "onnx": str(args.onnx),
        "espdl": str(espdl_path),
        "target": args.target,
        "img_size": args.img_size,
        "gray_to_3ch": args.gray_to_3ch,
        "n": len(rows),
        "pred_match": match,
        "pred_match_rate": match / max(1, len(rows)),
        "mean_cosine": mean_cos,
        "samples": rows,
    }
    out = args.out_dir / f"{args.name}_{args.target}_host_ref.json"
    out.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print(f"pred_match={match}/{len(rows)} mean_cosine={mean_cos:.4f} wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
