#!/usr/bin/env python3
"""PTQ a d2222 ONNX model with ESP-PPQ for ESP32-P4 / S3 @64 gray→3ch.

Outputs:
  *.espdl          — on-device model
  *_host_ref.json  — FP32 vs INT8 logits on fixed test images (for board compare)

Example:
  python tools/espdl_ptq_d2222.py \\
    --onnx deploy/mcu_espdl/models/d2222_ferplus_64.onnx \\
    --calib-root data/calib/ferplus_mcu64/train \\
    --test-root data/calib/ferplus_mcu64/test \\
    --target esp32p4 --img-size 64 --gray-to-3ch \\
    --out-dir deploy/mcu_espdl/models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb_tensor(path: Path, img_size: int, gray_to_3ch: bool) -> torch.Tensor:
    img = Image.open(path)
    if gray_to_3ch:
        img = img.convert("L").convert("RGB")
    else:
        img = img.convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))  # CHW


class FolderImages(Dataset):
    def __init__(self, root: Path, img_size: int, gray_to_3ch: bool, limit: int = 0):
        self.paths: list[Path] = []
        root = Path(root)
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                self.paths.append(p)
        if limit > 0:
            self.paths = self.paths[:limit]
        self.img_size = img_size
        self.gray_to_3ch = gray_to_3ch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        return load_rgb_tensor(self.paths[idx], self.img_size, self.gray_to_3ch)

    def path_at(self, idx: int) -> str:
        return str(self.paths[idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--calib-root", type=Path, required=True)
    ap.add_argument("--test-root", type=Path, default=None)
    ap.add_argument("--target", type=str, default="esp32p4", choices=["esp32p4", "esp32s3", "c"])
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--gray-to-3ch", action="store_true")
    ap.add_argument("--calib-steps", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--calib-limit", type=int, default=256)
    ap.add_argument("--test-limit", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", type=str, default="d2222_ferplus_64")
    args = ap.parse_args()

    from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_onnx
    from esp_ppq.executor.torch import TorchExecutor

    args.out_dir.mkdir(parents=True, exist_ok=True)
    espdl_path = args.out_dir / f"{args.name}_{args.target}.espdl"

    calib_ds = FolderImages(args.calib_root, args.img_size, args.gray_to_3ch, args.calib_limit)
    if len(calib_ds) == 0:
        raise SystemExit(f"no images under {args.calib_root}")
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False)

    def collate_fn(batch):
        # DataLoader already stacks to Tensor[B,C,H,W]
        if isinstance(batch, torch.Tensor):
            return batch.to(args.device)
        return torch.stack(list(batch), dim=0).to(args.device)

    quant_setting = QuantizationSettingFactory.espdl_setting()

    print(
        f"PTQ target={args.target} bits={args.bits} img={args.img_size} "
        f"gray3ch={args.gray_to_3ch} calib_n={len(calib_ds)}"
    )

    graph = espdl_quantize_onnx(
        onnx_import_file=str(args.onnx),
        espdl_export_file=str(espdl_path),
        calib_dataloader=calib_loader,
        calib_steps=args.calib_steps,
        input_shape=[1, 3, args.img_size, args.img_size],
        target=args.target,
        num_of_bits=args.bits,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=args.device,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )
    print(f"wrote {espdl_path}")

    # Host reference logits: ONNX Runtime FP32 vs PPQ executor INT8 on fixed test set
    test_root = args.test_root or args.calib_root
    test_ds = FolderImages(test_root, args.img_size, args.gray_to_3ch, args.test_limit)
    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"skip host ref logits ({e})")
        return 0

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    executor = TorchExecutor(graph=graph, device=args.device)

    rows = []
    for i in range(len(test_ds)):
        x = test_ds[i]
        path = test_ds.path_at(i)
        xin = x.unsqueeze(0)
        fp32 = sess.run(None, {"input": xin.numpy()})[0][0].tolist()
        int8_out = executor.forward(inputs=[xin.to(args.device)])[0]
        if isinstance(int8_out, torch.Tensor):
            q = int8_out.detach().cpu().numpy().reshape(-1).tolist()
        else:
            q = np.asarray(int8_out).reshape(-1).tolist()
        rows.append(
            {
                "index": i,
                "path": path,
                "fp32_logits": fp32,
                "int8_logits": q,
                "fp32_pred": int(np.argmax(fp32)),
                "int8_pred": int(np.argmax(q)),
            }
        )

    match = sum(1 for r in rows if r["fp32_pred"] == r["int8_pred"])
    ref = {
        "onnx": str(args.onnx),
        "espdl": str(espdl_path),
        "target": args.target,
        "img_size": args.img_size,
        "gray_to_3ch": args.gray_to_3ch,
        "n": len(rows),
        "pred_match": match,
        "pred_match_rate": match / max(1, len(rows)),
        "samples": rows,
    }
    ref_path = args.out_dir / f"{args.name}_{args.target}_host_ref.json"
    ref_path.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print(f"host ref pred_match={match}/{len(rows)} wrote {ref_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
