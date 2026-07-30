#!/usr/bin/env python3
"""PTQ a single-op ONNX for ESP-DL LUT mining (random-tensor calib, any CxHxW)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RandomNCHW(Dataset):
    def __init__(self, shape: tuple[int, int, int], n: int = 32):
        self.c, self.h, self.w = shape
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randn(self.c, self.h, self.w)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--target", choices=["esp32p4", "esp32s3", "c"], default="esp32p4")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--cin", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--calib-steps", type=int, default=16)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from esp_ppq.api import QuantizationSettingFactory, espdl_quantize_onnx

    args.out_dir.mkdir(parents=True, exist_ok=True)
    espdl_path = args.out_dir / f"{args.name}_{args.target}.espdl"
    ds = RandomNCHW((args.cin, args.h, args.w), n=max(32, args.calib_steps))
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    def collate_fn(batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(args.device)
        return torch.stack(list(batch), dim=0).to(args.device)

    print(f"op-PTQ {args.name} target={args.target} shape=1x{args.cin}x{args.h}x{args.w}")
    espdl_quantize_onnx(
        onnx_import_file=str(args.onnx),
        espdl_export_file=str(espdl_path),
        calib_dataloader=loader,
        calib_steps=args.calib_steps,
        input_shape=[1, args.cin, args.h, args.w],
        target=args.target,
        num_of_bits=args.bits,
        collate_fn=collate_fn,
        setting=QuantizationSettingFactory.espdl_setting(),
        device=args.device,
        error_report=False,
        skip_export=False,
        export_test_values=False,
        verbose=0,
    )
    print(f"wrote {espdl_path} bytes={espdl_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
