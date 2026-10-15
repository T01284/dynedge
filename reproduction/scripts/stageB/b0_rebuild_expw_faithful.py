#!/usr/bin/env python3
"""Rebuild the ExpW 112px cache as the original training run actually saw it.

scripts/stageB/b0_expw_align_probe.py showed that the original cache was split at label-row index
~39,300 (fit range 39,100-39,600; val accuracy of 32 supernet candidates reproduced with RMSE
0.06 pp): rows before the cut went through the original per-image path (MTCNN on the raw bbox
crop, resize on a miss), rows after it were bbox-crop resizes (batch-MTCNN bug), whether or not the
final run recorded them. Membership and split come from expw_split_lists.json.

Outputs: data/expw_faithful/aligned/{train,val,test}/..., data/expw_faithful/faithful_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.prepare_expw import crop_face, parse_label_lst, tensor_to_pil  # noqa: E402

_M = None


def _init() -> None:
    global _M
    import torch
    from facenet_pytorch import MTCNN

    torch.set_num_threads(2)
    _M = MTCNN(image_size=112, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
               post_process=True, device="cpu", keep_all=False)


def _one(job) -> str:
    path, image, row, use_mtcnn, out = job
    dst = Path(out) / path
    if dst.is_file():
        return "exists"
    img = Image.open(image)
    img.load()
    face = crop_face(img, row).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_mtcnn:
        try:
            t = _M(face)
        except Exception:  # noqa: BLE001
            t = None
        if t is not None:
            tensor_to_pil(t.detach().cpu()).save(dst, quality=95)
            return "mtcnn"
        face.resize((112, 112), Image.BILINEAR).save(dst, quality=95)
        return "mtcnn_miss_resize"
    face.resize((112, 112), Image.BILINEAR).save(dst, quality=95)
    return "resize"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-lists", type=Path, default=ROOT / "results/from_autodl/evidence/expw/expw_split_lists.json")
    ap.add_argument("--label-lst", type=Path, default=ROOT / "datasets/expw/official/data/label/label.lst")
    ap.add_argument("--image-root", type=Path, default=ROOT / "datasets/expw/official/images/origin")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/expw_faithful")
    ap.add_argument("--cut", type=int, default=39300)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rows, idx = {}, {}
    for i, r in enumerate(parse_label_lst(args.label_lst)):
        k = (r["image_name"], r["face_id"], r["label"])
        rows.setdefault(k, r)
        idx.setdefault(k, i)
    lists = json.loads(args.split_lists.read_text(encoding="utf-8"))
    jobs, seen = [], set()
    for split in ("train", "val", "test"):
        for e in lists[split]:
            if e["path"] in seen:
                continue
            seen.add(e["path"])
            k = (e["image_name"], e["face_id"], e["label"])
            use = e.get("mtcnn") is not False and idx[k] < args.cut
            jobs.append((e["path"], str(args.image_root / e["image_name"]), rows[k], use, str(args.out_dir)))
    stats: dict = {"cut": args.cut, "n": len(jobs)}
    with Pool(args.workers, initializer=_init) as pool:
        for i, kind in enumerate(pool.imap(_one, jobs, chunksize=32)):
            stats[kind] = stats.get(kind, 0) + 1
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(jobs)} {stats}", flush=True)
    (args.out_dir / "faithful_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
