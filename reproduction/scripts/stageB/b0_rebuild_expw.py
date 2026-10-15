#!/usr/bin/env python3
"""Rebuild the ExpW 112px cache from raw images with the original split lists.

Membership and split come from the per-sample list saved by the original run
(results/from_autodl/evidence/expw/expw_split_lists.json). Samples recorded as
resize fallback (mtcnn=false) are cropped and resized; the others were resumed
from an earlier MTCNN run with no record, so they are re-aligned with the same
MTCNN settings and crop/align helpers as datasets/prepare_expw.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.prepare_expw import crop_face, mtcnn_batch_align, parse_label_lst  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-lists", type=Path, default=ROOT / "results/from_autodl/evidence/expw/expw_split_lists.json")
    ap.add_argument("--label-lst", type=Path, default=ROOT / "datasets/expw/official/data/label/label.lst")
    ap.add_argument("--image-root", type=Path, default=ROOT / "datasets/expw/official/images/origin")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/expw_rebuild")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(image_size=112, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                  factor=0.709, post_process=True, device=args.device, keep_all=False)
    rows = {}
    for r in parse_label_lst(args.label_lst):
        rows.setdefault((r["image_name"], r["face_id"], r["label"]), r)
    lists = json.loads(args.split_lists.read_text(encoding="utf-8"))
    entries = [e for split in ("train", "val", "test") for e in lists[split]]
    if args.limit:
        entries = entries[: args.limit]

    stats = {"n": len(entries), "done_skip": 0, "fallback_recorded": 0, "mtcnn_hit": 0,
             "mtcnn_miss_resize": 0, "missing": 0}
    per_sample = {}

    def load(e):
        out = args.out_dir / e["path"]
        if out.is_file():
            return e, None, "exists"
        r = rows.get((e["image_name"], e["face_id"], e["label"]))
        p = args.image_root / e["image_name"]
        if r is None or not p.is_file():
            return e, None, "missing"
        img = Image.open(p)
        img.load()
        face = crop_face(img, r)
        return e, face, "ok" if face is not None else "missing"

    t0 = time.time()
    pending = []

    def flush():
        # One face per MTCNN call: with numpy>=1.24 a batch raises as soon as one
        # face has no detection, which silently turned whole batches into resizes.
        aligned_all = [mtcnn_batch_align([f], mtcnn, fixed=160)[0] for _, f in pending]
        for (e, _), (aligned, hit) in zip(pending, aligned_all):
            aligned.save(args.out_dir / e["path"], quality=95)
            stats["mtcnn_hit" if hit else "mtcnn_miss_resize"] += 1
            per_sample[e["path"]] = "mtcnn" if hit else "mtcnn_miss_resize"
        pending.clear()

    dirs = {str(Path(e["path"]).parent) for e in entries}
    for d in dirs:
        (args.out_dir / d).mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (e, face, kind) in enumerate(pool.map(load, entries)):
            if kind == "exists":
                stats["done_skip"] += 1
            elif kind == "missing":
                stats["missing"] += 1
                per_sample[e["path"]] = "missing"
            elif e.get("mtcnn") is False:
                face.convert("RGB").resize((112, 112), Image.BILINEAR).save(args.out_dir / e["path"], quality=95)
                stats["fallback_recorded"] += 1
                per_sample[e["path"]] = "fallback_recorded"
            else:
                pending.append((e, face))
                if len(pending) >= args.batch_size:
                    flush()
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{len(entries)} {time.time() - t0:.0f}s {json.dumps(stats)}", flush=True)
    if pending:
        flush()

    stats["sec"] = round(time.time() - t0, 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rebuild_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    with (args.out_dir / "rebuild_per_sample.json").open("a", encoding="utf-8") as f:
        json.dump(per_sample, f)
        f.write("\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
