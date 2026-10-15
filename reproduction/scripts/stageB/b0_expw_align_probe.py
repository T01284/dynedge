#!/usr/bin/env python3
"""Which crops did the original ExpW cache actually contain for the 73,683 resumed samples?

History (runs/phase0_monitor_local.log, 2026-07-19): an early per-image run (mtcnn(face) on the raw
bbox crop, no batching) reached >=38,994 of 91,793 label rows; a batched run (bs=512) resumed from
there; a bs=1024 run resumed 73,683 files and processed the last 18,110, all of them resize
fallbacks (with numpy>=1.24 facenet_pytorch raises on any batch containing a missed face, and the
batch helper then resizes the whole batch). Only the last run's per-sample record survived, so
the resumed files are a mix of per-image MTCNN crops and unrecorded batch-bug resizes, split at an
unknown label-row index c.

The original searches recorded the supernet's val accuracy for 412 candidates. We rebuild the val
split in three versions with identical membership (recorded fallbacks identical in all three):
  perimage : resumed samples through the original per-image path (MTCNN on the raw crop)
  aligned  : data/expw_rebuild (MTCNN one face per call on the 160 px resized crop)
  resize   : resumed samples bbox-cropped (10% margin) and resized to 112 (the batch-bug path)
and save per-sample correctness for a stratified subset of candidates. For each c, the predicted
original accuracy uses perimage for resumed rows with index < c and resize otherwise; c is fit by
least squares against the recorded accuracies.

Outputs: results/stageB/expw_align_probe.json (+ .npz with per-sample correctness)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.common import build_transforms  # noqa: E402
from datasets.prepare_expw import crop_face, parse_label_lst, tensor_to_pil  # noqa: E402
from models.elastic_mbv3 import ElasticFERSuperNet, SubnetConfig  # noqa: E402


def build_versions(entries, rows, image_root: Path, resize_root: Path, perimage_root: Path) -> dict:
    from facenet_pytorch import MTCNN

    mtcnn = MTCNN(image_size=112, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                  post_process=True, device="cpu", keep_all=False)
    st = {"perimage_hit": 0, "perimage_miss": 0}
    for n, e in enumerate(entries):
        r_dst, p_dst = resize_root / e["path"], perimage_root / e["path"]
        if r_dst.is_file() and p_dst.is_file():
            continue
        img = Image.open(image_root / e["image_name"])
        img.load()
        face = crop_face(img, rows[(e["image_name"], e["face_id"], e["label"])]).convert("RGB")
        resized = face.resize((112, 112), Image.BILINEAR)
        for d in (r_dst, p_dst):
            d.parent.mkdir(parents=True, exist_ok=True)
        resized.save(r_dst, quality=95)
        out = None
        if e.get("mtcnn") is not False:
            try:
                out = mtcnn(face)
            except Exception:  # noqa: BLE001
                out = None
        if out is not None:
            tensor_to_pil(out.detach().cpu()).save(p_dst, quality=95)
            st["perimage_hit"] += 1
        else:
            resized.save(p_dst, quality=95)
            st["perimage_miss"] += e.get("mtcnn") is not False
        if (n + 1) % 1000 == 0:
            print(f"built {n + 1}/{len(entries)} {st}", flush=True)
    return st


@torch.no_grad()
def correct_vec(model, loader, cfg: SubnetConfig) -> np.ndarray:
    out = []
    for x, y in loader:
        if cfg.gray or cfg.img_size == 64:
            x = x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        if x.shape[-1] != cfg.img_size:
            x = F.interpolate(x, size=(cfg.img_size, cfg.img_size), mode="bilinear", align_corners=False)
        out.append((model(x, subnet=cfg, return_all_exits=False).argmax(1) == y).numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=ROOT / "results/from_autodl/supernet/supernet_expw_seed42_v4_best.pt")
    ap.add_argument("--consistency", type=Path, default=ROOT / "results/stageB/expw_rebuild_consistency.json")
    ap.add_argument("--split-lists", type=Path, default=ROOT / "results/from_autodl/evidence/expw/expw_split_lists.json")
    ap.add_argument("--label-lst", type=Path, default=ROOT / "datasets/expw/official/data/label/label.lst")
    ap.add_argument("--image-root", type=Path, default=ROOT / "datasets/expw/official/images/origin")
    ap.add_argument("--aligned-root", type=Path, default=ROOT / "data/expw_rebuild")
    ap.add_argument("--resize-root", type=Path, default=ROOT / "data/expw_resize_probe")
    ap.add_argument("--perimage-root", type=Path, default=ROOT / "data/expw_perimage_probe")
    ap.add_argument("--n-per-width", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "results/stageB/expw_align_probe.json")
    args = ap.parse_args()

    label_rows = parse_label_lst(args.label_lst)
    rows, row_idx = {}, {}
    for i, r in enumerate(label_rows):
        k = (r["image_name"], r["face_id"], r["label"])
        rows.setdefault(k, r)
        row_idx.setdefault(k, i)
    val = json.loads(args.split_lists.read_text(encoding="utf-8"))["val"]
    build = build_versions(val, rows, args.image_root, args.resize_root, args.perimage_root)
    print("versions built", build, flush=True)

    tf = build_transforms(False, 112, gray_to_3ch=False)
    roots = {"perimage": args.perimage_root, "aligned": args.aligned_root, "resize": args.resize_root}
    dsets = {v: datasets.ImageFolder(r / "aligned" / "val", transform=tf) for v, r in roots.items()}
    rel = [str(Path(p).relative_to(roots["resize"])).replace("\\", "/") for p, _ in dsets["resize"].samples]
    for v, ds in dsets.items():
        assert [str(Path(p).relative_to(roots[v])).replace("\\", "/") for p, _ in ds.samples] == rel, v
    meta = {e["path"]: e for e in val}
    idx = np.array([row_idx[(meta[p]["image_name"], meta[p]["face_id"], meta[p]["label"])] for p in rel])
    recorded_fb = np.array([meta[p].get("mtcnn") is False for p in rel])
    loaders = {v: DataLoader(ds, batch_size=256, num_workers=4) for v, ds in dsets.items()}

    cons = json.loads(args.consistency.read_text(encoding="utf-8"))["rows"]
    by_w: dict[str, list] = {}
    for r in cons:
        by_w.setdefault(r["key"].split("_")[0], []).append(r)
    rng = random.Random(0)
    pick = [r for w in sorted(by_w) for r in rng.sample(by_w[w], min(args.n_per_width, len(by_w[w])))]

    saved = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = saved
    model = ElasticFERSuperNet(num_classes=7, slimmable=True, arch="v3_wide")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()

    corr = {v: [] for v in roots}
    orig = []
    for i, r in enumerate(pick):
        w, d, s = r["key"].split("_")
        cfg = SubnetConfig(width=float(w[1:]), depths=tuple(int(c) for c in d[1:]), img_size=int(s[1:-1]),
                           gray=s.endswith("g"))
        for v in roots:
            corr[v].append(correct_vec(model, loaders[v], cfg))
        orig.append(r["val_acc_orig"])
        print(f"{i + 1}/{len(pick)} {r['key']} orig={r['val_acc_orig']:.4f} "
              + " ".join(f"{v}={corr[v][-1].mean():.4f}" for v in roots), flush=True)
    C = {v: np.stack(corr[v]) for v in roots}
    orig = np.array(orig)
    np.savez_compressed(args.out.with_suffix(".npz"), idx=idx, recorded_fb=recorded_fb, orig=orig,
                        keys=np.array([r["key"] for r in pick]), **C)

    def predict(cut: int, aligned_version: str) -> np.ndarray:
        use_al = (~recorded_fb) & (idx < cut)
        return np.where(use_al[None, :], C[aligned_version], C["resize"]).mean(1)

    grid = np.arange(0, int(idx.max()) + 2, 100)
    fits = {}
    for av in ("perimage", "aligned"):
        sse = np.array([np.sum((predict(c, av) - orig) ** 2) for c in grid])
        best = int(grid[sse.argmin()])
        res = 100 * (predict(best, av) - orig)
        near = grid[sse <= sse.min() * 1.5]
        fits[av] = {"best_cut": best, "cut_range_sse_1.5x": [int(near.min()), int(near.max())],
                    "n_resumed_val_aligned": int(((~recorded_fb) & (idx < best)).sum()),
                    "n_resumed_val_resize": int(((~recorded_fb) & (idx >= best)).sum()),
                    "rmse_pp": float(np.sqrt(np.mean(res ** 2))), "max_abs_pp": float(np.abs(res).max())}
    pure = {name: {"rmse_pp": float(np.sqrt(np.mean((100 * (p - orig)) ** 2))),
                   "max_abs_pp": float(np.abs(100 * (p - orig)).max())}
            for name, p in (("all_resize", predict(0, "perimage")),
                            ("all_perimage", predict(10 ** 9, "perimage")),
                            ("all_aligned160", predict(10 ** 9, "aligned")))}
    summary = {"n_candidates": len(pick), "n_val": len(rel), "n_recorded_fallback_val": int(recorded_fb.sum()),
               "build": build, "pure_hypotheses": pure, "cut_fit": fits,
               "rows": [{"key": r["key"], "orig": float(o), **{v: float(C[v][i].mean()) for v in roots}}
                        for i, (r, o) in enumerate(zip(pick, orig))]}
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
