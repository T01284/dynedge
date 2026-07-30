"""Prepare ExpW: bbox crop -> batched MTCNN align -> 112x112 cache + stratified split."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def parse_label_lst(path: Path):
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            rows.append(
                {
                    "image_name": parts[0],
                    "face_id": int(parts[1]),
                    "top": int(float(parts[2])),
                    "left": int(float(parts[3])),
                    "right": int(float(parts[4])),
                    "bottom": int(float(parts[5])),
                    "confidence": float(parts[6]),
                    "label": int(parts[7]),
                }
            )
    return rows


def crop_face(img: Image.Image, row: dict, margin: float = 0.1) -> Image.Image | None:
    w, h = img.size
    top, left, right, bottom = row["top"], row["left"], row["right"], row["bottom"]
    bw, bh = right - left, bottom - top
    if bw <= 1 or bh <= 1:
        return None
    top = max(0, int(top - bh * margin))
    left = max(0, int(left - bw * margin))
    right = min(w, int(right + bw * margin))
    bottom = min(h, int(bottom + bh * margin))
    if right <= left or bottom <= top:
        return None
    return img.crop((left, top, right, bottom))


def tensor_to_pil(t) -> Image.Image:
    arr = ((t + 1) * 127.5).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def mtcnn_batch_align(faces: list[Image.Image], mtcnn, fixed: int = 160) -> list[tuple[Image.Image, bool]]:
    """Batch MTCNN on fixed-size inputs for higher GPU occupancy. Fallback = resize."""
    if not faces:
        return []
    if mtcnn is None:
        return [(f.resize((112, 112), Image.BILINEAR), False) for f in faces]

    fixed_imgs = [f.convert("RGB").resize((fixed, fixed), Image.BILINEAR) for f in faces]
    outs = None
    try:
        outs = mtcnn(fixed_imgs)
    except Exception:
        outs = None

    results: list[tuple[Image.Image, bool]] = []
    if outs is None:
        return [(f.convert("RGB").resize((112, 112), Image.BILINEAR), False) for f in faces]

    # facenet-pytorch returns list[Tensor|None] for list input, or a batch tensor
    if hasattr(outs, "ndim") and getattr(outs, "ndim", 0) == 4:
        # BxCxHxW
        for i in range(outs.size(0)):
            results.append((tensor_to_pil(outs[i]), True))
        return results

    if not isinstance(outs, (list, tuple)):
        outs = [outs]

    for face, out in zip(faces, outs):
        if out is None:
            results.append((face.convert("RGB").resize((112, 112), Image.BILINEAR), False))
        else:
            try:
                results.append((tensor_to_pil(out), True))
            except Exception:
                results.append((face.convert("RGB").resize((112, 112), Image.BILINEAR), False))
    # length mismatch guard
    while len(results) < len(faces):
        f = faces[len(results)]
        results.append((f.convert("RGB").resize((112, 112), Image.BILINEAR), False))
    return results


def stratified_split(rows: list[dict], seed: int, ratios=(0.8, 0.1, 0.1)):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["label"]].append(i)
    rng = random.Random(seed)
    split = {}
    for lab, idxs in by.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for j, ix in enumerate(idxs):
            if j < n_train:
                split[ix] = "train"
            elif j < n_train + n_val:
                split[ix] = "val"
            else:
                split[ix] = "test"
    return split


def out_rel(row: dict, split: str) -> Path:
    lab = row["label"]
    fname = f"{Path(row['image_name']).stem}_f{row['face_id']}.jpg"
    return Path("aligned") / split / f"{lab:02d}_{EMOTIONS[lab]}" / fname


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", type=Path, required=True, help=".../images/origin")
    ap.add_argument("--label-lst", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/expw_split.yaml"))
    ap.add_argument("--no-mtcnn", action="store_true", help="resize only (faster debug)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: process first N labels")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="MTCNN batch size (larger => higher GPU util; try 512/1024/2048)",
    )
    ap.add_argument("--mtcnn-input", type=int, default=160, help="fixed crop size fed to MTCNN")
    ap.add_argument("--resume", action="store_true", help="skip faces whose jpg already exists")
    ap.add_argument("--load-workers", type=int, default=8, help="CPU threads to prefetch crops")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 2026))
    min_conf = float(cfg.get("min_bbox_confidence", 0.0))

    rows = parse_label_lst(args.label_lst)
    n_total = len(rows)
    rows = [r for r in rows if r["confidence"] >= min_conf and 0 <= r["label"] <= 6]
    n_filt = len(rows)
    if args.limit > 0:
        rows = rows[: args.limit]

    mtcnn = None
    if not args.no_mtcnn:
        from facenet_pytorch import MTCNN

        mtcnn = MTCNN(
            image_size=112,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=args.device,
            keep_all=False,
        )

    out = args.out_dir
    for split in ("train", "val", "test"):
        for c in range(7):
            (out / "aligned" / split / f"{c:02d}_{EMOTIONS[c]}").mkdir(parents=True, exist_ok=True)

    split_map = stratified_split(rows, seed=seed)
    stats = {
        "n_total_labeled": n_total,
        "n_after_confidence_filter": n_filt,
        "n_processed": 0,
        "n_align_ok": 0,
        "n_align_skip": 0,
        "n_resume_skip": 0,
        "n_mtcnn_hit": 0,
        "n_resize_fallback": 0,
        "batch_size": args.batch_size,
        "by_split": {},
        "by_class": {},
    }
    skip_log = []
    split_lists = {"train": [], "val": [], "test": []}

    pending: list[dict] = []

    def flush_batch():
        nonlocal pending
        if not pending:
            return
        faces = [p["face"] for p in pending]
        aligned_list = mtcnn_batch_align(faces, mtcnn, fixed=args.mtcnn_input)
        for p, (aligned, used_mtcnn) in zip(pending, aligned_list):
            if used_mtcnn:
                stats["n_mtcnn_hit"] += 1
            else:
                stats["n_resize_fallback"] += 1
            rel = p["rel"]
            aligned.save(out / rel, quality=95)
            entry = {
                "path": str(rel).replace("\\", "/"),
                "label": p["row"]["label"],
                "image_name": p["row"]["image_name"],
                "face_id": p["row"]["face_id"],
                "mtcnn": used_mtcnn,
                "crop_wh": list(p["before"]),
            }
            split_lists[p["split"]].append(entry)
            stats["n_align_ok"] += 1
            stats["n_processed"] += 1
            stats["by_split"][p["split"]] = stats["by_split"].get(p["split"], 0) + 1
            lab_name = EMOTIONS[p["row"]["label"]]
            stats["by_class"][lab_name] = stats["by_class"].get(lab_name, 0) + 1
        pending = []

    def load_one(i: int, row: dict):
        """Return pending item, resume marker, or skip marker."""
        split = split_map[i]
        rel = out_rel(row, split)
        if args.resume and (out / rel).is_file():
            return ("resume", i, row, split, rel)
        img_path = args.image_root / row["image_name"]
        if not img_path.is_file():
            return ("skip", i, row, "missing_file")
        try:
            img = Image.open(img_path)
            img.load()
        except Exception as e:
            return ("skip", i, row, f"open_fail:{e}")
        face = crop_face(img, row)
        if face is None:
            return ("skip", i, row, "bad_bbox")
        return (
            "ok",
            {
                "row": row,
                "split": split,
                "rel": rel,
                "face": face,
                "before": face.size,
            },
        )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    indices = list(range(len(rows)))
    # Process in chunk windows so we can flush GPU batches promptly
    window = max(args.batch_size * 2, args.batch_size)
    pbar = tqdm(total=len(rows), desc=f"ExpW(bs={args.batch_size})")
    with ThreadPoolExecutor(max_workers=max(1, args.load_workers)) as pool:
        for start in range(0, len(indices), window):
            chunk = indices[start : start + window]
            futs = [pool.submit(load_one, i, rows[i]) for i in chunk]
            for fut in as_completed(futs):
                item = fut.result()
                kind = item[0]
                if kind == "resume":
                    _, i, row, split, rel = item
                    stats["n_resume_skip"] += 1
                    stats["n_align_ok"] += 1
                    stats["n_processed"] += 1
                    stats["by_split"][split] = stats["by_split"].get(split, 0) + 1
                    lab_name = EMOTIONS[row["label"]]
                    stats["by_class"][lab_name] = stats["by_class"].get(lab_name, 0) + 1
                    split_lists[split].append(
                        {
                            "path": str(rel).replace("\\", "/"),
                            "label": row["label"],
                            "image_name": row["image_name"],
                            "face_id": row["face_id"],
                            "mtcnn": None,
                            "resumed": True,
                        }
                    )
                elif kind == "skip":
                    _, i, row, reason = item
                    stats["n_align_skip"] += 1
                    skip_log.append({"i": i, "reason": reason, **row})
                else:
                    pending.append(item[1])
                    if len(pending) >= args.batch_size:
                        flush_batch()
                pbar.update(1)
            if len(pending) >= args.batch_size:
                flush_batch()
    pbar.close()
    flush_batch()

    stats["skip_rate"] = stats["n_align_skip"] / max(1, stats["n_align_skip"] + stats["n_align_ok"])
    if mtcnn is not None and (stats["n_mtcnn_hit"] + stats["n_resize_fallback"]):
        stats["mtcnn_fail_rate_among_ok"] = stats["n_resize_fallback"] / (
            stats["n_mtcnn_hit"] + stats["n_resize_fallback"]
        )

    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    with (out / "skip_log.jsonl").open("w", encoding="utf-8") as f:
        for x in skip_log:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    split_path = Path("configs/expw_split_lists.json")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_lists, indent=2), encoding="utf-8")

    cfg["n_total_labeled"] = n_total
    cfg["n_after_confidence_filter"] = n_filt
    cfg["n_align_ok"] = stats["n_align_ok"]
    cfg["n_align_skip"] = stats["n_align_skip"]
    cfg["skip_rate"] = stats["skip_rate"]
    cfg["split_file"] = str(split_path).replace("\\", "/")
    args.config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"Wrote {split_path} and updated {args.config}")


if __name__ == "__main__":
    main()
