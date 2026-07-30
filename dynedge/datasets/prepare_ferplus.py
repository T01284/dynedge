"""Build FERPlus image folders from fer2013.csv + fer2013new.csv (majority vote).

Majority protocol follows microsoft/FERPlus (Barsoum et al.):
  1) zero out single-vote outliers
  2) keep only if max vote > 50% of remaining votes
  3) discard if winner is unknown/NF or no clear majority
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def majority_label(votes: list[int]) -> int | None:
    """votes: 10 ints = 8 emotions + unknown + NF.

    Returns class index in [0,7], or None if sample should be discarded.
    """
    raw = [float(v) for v in votes]
    # Official outlier removal: drop counts <= 1
    for i in range(len(raw)):
        if raw[i] <= 1.0:
            raw[i] = 0.0
    total = sum(raw)
    if total <= 0:
        return None
    winner = int(np.argmax(raw))
    if raw[winner] <= 0.5 * total:
        return None
    # winner among unknown(8) / NF(9) → discard
    if winner >= 8:
        return None
    return winner


def pixels_to_rgb(pixel_str: str, size: int) -> Image.Image:
    arr = np.fromstring(pixel_str, dtype=np.uint8, sep=" ").reshape(48, 48)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fer2013-csv", type=Path, required=True)
    ap.add_argument("--ferplus-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument(
        "--force",
        action="store_true",
        help="wipe out-dir train/val/test before writing",
    )
    args = ap.parse_args()

    out = args.out_dir
    if args.force and out.exists():
        for split in ("train", "val", "test"):
            p = out / split
            if p.exists():
                shutil.rmtree(p)
        for name in ("prepare_stats.json", "skip_log.jsonl"):
            f = out / name
            if f.exists():
                f.unlink()

    for split in ("train", "val", "test"):
        for c in range(8):
            (out / split / f"{c:02d}_{EMOTIONS[c]}").mkdir(parents=True, exist_ok=True)

    # Usage mapping: Training->train, PublicTest->val, PrivateTest->test
    usage_map = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}

    with args.fer2013_csv.open(newline="", encoding="utf-8") as f:
        fer_rows = list(csv.DictReader(f))
    with args.ferplus_csv.open(newline="", encoding="utf-8") as f:
        plus_rows = list(csv.DictReader(f))

    assert len(fer_rows) == len(plus_rows), (len(fer_rows), len(plus_rows))

    stats = {
        "ok": 0,
        "skip_no_majority": 0,
        "protocol": "ferplus_official_majority",
        "img_size": args.img_size,
        "by_split": {},
        "by_class": {},
    }
    skip_log = []

    for i, (fer, plus) in enumerate(tqdm(list(zip(fer_rows, plus_rows)), desc="FERPlus")):
        usage = fer["Usage"]
        split = usage_map.get(usage)
        if split is None:
            skip_log.append({"idx": i, "reason": f"bad_usage:{usage}"})
            continue
        votes = [
            int(plus[k])
            for k in [
                "neutral",
                "happiness",
                "surprise",
                "sadness",
                "anger",
                "disgust",
                "fear",
                "contempt",
                "unknown",
                "NF",
            ]
        ]
        label = majority_label(votes)
        if label is None:
            stats["skip_no_majority"] += 1
            skip_log.append({"idx": i, "reason": "no_majority", "votes": votes})
            continue

        img = pixels_to_rgb(fer["pixels"], args.img_size)
        name = plus.get("Image name") or f"fer{i:07d}.png"
        rel = Path(split) / f"{label:02d}_{EMOTIONS[label]}" / name
        img.save(out / rel)
        stats["ok"] += 1
        stats["by_split"][split] = stats["by_split"].get(split, 0) + 1
        stats["by_class"][EMOTIONS[label]] = stats["by_class"].get(EMOTIONS[label], 0) + 1

    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out / "skip_log.jsonl").write_text(
        "\n".join(json.dumps(x) for x in skip_log[:5000]), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
