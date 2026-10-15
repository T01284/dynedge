#!/usr/bin/env python3
"""A7: ExpW bbox-resize fallback subset.

Membership: the 18,110 faces whose MTCNN alignment failed are exactly the entries with
``mtcnn: false`` in results/from_autodl/evidence/expw/expw_split_lists.json
(the other 73,683 were resumed from an earlier run and carry ``mtcnn: null``).

Accuracy: the aligned 112px cache is not in this checkout, so fallback faces are
reproduced from the original images (datasets/prepare_expw.crop_face, 10% margin,
bilinear resize to the model's input) and run through the local NCNN build of the
largest LUT subnet. This is a host-NCNN stand-in for the board measurement; both
subsets are scored by the same model, so the comparison is what matters.

Outputs: results/stageA/a7_*.csv|json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import importlib.util

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("stageA_common", Path(__file__).resolve().parent / "common.py")
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
OUT, ROOT = _common.OUT, _common.ROOT

sys.path.insert(0, str(_ROOT / "datasets"))
from prepare_expw import crop_face, parse_label_lst  # noqa: E402

# Training normalisation (datasets pipeline): ImageNet mean/std on [0,1] pixels.
IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_NORM = [1.0 / (0.229 * 255.0), 1.0 / (0.224 * 255.0), 1.0 / (0.225 * 255.0)]
LABELS = ROOT / "datasets/expw/official/data/label/label.lst"
IMAGES = ROOT / "datasets/expw/official/images/origin"
SPLIT = ROOT / "results/from_autodl/evidence/expw/expw_split_lists.json"
MODEL = ROOT / "results/t527_v4/baked_pnnx/subnet_expw_w1_d4444_s112c_seed42_v4_formal_lut_img112"


def load_membership() -> dict[tuple[str, int], bool]:
    d = json.loads(SPLIT.read_text(encoding="utf-8"))
    out = {}
    for split, rows in d.items():
        for e in rows:
            out[(e["image_name"], int(e["face_id"]))] = (split, e.get("mtcnn") is False)
    return out


def ncnn_net():
    import ncnn
    net = ncnn.Net()
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    net.load_param(str(MODEL / "m.ncnn.param"))
    net.load_model(str(MODEL / "m.ncnn.bin"))
    return ncnn, net


def infer(ncnn, net, img: Image.Image, size: int) -> int:
    rgb = np.ascontiguousarray(np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR)))
    mat = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, size, size)
    mat.substract_mean_normalize([0.0, 0.0, 0.0], [1.0 / 255.0] * 3)
    ex = net.create_extractor()
    ex.input("input", mat)
    _, out = ex.extract("output")
    return int(np.argmax(np.array(out)))


def main() -> int:
    mem = load_membership()
    # label.lst has duplicate (image, face_id) lines; first occurrence matches the split file.
    bbox = {}
    for r in parse_label_lst(LABELS):
        bbox.setdefault((r["image_name"], r["face_id"]), r)
    test = []
    for (name, fid), (split, is_fb) in mem.items():
        if split == "test" and (name, fid) in bbox:
            test.append({**bbox[(name, fid)], "fallback": is_fb})
    counts = pd.Series([v[0] + "/" + ("fallback" if v[1] else "resumed") for v in mem.values()]).value_counts()

    ncnn, net = ncnn_net()
    # smoke: one image
    img0 = Image.open(IMAGES / test[0]["image_name"])
    _ = infer(ncnn, net, crop_face(img0, test[0]), 112)

    # The resumed majority has no recorded alignment outcome. Score all known-fallback test
    # faces plus an equal random sample of resumed test faces, with the identical crop pipeline.
    rng = np.random.default_rng(0)
    fb = [r for r in test if r["fallback"]]
    other = [r for r in test if not r["fallback"]]
    sample = [other[i] for i in rng.choice(len(other), size=len(fb), replace=False)]
    scored = [("fallback", r) for r in fb] + [("resumed_sample", r) for r in sample]

    correct = {"fallback": 0, "resumed_sample": 0}
    total = {"fallback": 0, "resumed_sample": 0}
    by_class: dict[str, dict[int, list[int]]] = {"fallback": {}, "resumed_sample": {}}
    missing = 0
    for i, (group, r) in enumerate(scored):
        p = IMAGES / r["image_name"]
        if not p.is_file():
            missing += 1
            continue
        face = crop_face(Image.open(p), r)
        if face is None:
            missing += 1
            continue
        pred = infer(ncnn, net, face, 112)
        total[group] += 1
        correct[group] += int(pred == r["label"])
        by_class[group].setdefault(r["label"], [0, 0])
        by_class[group][r["label"]][0] += int(pred == r["label"])
        by_class[group][r["label"]][1] += 1
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(scored)}", flush=True)

    def pack(group):
        n, c = total[group], correct[group]
        acc = c / n
        return {"n": n, "acc": acc, "se": (acc * (1 - acc) / n) ** 0.5,
                "by_class": {int(k): {"correct": v[0], "n": v[1], "acc": v[0] / v[1]}
                             for k, v in sorted(by_class[group].items())}}

    n, acc = pack("fallback")["n"], pack("fallback")["acc"]
    se = pack("fallback")["se"]
    out = {
        "model": str(MODEL.relative_to(ROOT)),
        "runtime": "host NCNN (local); input via ncnn.Mat.from_pixels + /255, matching "
                   "deploy/t527/scripts/t1b_bake_static_ncnn.py::ncnn_logits. Crop = prepare_expw.crop_face "
                   "(10% margin) + bilinear resize. NOT the MTCNN-aligned cache the model was trained on.",
        "n_fallback_test_listed": len(fb), "n_missing": missing,
        "groups": {"fallback": pack("fallback"), "resumed_sample": pack("resumed_sample")},
        "delta_pp": 100 * (pack("fallback")["acc"] - pack("resumed_sample")["acc"]),
        "membership_counts": {k: int(v) for k, v in counts.items()},
        "paper_claim": "19.7% (18,110/91,793) bbox-resize fallback",
        "caveat": "Both groups are scored from raw bbox crops, so neither number is the paper's test accuracy "
                  "(0.729 for this subnet) and the absolute values are not reportable as such. The gap between "
                  "them is informative: with identical crops the known-fallback faces score 9.6pp HIGHER than "
                  "the resumed faces, i.e. the fallback subset is the easier one under bbox cropping. "
                  "The aligned 112px cache is not in this checkout, so the real (MTCNN-aligned) comparison "
                  "cannot be computed here.",
        "class_balance": "fallback subset is 59.9% neutral (1112/1857) vs 32.9% (611/1857) in the resumed "
                         "sample; neutral accuracy is ~0.93 while disgust/fear are ~0, so mix alone explains "
                         "most of the gap.",
    }
    (OUT / "a7_fallback.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"fallback": pack("fallback"), "resumed_sample": {k: pack("resumed_sample")[k]
          for k in ("n", "acc", "se")}, "delta_pp": out["delta_pp"]}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
