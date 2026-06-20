#!/usr/bin/env python
"""One combined diagram of the restyled IMAGE under every condition.

Rows = objects, columns = the 8 conditions (raw, all_on, 6x leave-one-out), with
a single shared header row. A slide-ready PNG showing what each restyle axis does
to the input *before* 3D generation. Output: eval/figures/restyle_conditions.png

    python eval/restyle_conditions_figure.py
    python eval/restyle_conditions_figure.py --objects piano_01 armchair_01 saxophone_01 hydrant_01
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL))
from resolve_path import resolve  # noqa: E402

IMAGES = EVAL / "dataset" / "images"
MEDIA = EVAL.parent / "media" / "3d_outputs"
OUT = EVAL / "figures" / "restyle_conditions.png"

CONDS = ["raw", "all_on", "loo_background", "loo_framing", "loo_view",
         "loo_lighting", "loo_isolation", "loo_part_visibility"]
LABELS = ["input (raw)", "all on", "− background", "− framing", "− view",
          "− lighting", "− isolation", "− parts"]

TILE = 200
PAD = 6
HEAD_H = 40
ROWLAB_W = 130


def _font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _input_image(stem, prefer_photo=False):
    """Pipeline input for the raw column. For the retrieved arm we want the real
    dataset photo; for the generated arm, the concept image."""
    photo = IMAGES / f"{stem}.png"
    if prefer_photo and photo.exists():
        return str(photo)
    cands = sorted(glob.glob(str(MEDIA / f"concept_{stem}__*.png")), key=os.path.getmtime)
    if cands:
        return cands[-1]
    return str(photo) if photo.exists() else None


def _fit(path, size):
    if not path or not Path(path).exists():
        return Image.new("RGB", (size, size), "#f3f3f3")
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (size, size), "#f3f3f3")
    im.thumbnail((size, size))
    c = Image.new("RGB", (size, size), "white")
    c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(EVAL / "results_retrieved.csv"))
    ap.add_argument("--objects", nargs="+", default=None)
    ap.add_argument("--max", type=int, default=5, help="max objects (rows)")
    ap.add_argument("--axes", nargs="+", default=None,
                    help="subset of loo axes to show, e.g. --axes background isolation view")
    ap.add_argument("--out", default=None, help="output png path")
    args = ap.parse_args()

    # optionally restrict the columns to raw, all_on + a curated subset of axes
    conds, labels = CONDS, LABELS
    if args.axes:
        keep = ["raw", "all_on"] + [f"loo_{a}" for a in args.axes]
        idx = [CONDS.index(c) for c in keep if c in CONDS]
        conds = [CONDS[i] for i in idx]
        labels = [LABELS[i] for i in idx]

    restyled, prompts = {}, {}
    for r in csv.DictReader(open(args.csv)):
        if r.get("status") != "ok" or not r.get("mesh_path"):
            continue
        stem = Path(r["mesh_path"]).stem.split("__")[0].replace("trellis2_", "").replace("ret_", "")
        prompts[stem] = r["prompt"]
        local = resolve(r.get("restyled_path"))
        if local:
            restyled[(stem, r["condition"])] = local

    # objects with the most condition coverage first
    def cov(s):
        return sum((s, c) in restyled for c in CONDS)
    stems = args.objects or sorted({s for s, _ in restyled}, key=lambda s: -cov(s))
    stems = [s for s in stems if cov(s) >= 6][:args.max]
    if not stems:
        raise SystemExit("no objects with enough restyle images")

    # retrieved arm -> show the real dataset photo as the raw input
    prefer_photo = "retrieved" in args.csv

    n = len(conds)
    W = ROWLAB_W + n * TILE + (n + 1) * PAD
    H = HEAD_H + len(stems) * (TILE + PAD) + PAD
    fig = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(fig)
    fhd, frow = _font(15, bold=True), _font(15, bold=True)

    xs = [ROWLAB_W + PAD + j * (TILE + PAD) for j in range(n)]
    for x, cond, lab in zip(xs, conds, labels):
        col = "#e63946" if cond == "all_on" else ("#888" if cond.startswith("loo") else "#222")
        d.text((x + TILE // 2, HEAD_H // 2), lab, font=fhd, fill=col, anchor="mm")

    y = HEAD_H
    for stem in stems:
        d.text((ROWLAB_W - 8, y + TILE // 2),
               (prompts.get(stem, stem)[:16]), font=frow, fill="#222", anchor="rm")
        for x, cond in zip(xs, conds):
            img = restyled.get((stem, cond)) or (_input_image(stem, prefer_photo) if cond == "raw" else None)
            fig.paste(_fit(img, TILE), (x, y))
        y += TILE + PAD

    out = Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(out)
    print(f"-> {out}  ({len(stems)} objects x {n} conditions, {W}x{H})")


if __name__ == "__main__":
    main()
