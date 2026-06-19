#!/usr/bin/env python
"""Per-object restyle tables — the restyled IMAGE under every condition.

The 2D twin of condition_grid.py: for each object, shows the input and the
restyled image produced under each condition (all_on + the six leave-one-out
variants), so the effect of each restyle axis is visible *before* 3D generation.
One figure per object: eval/figures/restyle/restyle_<stem>.png

    python eval/restyle_grid.py                      # retrieved arm (real photos)
    python eval/restyle_grid.py --csv eval/results_pilot.csv
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


def _input_image(stem):
    """The pipeline input shown in the raw column: a concept image (generated
    mode) if one exists on disk, otherwise the dataset photo (retrieved/manual)."""
    cands = sorted(glob.glob(str(MEDIA / f"concept_{stem}__*.png")), key=os.path.getmtime)
    if cands:
        return cands[-1]
    p = IMAGES / f"{stem}.png"
    return str(p) if p.exists() else None
OUT = EVAL / "figures" / "restyle"
TILE = 230
PAD = 8
HEAD_H = 30
TITLE_H = 44
CONDS = ["raw", "all_on", "loo_background", "loo_framing", "loo_view",
         "loo_lighting", "loo_isolation", "loo_part_visibility"]
LABELS = ["input (raw)", "all_on", "−background", "−framing", "−view",
          "−lighting", "−isolation", "−parts"]


def _font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


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
    args = ap.parse_args()

    # restyled image path by (stem, condition); prompt + input photo by stem
    restyled = {}
    prompts = {}
    for r in csv.DictReader(open(args.csv)):
        if r.get("status") != "ok":
            continue
        mp = r.get("mesh_path", "")
        stem = Path(mp).stem.split("__")[0].replace("trellis2_", "").replace("ret_", "") if mp else ""
        if not stem:
            continue
        prompts[stem] = r["prompt"]
        local = resolve(r.get("restyled_path"))
        if local:
            restyled[(stem, r["condition"])] = local

    stems = sorted({s for s, _ in restyled} | set(prompts))
    if args.objects:
        stems = [s for s in stems if s in args.objects]
    stems = [s for s in stems if sum((s, c) in restyled for c in CONDS) >= 3]
    if not stems:
        raise SystemExit("no objects with enough restyle images")

    OUT.mkdir(parents=True, exist_ok=True)
    fttl, fhd = _font(22, bold=True), _font(15, bold=True)
    n = len(CONDS)
    W = n * TILE + (n + 1) * PAD
    H = TITLE_H + HEAD_H + TILE + 2 * PAD

    for stem in stems:
        fig = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(fig)
        d.text((W // 2, TITLE_H // 2), prompts.get(stem, stem), font=fttl, fill="black", anchor="mm")
        for j, (cond, lab) in enumerate(zip(CONDS, LABELS)):
            x = PAD + j * (TILE + PAD)
            col = "#e63946" if cond == "all_on" else ("#888" if cond.startswith("loo") else "black")
            d.text((x + TILE // 2, TITLE_H + HEAD_H // 2), lab, font=fhd, fill=col, anchor="mm")
            # raw has no restyle -> show the pipeline input (concept or photo)
            img = restyled.get((stem, cond)) or (_input_image(stem) if cond == "raw" else None)
            fig.paste(_fit(img, TILE), (x, TITLE_H + HEAD_H + PAD))
        out = OUT / f"restyle_{stem}.png"
        fig.save(out)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
