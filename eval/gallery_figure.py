#!/usr/bin/env python
"""Qualitative results gallery for §4.5 — DreamFusion-style captioned grid.

Each object is a cell: [input photo | restyled | 3D mesh], with the prompt
captioned underneath. Cells are packed two per row. "Ours" is the retrieved
arm (restyle pipeline on a real photo); the mesh is rendered from a 3/4 hero
angle. Output: eval/figures/gallery_retrieved.png

    python eval/gallery_figure.py
    python eval/gallery_figure.py --objects cat_01 teapot_01 piano_01
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageChops

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL.parent))
import tools  # noqa: E402
from resolve_path import resolve  # noqa: E402

IMAGES = EVAL / "dataset" / "images"
OUT = EVAL / "figures"
RCACHE = OUT / "_hero_renders"
TILE = 280
G = 8                     # gap between the 3 images in a cell
SUBLAB_H = 22
CAP_H = 30
CELL_GAP = 40
NCOLS = 2
VIEW, RES = "front_top_right", 640
SUBLABELS = ["input photo", "restyled", "3D mesh"]
DEFAULT = ["bicycle_01", "armchair_01", "piano_01", "teapot_01",
           "saxophone_01", "chandelier_01", "hydrant_01", "guitar_01"]


def _font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _crop_white(im, pad=10):
    """Crop the white background so a rendered mesh fills its tile."""
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - pad), max(0, t - pad),
                      min(im.width, r + pad), min(im.height, b + pad)))
    return im


def _fit(path, size, crop=False):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = Image.new("RGB", (size, size), "#eeeeee")
    if crop:
        im = _crop_white(im)
    im.thumbnail((size, size))
    c = Image.new("RGB", (size, size), "white")
    c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return c


def hero(mesh_path, key):
    if not mesh_path or not Path(mesh_path).exists():
        return None
    dest = RCACHE / f"{key}.png"
    if dest.exists():
        return dest
    RCACHE.mkdir(parents=True, exist_ok=True)
    try:
        paths = tools.render_mesh_views(mesh_path, RCACHE / f"_tmp_{key}",
                                        views=[VIEW], image_size=RES)
        Path(paths[VIEW]).replace(dest)
        return dest
    except Exception as e:  # noqa: BLE001
        print(f"  render failed {key}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=DEFAULT)
    args = ap.parse_args()

    rows = {}
    for r in csv.DictReader(open(EVAL / "results_retrieved.csv")):
        if (r.get("status") == "ok" and r["condition"] == "all_on"
                and resolve(r.get("mesh_path"))):
            stem = Path(r["mesh_path"]).stem.split("__")[0].replace("trellis2_", "").replace("ret_", "")
            rows[stem] = r
    picks = [s for s in args.objects if s in rows]
    if not picks:
        raise SystemExit("no matching objects")

    print(f"rendering {len(picks)} meshes from {VIEW}...")
    cells = {}
    for s in picks:
        r = rows[s]
        cells[s] = (IMAGES / f"{s}.png", resolve(r.get("restyled_path")) or "",
                    hero(resolve(r["mesh_path"]), f"ours__{s}"), r["prompt"])

    cell_w = 3 * TILE + 2 * G
    cell_h = SUBLAB_H + TILE + CAP_H
    nrows = (len(picks) + NCOLS - 1) // NCOLS
    W = NCOLS * cell_w + (NCOLS + 1) * CELL_GAP
    H = CELL_GAP + nrows * (cell_h + CELL_GAP)
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)
    fsub, fcap = _font(15), _font(17, bold=True)

    for i, s in enumerate(picks):
        cx = CELL_GAP + (i % NCOLS) * (cell_w + CELL_GAP)
        cy = CELL_GAP + (i // NCOLS) * (cell_h + CELL_GAP)
        inp, restyled, mesh, prompt = cells[s]
        imgs = [inp, restyled, mesh]
        for j, im in enumerate(imgs):
            x = cx + j * (TILE + G)
            d.text((x + TILE // 2, cy + SUBLAB_H // 2), SUBLABELS[j],
                   font=fsub, fill="#777777", anchor="mm")
            # crop white margin off the rendered mesh so it fills the tile
            grid.paste(_fit(im, TILE, crop=(j == 2)), (x, cy + SUBLAB_H))
        d.text((cx + cell_w // 2, cy + SUBLAB_H + TILE + CAP_H // 2),
               prompt, font=fcap, fill="black", anchor="mm")

    OUT.mkdir(exist_ok=True)
    out = OUT / "gallery_retrieved.png"
    grid.save(out)
    print(f"-> {out}  ({len(picks)} objects)")


if __name__ == "__main__":
    main()
