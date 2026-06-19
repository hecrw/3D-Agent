#!/usr/bin/env python
"""Master results gallery — every object our model generated, in one grid.

For each object (all_on condition) renders the final mesh from a 3/4 hero angle
and lays them out as a dense captioned grid. The comprehensive "catalogue of
generations" figure. Output: eval/figures/master_gallery.png

    python eval/master_gallery.py                       # generated arm (results_pilot)
    python eval/master_gallery.py --csv eval/results_retrieved.csv --cols 5
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

OUT = EVAL / "figures"
RCACHE = OUT / "_hero_renders"
TILE = 300
PAD = 14
CAP_H = 30
VIEW, RES = "front_top_right", 640


def _font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _fit(path, size):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = Image.new("RGB", (size, size), "#eeeeee")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - 10), max(0, t - 10),
                      min(im.width, r + 10), min(im.height, b + 10)))
    im.thumbnail((size, size))
    c = Image.new("RGB", (size, size), "white")
    c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return c


def hero(mesh_path, key):
    dest = RCACHE / f"{key}.png"
    if dest.exists():
        return dest
    if not mesh_path or not Path(mesh_path).exists():
        return None
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
    ap.add_argument("--csv", default=str(EVAL / "results_pilot.csv"))
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    objs = {}
    for r in csv.DictReader(open(args.csv)):
        local = resolve(r.get("mesh_path")) if r.get("status") == "ok" and r["condition"] == "all_on" else None
        if local:
            stem = Path(r["mesh_path"]).stem.split("__")[0]
            objs[r["prompt"]] = (stem, local)
    items = sorted(objs.items())
    if not items:
        raise SystemExit("no all_on meshes found on disk for this CSV")
    print(f"rendering {len(items)} meshes from {VIEW}...")
    tiles = [(prompt, hero(mp, f"ours__{stem}")) for prompt, (stem, mp) in items]

    ncols = args.cols
    nrows = (len(tiles) + ncols - 1) // ncols
    cell_h = TILE + CAP_H
    W = ncols * TILE + (ncols + 1) * PAD
    H = (nrows) * (cell_h + PAD) + PAD
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)
    fcap = _font(14)
    for i, (prompt, rp) in enumerate(tiles):
        cx = PAD + (i % ncols) * (TILE + PAD)
        cy = PAD + (i // ncols) * (cell_h + PAD)
        grid.paste(_fit(rp, TILE), (cx, cy))
        cap = prompt if len(prompt) <= 30 else prompt[:29] + "…"
        d.text((cx + TILE // 2, cy + TILE + CAP_H // 2), cap, font=fcap,
               fill="black", anchor="mm")

    OUT.mkdir(exist_ok=True)
    arm = "retrieved" if "retrieved" in args.csv else "generated"
    out = OUT / f"master_gallery_{arm}.png"
    grid.save(out)
    print(f"-> {out}  ({len(tiles)} objects, {ncols}x{nrows})")


if __name__ == "__main__":
    main()
