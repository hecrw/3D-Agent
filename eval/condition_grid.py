#!/usr/bin/env python
"""Per-object condition tables — one figure per object showing all 8 conditions.

For each object, renders the generated mesh under every experimental condition
(raw, all_on, and the six leave-one-out variants) from a 3/4 hero angle and lays
them in a single labelled row. One figure per object ("many tables"):
  eval/figures/conditions/condition_<stem>.png

    python eval/condition_grid.py                       # generated arm
    python eval/condition_grid.py --csv eval/results_retrieved.csv
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

OUT = EVAL / "figures" / "conditions"
RCACHE = EVAL / "figures" / "_hero_renders"
TILE = 230
PAD = 8
HEAD_H = 30
TITLE_H = 44
VIEW, RES = "front_top_right", 640
CONDS = ["raw", "all_on", "loo_background", "loo_framing", "loo_view",
         "loo_lighting", "loo_isolation", "loo_part_visibility"]
LABELS = ["raw", "all_on", "−background", "−framing", "−view",
          "−lighting", "−isolation", "−parts"]


def _font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def _fit(path, size):
    if path is None:
        return Image.new("RGB", (size, size), "#f3f3f3")
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (size, size), "#f3f3f3")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(0, l - 8), max(0, t - 8),
                      min(im.width, r + 8), min(im.height, b + 8)))
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
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(EVAL / "results_pilot.csv"))
    ap.add_argument("--objects", nargs="+", default=None)
    args = ap.parse_args()
    arm = "retrieved" if "retrieved" in args.csv else "generated"

    # index mesh path + prompt by (object stem, condition)
    by = {}
    prompts = {}
    for r in csv.DictReader(open(args.csv)):
        if r.get("status") != "ok" or not r.get("mesh_path"):
            continue
        local = resolve(r["mesh_path"])
        if not local:
            continue
        stem = Path(r["mesh_path"]).stem.split("__")[0].replace("trellis2_", "").replace("ret_", "")
        by[(stem, r["condition"])] = local
        prompts[stem] = r["prompt"]

    stems = sorted({s for s, _ in by})
    if args.objects:
        stems = [s for s in stems if s in args.objects]
    # keep objects that have at least all_on + a few conditions
    stems = [s for s in stems if sum((s, c) in by for c in CONDS) >= 4]
    if not stems:
        raise SystemExit("no objects with enough conditions on disk")

    OUT.mkdir(parents=True, exist_ok=True)
    fttl, fhd = _font(22, bold=True), _font(15, bold=True)
    n = len(CONDS)
    W = n * TILE + (n + 1) * PAD
    H = TITLE_H + HEAD_H + TILE + 2 * PAD

    for stem in stems:
        print(f"rendering conditions for {stem}...")
        fig = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(fig)
        d.text((W // 2, TITLE_H // 2), prompts[stem], font=fttl, fill="black", anchor="mm")
        for j, (cond, lab) in enumerate(zip(CONDS, LABELS)):
            x = PAD + j * (TILE + PAD)
            col = "#e63946" if cond == "all_on" else ("#888" if cond.startswith("loo") else "black")
            d.text((x + TILE // 2, TITLE_H + HEAD_H // 2), lab, font=fhd, fill=col, anchor="mm")
            img = hero(by.get((stem, cond)), f"ours__{('ret_' if 'retrieved' in args.csv else '')}{stem}__{cond}")
            fig.paste(_fit(img, TILE), (x, TITLE_H + HEAD_H + PAD))
        out = OUT / f"condition_{arm}_{stem}.png"
        fig.save(out)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
