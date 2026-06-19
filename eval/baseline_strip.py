#!/usr/bin/env python
"""Pipeline-vs-baseline qualitative strip for §4.5.4 — fair, hero-angle version.

"Ours" is the RETRIEVED all_on mesh, i.e. our restyle pipeline run on the SAME
real photo the baselines were given — so every column starts from identical
input. Each mesh is re-rendered from a 3/4 hero angle (front_top_right) for a
flattering, consistent view. Output: eval/figures/baseline_strip.png

    python eval/baseline_strip.py
    python eval/baseline_strip.py --objects piano_01 armchair_01 teapot_01
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
PAD = 12
LABEL_W = 170
HEAD_H = 36
VIEW = "front_top_right"            # 3/4 hero angle
RES = 640
COLS = [("Ours (restyle)", "ours"),
        ("TRELLIS", "trellis"), ("Hunyuan3D-2", "hunyuan"), ("PartCrafter", "partcrafter")]
DEFAULT = ["armchair_01", "piano_01", "teapot_01", "saxophone_01", "chandelier_01"]


def _font(sz, bold=False):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def hero_render(mesh_path: str, key: str) -> Path | None:
    """Render mesh from the 3/4 hero angle (cached)."""
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


def _fit(path, size):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = Image.new("RGB", (size, size), "#eeeeee")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bbox:  # crop white margin so the mesh fills the tile
        l, t, r, b = bbox
        im = im.crop((max(0, l - 10), max(0, t - 10),
                      min(im.width, r + 10), min(im.height, b + 10)))
    im.thumbnail((size, size))
    c = Image.new("RGB", (size, size), "white")
    c.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return c


def render_each(picks, mesh_for, prompts):
    """One big standalone figure per object: title + 4 large columns."""
    big = 460
    pad = 18
    head_h = 40
    title_h = 48
    fttl, fhd = _font(26, bold=True), _font(20, bold=True)
    each_dir = OUT / "baseline_each"
    each_dir.mkdir(parents=True, exist_ok=True)
    for stem in picks:
        renders = {src: hero_render(mesh_for(stem, src), f"{src}__{stem}") for _, src in COLS}
        W = len(COLS) * big + (len(COLS) + 1) * pad
        H = title_h + head_h + big + 2 * pad
        fig = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(fig)
        d.text((W // 2, title_h // 2), prompts[stem], font=fttl, fill="black", anchor="mm")
        xs = [pad + i * (big + pad) for i in range(len(COLS))]
        for x, (name, src) in zip(xs, COLS):
            d.text((x + big // 2, title_h + head_h // 2), name, font=fhd,
                   fill=("#e63946" if src == "ours" else "black"), anchor="mm")
            fig.paste(_fit(renders[src], big), (x, title_h + head_h))
        out = each_dir / f"baseline_{stem}.png"
        fig.save(out)
        print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=DEFAULT)
    ap.add_argument("--each", action="store_true",
                    help="emit one big standalone figure per object")
    args = ap.parse_args()

    ours = {}
    for r in csv.DictReader(open(EVAL / "results_retrieved.csv")):
        local = resolve(r.get("mesh_path")) if r.get("status") == "ok" and r["condition"] == "all_on" else None
        if local:
            stem = Path(r["mesh_path"]).stem.split("__")[0].replace("trellis2_", "").replace("ret_", "")
            ours[stem] = (r["prompt"], local)

    def mesh_for(stem, src):
        if src == "ours":
            return ours.get(stem, (None, None))[1]
        return str(EVAL / f"baseline_glbs_{src}" / f"{stem}.glb")

    picks = [s for s in args.objects
             if s in ours and all(Path(mesh_for(s, src)).exists() for _, src in COLS if src != "ours")]
    if not picks:
        raise SystemExit("no objects with all four meshes")

    if args.each:
        render_each(picks, mesh_for, {s: ours[s][0] for s in picks})
        return

    # render everything (hero angle, cached)
    print(f"rendering {len(picks)} objects x {len(COLS)} meshes from {VIEW}...")
    cells = {}
    for stem in picks:
        for _, src in COLS:
            cells[(stem, src)] = hero_render(mesh_for(stem, src), f"{src}__{stem}")

    W = LABEL_W + len(COLS) * TILE + (len(COLS) + 1) * PAD
    H = HEAD_H + PAD + len(picks) * (TILE + PAD) + PAD
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)
    fh, fl = _font(18, bold=True), _font(15)
    xs = [LABEL_W + PAD + i * (TILE + PAD) for i in range(len(COLS))]
    for x, (name, src) in zip(xs, COLS):
        d.text((x + TILE // 2, HEAD_H // 2), name, font=fh,
               fill=("#e63946" if src == "ours" else "black"), anchor="mm")

    y = HEAD_H + PAD
    for stem in picks:
        p = ours[stem][0]
        d.text((PAD, y + TILE // 2), p[:26] + ("…" if len(p) > 26 else ""),
               font=fl, fill="black", anchor="lm")
        for x, (_, src) in zip(xs, COLS):
            grid.paste(_fit(cells[(stem, src)], TILE), (x, y))
        y += TILE + PAD

    OUT.mkdir(exist_ok=True)
    out = OUT / "baseline_strip.png"
    grid.save(out)
    print(f"-> {out}  ({', '.join(picks)})")


if __name__ == "__main__":
    main()
