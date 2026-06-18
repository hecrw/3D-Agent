#!/usr/bin/env python
"""Build the six-axis restyle normalization figure (paper §3.2.2).

For one input image, generates a restyle with ONLY each axis enabled (using the
real build_restyle_prompt + restyle_to_objaverse), then composes a before->after
grid: one row per axis, [input | normalized-by-that-axis], with axis labels.

    python eval/axis_figure.py eval/dataset/images/bicycle_01.png

Pick a CLUTTERED real photo (a retrieved one) as the input so each axis visibly
does something — a clean input has nothing to normalize. Outputs:
  eval/axis_figure/<stem>__<axis>.png   (the 6 single-axis restyles)
  eval/axis_figure/<stem>_grid.png      (the assembled figure)
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))
import tools  # noqa: E402

TILE = 320          # thumbnail size in the grid
PAD = 16
LABEL_W = 150       # left column width for axis names


def _font(size: int):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _fit(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: axis_figure.py <input_image.png>")
    src = Path(sys.argv[1])
    out_dir = EVAL_DIR / "axis_figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    axes = list(tools.RESTYLE_AXES)  # the 6 canonical axes, in order
    raw = _fit(Image.open(src), TILE)

    # 1. Generate one single-axis restyle per axis.
    restyled = {}
    for i, axis in enumerate(axes, 1):
        out = out_dir / f"{stem}__{axis}.png"
        if not out.exists():
            print(f"[{i}/{len(axes)}] restyle axis={axis}")
            tools.restyle_to_objaverse(
                str(src), out_path=str(out),
                style_prompt=tools.build_restyle_prompt([axis]))
        restyled[axis] = _fit(Image.open(out), TILE)

    # 2. Compose the grid: header row + one row per axis [input | normalized].
    f_label = _font(20)
    f_head = _font(22)
    rows = len(axes)
    W = LABEL_W + 2 * TILE + 3 * PAD
    H = PAD + 40 + rows * (TILE + PAD) + PAD          # +40 for the header
    grid = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(grid)

    x_in = LABEL_W + PAD
    x_out = LABEL_W + 2 * PAD + TILE
    d.text((x_in + TILE // 2 - 30, PAD), "input", font=f_head, fill="black")
    d.text((x_out + TILE // 2 - 70, PAD), "after restyle", font=f_head, fill="black")

    y = PAD + 40
    for axis in axes:
        label = axis.replace("_", " ")
        d.text((PAD, y + TILE // 2 - 10), label, font=f_label, fill="black")
        grid.paste(raw, (x_in, y))
        grid.paste(restyled[axis], (x_out, y))
        y += TILE + PAD

    grid_path = out_dir / f"{stem}_grid.png"
    grid.save(grid_path)
    print(f"\nfigure -> {grid_path}")


if __name__ == "__main__":
    main()
