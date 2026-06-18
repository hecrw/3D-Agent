#!/usr/bin/env python
"""Fetch a sample of real Objaverse assets and render them cleanly.

For the domain-gap figure (paper §1.5 / §3.2.2): column 3 needs a genuine
Objaverse training render. This downloads a few assets of a chosen LVIS category
and renders a clean front + three-quarter view with the project's own
render_mesh_views — so the Objaverse render matches your pipeline's render style.

    pip install objaverse
    python eval/objaverse_sample.py sofa --n 3
    python eval/objaverse_sample.py            # lists available categories

Outputs eval/objaverse_sample/<uid>__front.png and <uid>__hero.png.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    try:
        import objaverse
    except ImportError:
        sys.exit("pip install objaverse")

    anns = objaverse.load_lvis_annotations()  # {category: [uid, ...]}
    if len(sys.argv) < 2:
        cats = sorted(anns)
        print(f"{len(cats)} LVIS categories. Examples:")
        for c in ["sofa", "chair", "piano", "bicycle", "teapot", "guitar",
                  "saxophone", "cat", "vase", "lamp", "armchair"]:
            if c in anns:
                print(f"  {c}  ({len(anns[c])} assets)")
        print("\nUsage: python eval/objaverse_sample.py <category> [--n 3]")
        return

    category = sys.argv[1].lower()
    n = 3
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])

    if category not in anns:
        close = [c for c in anns if category in c]
        sys.exit(f"category {category!r} not in LVIS. "
                 f"Close matches: {close[:12] or 'none — run with no args to list'}")

    uids = anns[category][:n]
    print(f"downloading {len(uids)} '{category}' assets from Objaverse...")
    objects = objaverse.load_objects(uids=uids)  # {uid: local_glb_path}

    glb_dir = EVAL_DIR / "objaverse_sample" / "glbs"
    glb_dir.mkdir(parents=True, exist_ok=True)
    for uid, glb in objects.items():
        dest_glb = glb_dir / f"{category}__{uid}.glb"
        shutil.copy(glb, dest_glb)
        print(f"saved {dest_glb}")

    print(f"\n{len(objects)} '{category}' assets -> {glb_dir}/")


if __name__ == "__main__":
    main()
