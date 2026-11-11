#!/usr/bin/env python
"""Backfill the vlm_score column for ok rows that are missing it.

    .venv/bin/python eval/backfill_vlm_score.py eval/results_pilot.csv

Only needs the saved mesh (mesh_path) per row — re-renders its views and scores
with VLM judge (Gemini). No mesh regeneration, no backend calls. Rewrites the
CSV in place (atomic via temp file). Safe to re-run; already-scored rows skip.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Importing run_pilot applies its monkey-patches (subprocess render, Gemini
# VLM judge) and runs django.setup().
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
import run_pilot  # noqa: E402
import tools       # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: backfill_vlm_score.py <results.csv>")
    path = Path(sys.argv[1])
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "vlm_score" not in fields:
        sys.exit(f"{path}: missing vlm_score column")

    todo = [r for r in rows
            if r.get("status") == "ok"
            and not r.get("vlm_score")
            and r.get("mesh_path") and Path(r["mesh_path"]).exists()]
    print(f"{len(todo)} ok rows need a VLM judge score")

    for i, r in enumerate(todo, 1):
        mesh = r["mesh_path"]
        print(f"[{i}/{len(todo)}] {r['condition']:22} {Path(mesh).name}")
        try:
            views_dir = Path(mesh).with_suffix(".views")
            view_paths = list(tools.render_mesh_views(mesh, views_dir).values())
            score = run_pilot._score_vlm(view_paths, r["prompt"])
            import math
            r["vlm_score"] = "" if math.isnan(score) else f"{score:.1f}"
            print(f"    -> vlm={r['vlm_score']}")
        except Exception as e:  # noqa: BLE001
            print(f"    !! failed: {type(e).__name__}: {e}")

    # atomic rewrite
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
