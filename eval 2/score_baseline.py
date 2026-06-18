#!/usr/bin/env python
"""Score baseline meshes (vanilla backbone, no restyle) from Colab.

    .venv/bin/python eval/score_baseline.py eval/baseline_glbs eval/results_baseline.csv

Takes a folder of <stem>.glb baseline meshes produced by the Colab notebook,
matches each to its caption in eval/dataset/captions.csv (by filename stem), and
scores it with the SAME metrics as the rest of the eval (CLIP + Gen3DEval via
Gemini + ULIP via the Modal app). Writes a results CSV with condition="baseline"
so it slots straight into analyze.py alongside your restyle results.

Resumable: rows already present with status=ok are skipped.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

# Importing run_pilot applies its patches (subprocess render, Gemini Gen3DEval,
# ULIP client) and runs django.setup().
import run_pilot  # noqa: E402
import tools       # noqa: E402

CAPTIONS_CSV = EVAL_DIR / "dataset" / "captions.csv"
FIELDS = run_pilot.CSV_FIELDS


def load_captions() -> dict[str, str]:
    out = {}
    with open(CAPTIONS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            out[Path(row["filename"].strip()).stem] = row["caption"].strip()
    return out


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists() and path.stat().st_size:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "ok":
                    done.add(r["prompt"])
    return done


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: score_baseline.py <glb_folder> <out.csv>")
    glb_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    captions = load_captions()
    done = load_done(out_path)

    meshes = sorted(glb_dir.glob("*.glb"))
    if not meshes:
        sys.exit(f"no .glb files in {glb_dir}")

    new_file = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()
        for mesh in meshes:
            stem = mesh.stem
            caption = captions.get(stem)
            if not caption:
                print(f"skip {mesh.name}: no caption for stem {stem!r}")
                continue
            if caption in done:
                print(f"skip (done) {caption!r}")
                continue
            print(f"scoring {mesh.name}  ({caption!r})")
            row = {f: "" for f in FIELDS}
            row.update(prompt=caption, condition="baseline",
                       mesh_path=str(mesh))
            try:
                views_dir = mesh.with_suffix(".views")
                view_paths = list(tools.render_mesh_views(str(mesh), views_dir).values())
                clip = tools.check_alignment(view_paths, caption)
                row.update(clip_mean=f"{clip.score:.4f}",
                           clip_accept=str(clip.accept),
                           worst_view=clip.worst_view or "",
                           per_view=json.dumps(clip.per_view))
                g = run_pilot._score_gen3deval(view_paths, caption)
                row["gen3deval"] = "" if math.isnan(g) else f"{g:.1f}"
                u = run_pilot._score_ulip(str(mesh), caption)
                row["ulip_mean"] = "" if math.isnan(u) else f"{u:.4f}"
                row["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                row.update(status="error", error=f"{type(e).__name__}: {e}")
            writer.writerow(row)
            fh.flush()
            print(f"    -> {row['status']}  clip={row['clip_mean']}  "
                  f"gen3d={row['gen3deval']}  ulip={row['ulip_mean']}")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
