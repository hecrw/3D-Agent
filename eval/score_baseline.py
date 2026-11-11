#!/usr/bin/env python
"""Score baseline meshes (vanilla backbone, no restyle) from Colab.

    .venv/bin/python eval/score_baseline.py <glb_dir> <out.csv> \
        --backbone trellis2 --input-mode retrieved --run-id <run-id>

Takes a folder of <stem>.glb baseline meshes produced by the Colab notebook,
matches each to its caption in eval/dataset/captions.csv (by filename stem), and
scores it with the SAME metrics as the rest of the eval (CLIP + VLM judge via
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

# Importing run_pilot applies its patches (subprocess render, Gemini VLM judge,
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
    import argparse
    import os
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("glb_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--backbone", default="", help="trellis2 | hunyuan3d2 | partcrafter")
    ap.add_argument("--input-mode", default="", help="generated | retrieved")
    ap.add_argument("--run-id", default="")
    a = ap.parse_args()
    glb_dir, out_path = a.glb_dir, a.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
                       mesh_path=str(mesh), run_id=a.run_id,
                       input_mode=a.input_mode, stem=stem,
                       backbone_tool=a.backbone,
                       backbone_endpoint=f"{a.backbone}:local-direct",
                       judge_model=run_pilot.JUDGE_MODEL,
                       started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       condition_enforced="True")
            try:
                views_dir = mesh.with_suffix(".views")
                view_paths = list(tools.render_mesh_views(str(mesh), views_dir).values())
                clip = tools.check_alignment(view_paths, caption)
                row.update(clip_mean=f"{clip.score:.4f}",
                           clip_accept=str(clip.accept),
                           worst_view=clip.worst_view or "",
                           per_view=json.dumps(clip.per_view))
                raw = {}
                g = run_pilot._score_vlm(view_paths, caption, raw_out=raw)
                row["vlm_score"] = "" if math.isnan(g) else f"{g:.1f}"
                row["vlm_score_raw"] = raw.get("text", "")
                u = run_pilot._score_ulip(str(mesh), caption)
                row["ulip_mean"] = "" if math.isnan(u) else f"{u:.4f}"
                row["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                row.update(status="error", error=f"{type(e).__name__}: {e}")
            writer.writerow(row)
            fh.flush()
            print(f"    -> {row['status']}  clip={row['clip_mean']}  "
                  f"vlm={row['vlm_score']}  ulip={row['ulip_mean']}")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
