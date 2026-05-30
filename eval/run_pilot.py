#!/usr/bin/env python
"""Restyle-preprocessing ablation sweep.

For each (image, condition, backbone) it:
  1. builds the restyle prompt for the condition (or skips restyle for `raw`),
  2. restyles the image (Gemini),
  3. generates a mesh on the chosen backbone (Modal),
  4. renders multi-view PNGs,
  5. scores view/caption alignment with CLIP,
and appends one row to a CSV. Already-present rows are skipped so the sweep is
resumable after an interruption or a partial Modal failure.

Run from the repo root:
    .venv/bin/python eval/run_pilot.py --limit 10 --out eval/results_pilot.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

# Import the project's tool implementations. tools.py is Django-free, so the
# harness runs standalone without booting the web app.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tools  # noqa: E402

DATASET_DIR = REPO_ROOT / "eval" / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
CAPTIONS_CSV = DATASET_DIR / "captions.csv"
WORK_DIR = REPO_ROOT / "eval" / "work"  # restyled images, meshes, renders

# Cheap, FIXED mesh settings. Held constant across conditions so mesh quality
# does not confound the restyle effect — and cheap so the sweep is affordable.
CHEAP = {
    "trellis2":   dict(pipeline_type="1024", remesh=False,
                       decimation_target=200_000, texture_size=1024),
    "hunyuan3d2": dict(steps=30, octree_resolution=192),
    "partcrafter": dict(num_parts=3, num_inference_steps=30),
}

BACKENDS = ("trellis2", "hunyuan3d2", "partcrafter")

CSV_FIELDS = [
    "image", "condition", "backbone",
    "clip_mean", "clip_accept", "worst_view", "per_view",
    "restyled_path", "mesh_path", "status", "error", "seconds",
]


def conditions() -> list[tuple[str, "list[str] | None"]]:
    """(name, axes) for every condition. axes=None for raw (no restyle)."""
    axis_names = list(tools.RESTYLE_AXES)
    conds: list[tuple[str, list[str] | None]] = [
        ("raw", None),                 # no restyle at all
        ("all_on", axis_names),        # every axis
    ]
    for ax in axis_names:              # leave-one-out
        conds.append((f"loo_{ax}", [a for a in axis_names if a != ax]))
    return conds


def load_captions() -> dict[str, str]:
    if not CAPTIONS_CSV.exists():
        sys.exit(f"missing {CAPTIONS_CSV} — see eval/README.md for the format")
    out: dict[str, str] = {}
    with open(CAPTIONS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["filename"].strip()] = row["caption"].strip()
    return out


def load_done(csv_path: Path) -> set[tuple[str, str, str]]:
    """Keys (image, condition, backbone) already recorded with a terminal status."""
    done: set[tuple[str, str, str]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") in ("ok", "error"):
                done.add((row["image"], row["condition"], row["backbone"]))
    return done


def generate(backbone: str, image_path: str, out_path: str) -> str:
    if backbone == "trellis2":
        return tools.trellis2(image_path, out_path, **CHEAP["trellis2"])
    if backbone == "hunyuan3d2":
        return tools.hunyuan3d2(image_path, out_path, **CHEAP["hunyuan3d2"])
    if backbone == "partcrafter":
        return tools.partcrafter(image_path, out_path, **CHEAP["partcrafter"])
    raise ValueError(f"unknown backbone {backbone!r}")


def run_one(image_file: Path, caption: str, cond_name: str,
            axes: "list[str] | None", backbone: str) -> dict:
    stem = f"{image_file.stem}__{cond_name}__{backbone}"
    row = {f: "" for f in CSV_FIELDS}
    row.update(image=image_file.name, condition=cond_name, backbone=backbone)
    t0 = time.time()
    try:
        # 1. restyle (or pass the raw image straight through)
        if axes is None:
            restyled = str(image_file)
        else:
            restyled = str(WORK_DIR / f"{stem}.restyle.png")
            prompt = tools.build_restyle_prompt(axes)
            tools.restyle_to_objaverse(str(image_file), restyled, style_prompt=prompt)
        row["restyled_path"] = restyled

        # 2. generate mesh
        mesh = str(WORK_DIR / f"{stem}.glb")
        generate(backbone, restyled, mesh)
        row["mesh_path"] = mesh

        # 3. render views + 4. CLIP score against the GT caption
        views_dir = WORK_DIR / f"{stem}.views"
        view_paths = tools.render_mesh_views(mesh, views_dir, views="default")
        rep = tools.check_alignment(view_paths, caption)
        row.update(
            clip_mean=f"{rep.score:.4f}",
            clip_accept=str(rep.accept),
            worst_view=rep.worst_view or "",
            per_view=json.dumps(rep.per_view),
            status="ok",
        )
    except Exception as e:  # noqa: BLE001 — one failure must not kill the sweep
        row.update(status="error", error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
    row["seconds"] = f"{time.time() - t0:.1f}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="eval/results_pilot.csv", type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="max images to process (0 = all)")
    ap.add_argument("--backbones", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    args = ap.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    captions = load_captions()
    images = sorted(p for p in IMAGES_DIR.glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"no images in {IMAGES_DIR}")

    conds = conditions()
    done = load_done(args.out)
    total = len(images) * len(conds) * len(args.backbones)
    print(f"{len(images)} images x {len(conds)} conditions x "
          f"{len(args.backbones)} backbones = {total} generations "
          f"({len(done)} already done)")

    new_file = not args.out.exists()
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        i = 0
        for img in images:
            caption = captions.get(img.name)
            if not caption:
                print(f"  skip {img.name}: no caption in captions.csv")
                continue
            for cond_name, axes in conds:
                for backbone in args.backbones:
                    i += 1
                    key = (img.name, cond_name, backbone)
                    if key in done:
                        print(f"[{i}/{total}] skip (done) {key}")
                        continue
                    print(f"[{i}/{total}] {img.name} | {cond_name} | {backbone}")
                    row = run_one(img, caption, cond_name, axes, backbone)
                    writer.writerow(row)
                    fh.flush()  # persist each row so a crash loses at most one
                    print(f"    -> {row['status']} "
                          f"clip={row['clip_mean']} ({row['seconds']}s)")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
