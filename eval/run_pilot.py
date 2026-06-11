#!/usr/bin/env python
"""Restyle-preprocessing ablation sweep (text-to-3D pipeline).

For each (prompt, condition, backbone) it:
  1. generates a concept image from the text prompt (Gemini),
  2. restyles the concept image for the condition (or skips for `raw`),
  3. generates a mesh on the chosen backbone (Modal),
  4. renders multi-view PNGs,
  5. scores with CLIP, Gen3DEval (VLM-as-judge), and optionally ULIP-2,
and appends one row to a CSV. Already-present rows are skipped so the sweep is
resumable after an interruption or a partial Modal failure.

Run from the repo root:
    .venv/bin/python eval/run_pilot.py --limit 10 --out eval/results_pilot.csv

Metrics:
  clip_mean      — CLIP image-text cosine similarity (mean over views)
  gen3deval      — VLM-as-judge score 1-10 (Claude Haiku, up to 4 views)
  ulip_mean      — ULIP-2 image-text similarity (nan if ULIP not installed)
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tools  # noqa: E402

DATASET_DIR = REPO_ROOT / "eval" / "dataset"
IMAGES_DIR  = DATASET_DIR / "images"   # concept images written here
CAPTIONS_CSV = DATASET_DIR / "captions.csv"
WORK_DIR = REPO_ROOT / "eval" / "work"  # restyled images, meshes, renders

# Cheap, FIXED mesh settings held constant across conditions so mesh quality
# does not confound the restyle effect.
CHEAP = {
    "trellis2":    dict(pipeline_type="1024", remesh=False,
                        decimation_target=200_000, texture_size=1024),
    "hunyuan3d2":  dict(steps=30, octree_resolution=192),
    "partcrafter": dict(num_parts=3, num_inference_steps=30),
}

BACKENDS = ("trellis2", "hunyuan3d2", "partcrafter")

CSV_FIELDS = [
    "image", "condition", "backbone",
    "clip_mean", "clip_accept", "worst_view", "per_view",
    "gen3deval", "ulip_mean",
    "concept_path", "restyled_path", "mesh_path",
    "status", "error", "seconds",
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_gen3deval(view_paths: list[str], caption: str) -> float:
    """VLM-as-judge quality score (1–10) using Claude Haiku.

    Sends up to 4 rendered views + the prompt caption. Returns nan on failure.
    """
    try:
        import anthropic
    except ImportError:
        return math.nan

    client = anthropic.Anthropic()
    selected = view_paths[:4]
    content: list = []
    for p in selected:
        with open(p, "rb") as fh:
            data = base64.standard_b64encode(fh.read()).decode()
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data},
        })
    content.append({"type": "text", "text": (
        f'These are rendered views of a 3D mesh generated from the prompt: "{caption}". '
        "Rate the overall 3D generation quality from 1 to 10 considering: "
        "(1) geometric accuracy and completeness, "
        "(2) texture and appearance quality, "
        "(3) semantic alignment with the prompt. "
        "Reply with a single integer between 1 and 10 — nothing else."
    )})
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": content}],
        )
        return float(resp.content[0].text.strip())
    except Exception:  # noqa: BLE001
        return math.nan


def _score_ulip(view_paths: list[str], caption: str) -> float:
    """ULIP-2 image-text similarity score (mean over views).

    Requires the ULIP-2 library:
        git clone https://github.com/salesforce/ULIP
        pip install -e ULIP/

    Returns nan if not installed — the sweep continues without it.
    """
    try:
        import torch
        from ulip.models.ULIP2 import ULIP2  # type: ignore[import]
        from ulip.utils import get_text_features, get_image_features  # type: ignore[import]
    except ImportError:
        return math.nan

    try:
        model = ULIP2(point_encoder="PointBERT")
        model.eval()
        sims = []
        txt_feat = get_text_features(model, [caption])
        for p in view_paths:
            img_feat = get_image_features(model, [p])
            sim = torch.nn.functional.cosine_similarity(img_feat, txt_feat).item()
            sims.append(sim)
        return sum(sims) / len(sims) if sims else math.nan
    except Exception:  # noqa: BLE001
        return math.nan


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_captions() -> dict[str, str]:
    if not CAPTIONS_CSV.exists():
        sys.exit(f"missing {CAPTIONS_CSV} — see eval/README.md for the format")
    out: dict[str, str] = {}
    with open(CAPTIONS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["filename"].strip()] = row["caption"].strip()
    return out


def ensure_concept_images(captions: dict[str, str]) -> dict[str, Path]:
    """Generate a concept image for each caption that doesn't exist yet."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, caption in captions.items():
        out = IMAGES_DIR / filename
        if not out.exists():
            print(f"[concept] generating '{filename}' ...")
            tools.generate_concept_image(caption, str(out))
        paths[filename] = out
    return paths


def load_done(csv_path: Path) -> set[tuple[str, str, str]]:
    done: set[tuple[str, str, str]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") in ("ok", "error"):
                done.add((row["image"], row["condition"], row["backbone"]))
    return done


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def conditions() -> list[tuple[str, "list[str] | None"]]:
    """(name, axes) for every condition. axes=None means no restyle (raw)."""
    axis_names = list(tools.RESTYLE_AXES)
    conds: list[tuple[str, list[str] | None]] = [
        ("raw",    None),        # baseline — no restyle
        ("all_on", axis_names),  # all 6 axes
    ]
    for ax in axis_names:        # leave-one-out ×6
        conds.append((f"loo_{ax}", [a for a in axis_names if a != ax]))
    return conds


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(backbone: str, image_path: str, out_path: str) -> str:
    if backbone == "trellis2":
        return tools.trellis2(image_path, out_path, **CHEAP["trellis2"])
    if backbone == "hunyuan3d2":
        return tools.hunyuan3d2(image_path, out_path, **CHEAP["hunyuan3d2"])
    if backbone == "partcrafter":
        return tools.partcrafter(image_path, out_path, **CHEAP["partcrafter"])
    raise ValueError(f"unknown backbone {backbone!r}")


# ---------------------------------------------------------------------------
# Main per-cell runner
# ---------------------------------------------------------------------------

def run_one(concept_path: Path, caption: str,
            cond_name: str, axes: "list[str] | None",
            backbone: str) -> dict:
    stem = f"{concept_path.stem}__{cond_name}__{backbone}"
    row = {f: "" for f in CSV_FIELDS}
    row.update(image=concept_path.name, condition=cond_name, backbone=backbone,
               concept_path=str(concept_path))
    t0 = time.time()
    try:
        # 1. restyle (or pass concept image straight through for raw)
        if axes is None:
            restyled = str(concept_path)
        else:
            restyled = str(WORK_DIR / f"{stem}.restyle.png")
            prompt = tools.build_restyle_prompt(axes)
            tools.restyle_to_objaverse(str(concept_path), restyled,
                                       style_prompt=prompt)
        row["restyled_path"] = restyled

        # 2. generate mesh
        mesh = str(WORK_DIR / f"{stem}.glb")
        generate(backbone, restyled, mesh)
        row["mesh_path"] = mesh

        # 3. render multi-view PNGs
        views_dir = WORK_DIR / f"{stem}.views"
        view_paths = tools.render_mesh_views(mesh, views_dir, views="default")

        # 4. CLIP
        clip_rep = tools.check_alignment(view_paths, caption)
        row.update(
            clip_mean=f"{clip_rep.score:.4f}",
            clip_accept=str(clip_rep.accept),
            worst_view=clip_rep.worst_view or "",
            per_view=json.dumps(clip_rep.per_view),
        )

        # 5. Gen3DEval (VLM-as-judge)
        gen3d = _score_gen3deval(view_paths, caption)
        row["gen3deval"] = "" if math.isnan(gen3d) else f"{gen3d:.1f}"

        # 6. ULIP-2 (optional — nan if library not installed)
        ulip = _score_ulip(view_paths, caption)
        row["ulip_mean"] = "" if math.isnan(ulip) else f"{ulip:.4f}"

        row["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
    row["seconds"] = f"{time.time() - t0:.1f}"
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="eval/results_pilot.csv", type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="max prompts to process (0 = all)")
    ap.add_argument("--backbones", nargs="+", choices=BACKENDS,
                    default=list(BACKENDS))
    args = ap.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    captions = load_captions()
    image_paths = ensure_concept_images(captions)

    filenames = sorted(image_paths)
    if args.limit:
        filenames = filenames[:args.limit]

    conds = conditions()
    done  = load_done(args.out)
    total = len(filenames) * len(conds) * len(args.backbones)
    print(f"{len(filenames)} prompts x {len(conds)} conditions x "
          f"{len(args.backbones)} backbones = {total} generations "
          f"({len(done)} already done)")

    new_file = not args.out.exists()
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        i = 0
        for fname in filenames:
            caption = captions[fname]
            concept = image_paths[fname]
            for cond_name, axes in conds:
                for backbone in args.backbones:
                    i += 1
                    key = (concept.name, cond_name, backbone)
                    if key in done:
                        print(f"[{i}/{total}] skip (done) {key}")
                        continue
                    print(f"[{i}/{total}] {fname} | {cond_name} | {backbone}")
                    row = run_one(concept, caption, cond_name, axes, backbone)
                    writer.writerow(row)
                    fh.flush()
                    print(f"    -> {row['status']}  "
                          f"clip={row['clip_mean']}  "
                          f"gen3d={row['gen3deval']}  "
                          f"ulip={row['ulip_mean']}  "
                          f"({row['seconds']}s)")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
