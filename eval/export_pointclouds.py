#!/usr/bin/env python
"""Export point clouds from a results CSV's meshes, for ULIP scoring on Colab.

    .venv/bin/python eval/export_pointclouds.py eval/results_retrieved.csv

Samples a 10k xyz+rgb point cloud from each ok row's mesh (locally — trimesh
works on macOS) and writes <mesh_stem>.npy + a manifest.csv into eval/pointclouds/.
Zip that folder and upload to the Colab ULIP notebook. Tiny uploads (~240 KB each)
instead of the multi-GB meshes. Resumable: existing .npy files are skipped.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
from ulip_client import pc_key, sample_colored_pointcloud  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: export_pointclouds.py <results.csv> [out_dir]")
    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else EVAL_DIR / "pointclouds"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("status") == "ok"
                and r.get("mesh_path") and Path(r["mesh_path"]).exists()]

    manifest, seen = [], {}
    for i, r in enumerate(rows, 1):
        # Key MUST be unique per mesh. The old key (mesh stem only) collided for
        # the three baselines (all named e.g. airplane_01.glb), so every
        # backbone was scored on whichever point cloud was exported first.
        stem = pc_key(r["mesh_path"])
        if stem in seen and seen[stem] != r["mesh_path"]:
            sys.exit(f"point-cloud key collision: {stem!r} for "
                     f"{seen[stem]} and {r['mesh_path']}")
        seen[stem] = r["mesh_path"]
        npy = out_dir / f"{stem}.npy"
        # re-sample if the mesh is newer than a cached cloud (stale cache guard)
        if not npy.exists() or npy.stat().st_mtime < Path(r["mesh_path"]).stat().st_mtime:
            print(f"[{i}/{len(rows)}] sampling {stem}")
            np.save(npy, sample_colored_pointcloud(r["mesh_path"]))
        manifest.append({"key": stem, "caption": r["prompt"]})

    with open(out_dir / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "caption"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\nexported {len(manifest)} point clouds to {out_dir}/")
    print(f"now zip + upload:  cd {out_dir.parent} && zip -r pointclouds.zip {out_dir.name}")


if __name__ == "__main__":
    main()
