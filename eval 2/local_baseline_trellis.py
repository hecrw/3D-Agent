#!/usr/bin/env python
"""Local vanilla TRELLIS.2 baseline (no restyle) — Linux + CUDA GPU (e.g. RTX 3090).

The "theirs" baseline: plain microsoft/TRELLIS.2-4B image-to-3D on the raw input
images, with NO restyle preprocessing. Resumable (skips inputs whose .glb exists).

SETUP (run once, ~15-20 min):
    cd ~ && git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd ~/TRELLIS.2
    export TORCH_CUDA_ARCH_LIST=8.6        # 8.6 = RTX 3090/Ampere
    export MAX_JOBS=4 OPENCV_IO_ENABLE_OPENEXR=1
    . ./setup.sh --basic --flash-attn --o-voxel --cumesh --flexgemm --nvdiffrast --nvdiffrec

RUN:
    python eval/local_baseline_trellis.py --repo ~/TRELLIS.2
    # then score:
    python eval/score_baseline.py eval/baseline_glbs_trellis eval/results_baseline_trellis.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/TRELLIS.2"),
                    help="path to the cloned TRELLIS.2 repo")
    ap.add_argument("--images", default=str(EVAL_DIR / "dataset" / "images"))
    ap.add_argument("--out", default=str(EVAL_DIR / "baseline_glbs_trellis"))
    args = ap.parse_args()

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        sys.exit(f"TRELLIS.2 repo not found at {repo} — see SETUP in this file's docstring.")
    sys.path.insert(0, str(repo))
    os.chdir(repo)  # some TRELLIS modules resolve assets relative to the repo root

    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    print("loading TRELLIS.2-4B (downloads weights on first run)...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print("pipeline ready")

    img_dir = Path(args.images)
    imgs = sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(imgs)} input images -> {out_dir}")

    for i, p in enumerate(imgs, 1):
        out_glb = out_dir / f"{p.stem}.glb"
        if out_glb.exists():
            print(f"[{i}/{len(imgs)}] skip (done) {p.stem}")
            continue
        print(f"[{i}/{len(imgs)}] {p.stem} ...")
        try:
            image = Image.open(p).convert("RGB")
            mesh = pipeline.run(image)[0]
            mesh.simplify(16777216)  # nvdiffrast limit
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1000000, texture_size=4096,
                remesh=True, remesh_band=1, remesh_project=0, verbose=False)
            glb.export(str(out_glb), extension_webp=True)
            print(f"    -> {out_glb}")
        except Exception:
            traceback.print_exc()

    print(f"\ndone. score with:\n  python eval/score_baseline.py {out_dir} "
          f"{EVAL_DIR / 'results_baseline_trellis.csv'}")


if __name__ == "__main__":
    main()
