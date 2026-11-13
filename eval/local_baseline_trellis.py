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
    # resolve BEFORE chdir: relative --images/--out would otherwise point inside the repo
    args.images = str(Path(args.images).resolve())
    args.out = str(Path(args.out).resolve())
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

    # Watchdog: some inputs send TRELLIS.2's GLB export (cumesh remeshing) into a
    # pathological slow path that never returns. Mark the image in progress; if
    # it runs past the limit the process exits (code 3), and on the restart
    # (rerun_all.sh loops) that image is recorded in failed.txt and skipped.
    import threading
    limit = float(os.environ.get("TRELLIS_GEN_TIMEOUT", "480"))
    marker, failed_f = out_dir / ".inprogress", out_dir / "failed.txt"
    if marker.exists():
        stem = marker.read_text().strip()
        if not (out_dir / f"{stem}.glb").exists():
            with open(failed_f, "a") as fh:
                fh.write(f"{stem}\ttimeout>{limit:.0f}s\n")
            print(f"recorded {stem} as failed (exceeded {limit:.0f} s)")
        marker.unlink()
    failed = {l.split("\t")[0] for l in failed_f.read_text().splitlines()} if failed_f.exists() else set()

    for i, p in enumerate(imgs, 1):
        out_glb = out_dir / f"{p.stem}.glb"
        if out_glb.exists():
            print(f"[{i}/{len(imgs)}] skip (done) {p.stem}")
            continue
        if p.stem in failed:
            print(f"[{i}/{len(imgs)}] skip (failed before) {p.stem}")
            continue
        print(f"[{i}/{len(imgs)}] {p.stem} ...", flush=True)
        marker.write_text(p.stem)
        dog = threading.Timer(limit, lambda: (print(f"[watchdog] {p.stem} exceeded {limit:.0f} s — exiting", flush=True), os._exit(3)))
        dog.daemon = True
        dog.start()
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
        except Exception as e:
            traceback.print_exc()
            with open(failed_f, "a") as fh:
                fh.write(f"{p.stem}\terror:{type(e).__name__}\n")
        finally:
            dog.cancel()
            marker.unlink(missing_ok=True)

    print(f"\ndone. score with:\n  python eval/score_baseline.py {out_dir} "
          f"{EVAL_DIR / 'results_baseline_trellis.csv'}")


if __name__ == "__main__":
    main()
