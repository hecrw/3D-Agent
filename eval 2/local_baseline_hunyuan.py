#!/usr/bin/env python
"""Local vanilla Hunyuan3D-2 baseline (no restyle) — Linux + CUDA GPU (RTX 3090).

Plain tencent/Hunyuan3D-2 image-to-3D (shape + PBR paint), NO restyle. The paint
pass needs ~16 GB VRAM, which fits on a 3090. Resumable.

SETUP (run once, ~15-20 min):
    cd ~ && git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
    cd ~/Hunyuan3D-2
    export TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4      # 8.6 = RTX 3090/Ampere
    pip install diffusers==0.32.2 transformers==4.49.0 accelerate einops omegaconf \
        trimesh pymeshlab pygltflib xatlas opencv-python-headless 'numpy<2' \
        'tqdm>=4.66.3' rembg onnxruntime-gpu 'huggingface_hub[hf_transfer]'
    pip install -e . --no-deps
    cd hy3dgen/texgen/custom_rasterizer && python3 setup.py install && cd -
    cd hy3dgen/texgen/differentiable_renderer && python3 setup.py install && cd -

RUN:
    python eval/local_baseline_hunyuan.py --repo ~/Hunyuan3D-2
    python eval/score_baseline.py eval/baseline_glbs_hunyuan eval/results_baseline_hunyuan.csv
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
    ap.add_argument("--repo", default=os.path.expanduser("~/Hunyuan3D-2"))
    ap.add_argument("--images", default=str(EVAL_DIR / "dataset" / "images"))
    ap.add_argument("--out", default=str(EVAL_DIR / "baseline_glbs_hunyuan"))
    ap.add_argument("--no-texture", action="store_true",
                    help="geometry only (much faster, less VRAM)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        sys.exit(f"Hunyuan3D-2 repo not found at {repo} — see SETUP in this file's docstring.")
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from PIL import Image
    import torch
    from hy3dgen.shapegen import (Hunyuan3DDiTFlowMatchingPipeline,
                                  FloaterRemover, DegenerateFaceRemover, FaceReducer)
    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.rembg import BackgroundRemover

    print("loading Hunyuan3D-2 (downloads weights on first run)...")
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2", use_safetensors=True, device="cuda")
    shape.enable_flashvdm(mc_algo="mc")
    rembg = BackgroundRemover()
    with_texture = not args.no_texture
    if with_texture:
        paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
        floater, degen, reducer = FloaterRemover(), DegenerateFaceRemover(), FaceReducer()
    print("pipelines ready")

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
            img = rembg(Image.open(p).convert("RGB"))
            gen = torch.Generator(device="cuda").manual_seed(42)
            mesh = shape(image=img, num_inference_steps=50, octree_resolution=256,
                         guidance_scale=5.0, generator=gen, mc_algo="mc")[0]
            if with_texture:
                mesh = floater(mesh)
                mesh = degen(mesh)
                mesh = reducer(mesh, max_facenum=40000)
                mesh = paint(mesh, image=img)
            mesh.export(str(out_glb))
            print(f"    -> {out_glb}")
        except Exception:
            traceback.print_exc()

    print(f"\ndone. score with:\n  python eval/score_baseline.py {out_dir} "
          f"{EVAL_DIR / 'results_baseline_hunyuan.csv'}")


if __name__ == "__main__":
    main()
