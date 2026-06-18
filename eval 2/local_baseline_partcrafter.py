#!/usr/bin/env python
"""Local vanilla PartCrafter baseline (no restyle) — Linux + CUDA GPU (RTX 3090).

Plain wgsxm/PartCrafter image-to-3D, NO restyle. num_parts=1 gives a single-object
mesh (fair baseline for the single-object prompts); raise it to decompose. Resumable.

SETUP (run once, ~10-15 min):
    cd ~ && git clone https://github.com/wgsxm/PartCrafter.git
    cd ~/PartCrafter
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
    pip install scikit-learn diffusers transformers einops 'huggingface_hub[hf_transfer]' \
        opencv-python-headless trimesh omegaconf scikit-image numpy==1.26.4 peft \
        jaxtyping typeguard matplotlib imageio imageio-ffmpeg pyrender colormaps \
        accelerate pillow

RUN:
    python eval/local_baseline_partcrafter.py --repo ~/PartCrafter
    python eval/score_baseline.py eval/baseline_glbs_partcrafter eval/results_baseline_partcrafter.csv
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
    ap.add_argument("--repo", default=os.path.expanduser("~/PartCrafter"))
    ap.add_argument("--images", default=str(EVAL_DIR / "dataset" / "images"))
    ap.add_argument("--out", default=str(EVAL_DIR / "baseline_glbs_partcrafter"))
    ap.add_argument("--num-parts", type=int, default=1,
                    help="1 = single-object baseline; raise to decompose into parts")
    args = ap.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        sys.exit(f"PartCrafter repo not found at {repo} — see SETUP in this file's docstring.")
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import numpy as np
    import torch
    import trimesh
    from huggingface_hub import snapshot_download
    from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
    from src.models.briarmbg import BriaRMBG
    from src.utils.data_utils import get_colored_mesh_composition
    from src.utils.image_utils import prepare_image

    print("downloading weights + loading pipeline...")
    snapshot_download("wgsxm/PartCrafter", local_dir=str(repo / "weights" / "PartCrafter"))
    snapshot_download("briaai/RMBG-1.4", local_dir=str(repo / "weights" / "RMBG-1.4"))
    rmbg_net = BriaRMBG.from_pretrained(str(repo / "weights" / "RMBG-1.4")).to("cuda").eval()
    pipe = PartCrafterPipeline.from_pretrained(
        str(repo / "weights" / "PartCrafter")).to("cuda", torch.float16)
    print("pipeline ready")

    num_parts, num_tokens, steps, guid = args.num_parts, 1024, 50, 7.0
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
            pil = prepare_image(str(p), bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
            with torch.no_grad():
                outputs = pipe(
                    image=[pil] * num_parts, attention_kwargs={"num_parts": num_parts},
                    num_tokens=num_tokens,
                    generator=torch.Generator(device=pipe.device).manual_seed(42),
                    num_inference_steps=steps, guidance_scale=guid,
                    max_num_expanded_coords=int(1e9), use_flash_decoder=False).meshes
            outputs = [m if m is not None
                       else trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
                       for m in outputs]
            get_colored_mesh_composition(outputs).export(str(out_glb))
            print(f"    -> {out_glb}")
        except Exception:
            traceback.print_exc()

    print(f"\ndone. score with:\n  python eval/score_baseline.py {out_dir} "
          f"{EVAL_DIR / 'results_baseline_partcrafter.csv'}")


if __name__ == "__main__":
    main()
