#!/usr/bin/env python
"""Local ULIP-2 scoring (point-cloud <-> text) — Linux + CUDA GPU (RTX 3090).

Scores each exported point cloud against its caption with ULIP-2 colored PointBERT.
Self-contained: clones + patches the ULIP repo if missing, loads the model, scores
eval/pointclouds/*.npy, and writes eval/ulip_scores.csv. Then merge:
    python eval/merge_ulip.py eval/results_pilot.csv eval/ulip_scores.csv

Prereq: export the point clouds first (CPU, trimesh):
    python eval/export_pointclouds.py eval/results_pilot.csv

SETUP (run once):
    pip install open_clip_torch==2.24.0 timm easydict pyyaml 'huggingface_hub[hf_transfer]'
    # (the ULIP repo itself is auto-cloned by this script)

The only CUDA op ULIP needs (furthest-point-sampling) is swapped for a pure-torch
version, so there is nothing to build — no pointnet2_ops, no knn_cuda, no open3d.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
import types

EVAL_DIR = pathlib.Path(__file__).resolve().parent


def ensure_ulip(ulip_dir: pathlib.Path) -> None:
    """Clone ULIP and make its CUDA-only / unused imports optional (idempotent)."""
    if not ulip_dir.exists():
        print(f"cloning ULIP -> {ulip_dir}")
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/salesforce/ULIP.git", str(ulip_dir)],
                       check=True)

    d = ulip_dir / "models/pointbert/dvae.py"
    s = d.read_text()
    if "KNN=None" not in s:
        s = s.replace("from knn_cuda import KNN",
                      "try:\n    from knn_cuda import KNN\nexcept Exception:\n    KNN=None")
        s = s.replace("knn = KNN(k=4, transpose_mode=False)", "knn = None")
        d.write_text(s)

    m = ulip_dir / "models/pointbert/misc.py"
    s = m.read_text()
    if "pointnet2_utils=None" not in s:
        s = s.replace("from pointnet2_ops import pointnet2_utils",
                      "try:\n    from pointnet2_ops import pointnet2_utils\n"
                      "except Exception:\n    pointnet2_utils=None")
        m.write_text(s)

    io = ulip_dir / "utils/io.py"
    s = io.read_text()
    if "open3d=None" not in s:
        s = s.replace("import open3d",
                      "try:\n    import open3d\nexcept Exception:\n    open3d=None")
        io.write_text(s)


def load_model(ulip_dir: pathlib.Path):
    import torch

    # torch._six was removed in PyTorch 1.9; ULIP still imports string_classes from it.
    if "torch._six" not in sys.modules:
        six = types.ModuleType("torch._six")
        six.string_classes = (str,)
        six.inf = float("inf")
        sys.modules["torch._six"] = six

    sys.path.insert(0, str(ulip_dir))
    os.chdir(ulip_dir)  # config yaml path is relative to here
    from easydict import EasyDict
    from huggingface_hub import hf_hub_download
    from models.ULIP_models import ULIP2_PointBERT_Colored
    import models.pointbert.misc as _misc

    def _fps(data, number):
        """Pure-torch furthest point sampling (replaces pointnet2_ops). data: (B,N,C)."""
        B, N, C = data.shape
        xyz = data[:, :, :3]
        dev = data.device
        centroids = torch.zeros(B, number, dtype=torch.long, device=dev)
        distance = torch.ones(B, N, device=dev) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=dev)
        batch = torch.arange(B, dtype=torch.long, device=dev)
        for i in range(number):
            centroids[:, i] = farthest
            c = xyz[batch, farthest, :].view(B, 1, 3)
            dist = ((xyz - c) ** 2).sum(-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = distance.max(-1)[1]
        return torch.gather(data, 1, centroids.unsqueeze(-1).expand(-1, -1, C))

    _misc.fps = _fps
    print("FPS patched (pure-torch, no pointnet2_ops)")

    model = ULIP2_PointBERT_Colored(EasyDict(evaluate_3d=True, npoints=10000)).cuda().eval()
    ckpt = hf_hub_download(
        repo_id="SFXX/ulip", repo_type="dataset",
        filename="ULIP-2/pretrained_models/"
                 "ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)  # ckpt has numpy scalars
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    rep = model.load_state_dict(sd, strict=False)
    print(f"loaded; missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pointclouds", default=str(EVAL_DIR / "pointclouds"),
                    help="dir with <key>.npy files + manifest.csv (from export_pointclouds.py)")
    ap.add_argument("--ulip-dir", default=os.path.expanduser("~/ULIP"))
    ap.add_argument("--out", default=str(EVAL_DIR / "ulip_scores.csv"))
    args = ap.parse_args()

    pc_dir = pathlib.Path(args.pointclouds)
    manifest = pc_dir / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found — run export_pointclouds.py first.")

    ensure_ulip(pathlib.Path(args.ulip_dir).resolve())
    import numpy as np
    import torch
    model = load_model(pathlib.Path(args.ulip_dir).resolve())

    rows = list(csv.DictReader(open(manifest)))
    print(f"{len(rows)} point clouds to score")
    out_path = pathlib.Path(args.out)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "ulip"])
        w.writeheader()
        for i, r in enumerate(rows, 1):
            key, caption = r["key"], r["caption"]
            try:
                pc = torch.as_tensor(np.load(pc_dir / f"{key}.npy"),
                                     dtype=torch.float32, device="cuda").unsqueeze(0)
                with torch.no_grad():
                    pe = model.encode_pc(pc)
                    pe = pe / pe.norm(dim=-1, keepdim=True)
                    te = model.encode_text(model.tokenizer([caption]).cuda())
                    te = te / te.norm(dim=-1, keepdim=True)
                    sim = float((pe @ te.T).item())
                w.writerow({"key": key, "ulip": f"{sim:.4f}"})
                print(f"[{i}/{len(rows)}] {key}: {sim:.4f}")
            except Exception as e:
                print(f"[{i}/{len(rows)}] {key}: FAILED {e}")
                w.writerow({"key": key, "ulip": ""})

    print(f"\nwrote {out_path}. merge with:\n"
          f"  python eval/merge_ulip.py <results.csv> {out_path}")


if __name__ == "__main__":
    main()
