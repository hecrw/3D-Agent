"""Client for the ULIP-2 Modal scorer (eval/ulip_modal.py).

Samples a 10k xyz+rgb point cloud from a .glb locally (trimesh works on macOS),
then POSTs it to the GPU scorer and returns the cosine similarity to the caption.
Returns nan on any failure so it never breaks the sweep.
"""
from __future__ import annotations

import io
import math
import os

import numpy as np


def _endpoint() -> str | None:
    ws = os.environ.get("TRELLIS_WORKSPACE", "")
    if not ws:
        return None
    return f"https://{ws}--ulip2-scorer-web.modal.run/score"


def sample_colored_pointcloud(glb_path: str, n: int = 10000) -> np.ndarray:
    """(n, 6) float32: xyz normalized to the unit sphere + rgb in 0..1.

    Matches ULIP-2's expected input (pc_normalize + colored 10k points).
    """
    import trimesh
    mesh = trimesh.load(glb_path, force="mesh")
    pts, face_idx = trimesh.sample.sample_surface(mesh, n)

    rgb = None
    try:
        # Bake texture -> per-vertex colors, then barycentric-interpolate to the
        # sampled surface points (vertex_colors carries the real texture color;
        # face_colors is often empty for TextureVisuals).
        vc = mesh.visual.to_color().vertex_colors[:, :3].astype(np.float32)
        faces = mesh.faces[face_idx]                  # (n, 3) vertex indices
        tri = mesh.vertices[faces]                    # (n, 3, 3)
        bary = trimesh.triangles.points_to_barycentric(tri, pts)  # (n, 3)
        rgb = np.einsum("nij,ni->nj", vc[faces], bary) / 255.0     # (n, 3)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    except Exception:  # noqa: BLE001
        rgb = None
    if rgb is None:  # untextured mesh — neutral gray
        rgb = np.full((n, 3), 0.4, dtype=np.float32)

    xyz = np.asarray(pts, dtype=np.float32)
    xyz -= xyz.mean(axis=0)
    scale = float(np.max(np.sqrt((xyz ** 2).sum(axis=1)))) + 1e-8
    xyz /= scale
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def score_ulip(glb_path: str, caption: str, timeout: int = 180) -> float:
    """ULIP-2 similarity for a mesh vs caption, via the Modal scorer. nan on failure."""
    endpoint = _endpoint()
    if not endpoint:
        return math.nan
    try:
        import requests
        pc = sample_colored_pointcloud(glb_path)
        buf = io.BytesIO()
        np.save(buf, pc)
        buf.seek(0)
        resp = requests.post(
            endpoint,
            data={"caption": caption},
            files={"pc": ("pc.npy", buf, "application/octet-stream")},
            timeout=timeout,
        )
        resp.raise_for_status()
        return float(resp.json()["ulip"])
    except Exception:  # noqa: BLE001 — ULIP is optional, never break the sweep
        return math.nan
