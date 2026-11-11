"""Mesh renderers for the two post-hoc evaluation protocols.

1. T3Bench (Ma et al. 2023, github.com/THU-LYJ-Lab/T3Bench @ 6367462): a faithful
   re-implementation of render/meshrender.py (quality) and
   render/meshrender_cap.py (captioning). Same normalization (subtract vertex
   mean, divide by max |coord| -> [-1,1]^3), same scene (ambient light 1, black
   background, base colour factor forced to white so the texture shows
   unshaded, FLAT render flag, 512x512), same camera poses (icosphere vertices,
   identical look-at / up-vector maths), same fields of view.
   The one adaptation: our meshes are glTF (+Y up) while T3Bench's threestudio
   exports are +Z up, so we rotate +90 deg about X first so "up" means the same
   thing in both. Their scripts also re-create the renderer / deep-copy the
   scene for every view; we reuse one renderer, which gives identical pixels.

2. Turntable for the Gen3DEval-protocol VLM judge (Maiti et al., CVPR 2025):
   N views evenly spaced in azimuth at one fixed elevation, as RGB (lit) and as
   camera-space normal maps (colour = (n+1)/2, flat per-face normals, so facets
   and surface noise are visible).

Headless Linux needs EGL: we default PYOPENGL_PLATFORM=egl there.
"""
from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

T3_RADIUS = 2.2
T3_QUALITY_YFOVS = (np.pi / 3.0, np.pi / 4.0, np.pi / 5.0, np.pi / 6.0, np.pi / 7.5)
T3_SIZE = 512


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_parts(mesh_path: str | Path) -> list[trimesh.Trimesh]:
    """All sub-meshes in world space (scene-graph transforms baked in), keeping
    each part's own visuals (texture or vertex colours)."""
    loaded = trimesh.load(str(mesh_path))
    parts = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded]
    parts = [p.copy() for p in parts
             if isinstance(p, trimesh.Trimesh) and len(p.faces) > 0]
    if not parts:
        raise ValueError(f"no triangle geometry in {mesh_path}")
    return parts


def _apply(parts, matrix):
    for p in parts:
        p.apply_transform(matrix)


# ---------------------------------------------------------------------------
# T3Bench protocol
# ---------------------------------------------------------------------------

def _t3_normalize(parts):
    """glTF +Y-up -> +Z-up, then T3Bench's normalization over ALL parts."""
    _apply(parts, trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    allv = np.concatenate([p.vertices for p in parts])
    mean = allv.mean(axis=0)
    scale = np.abs(allv - mean).max()
    for p in parts:
        p.vertices = (p.vertices - mean) / scale


def _t3_pose(v: np.ndarray) -> np.ndarray:
    """Camera pose exactly as T3Bench render/meshrender.py builds it."""
    f = v / np.linalg.norm(v)
    u = np.array([0, 0, 1]) if np.abs(np.dot(f, [0, 0, 1])) < 0.99 else np.array([0, 1, 0])
    r = np.cross(u, -v)
    r = r / np.linalg.norm(r)
    u_ = np.cross(-v, r)
    u_ = u_ / np.linalg.norm(u_)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([-r, u_, f], axis=1)
    pose[:3, 3] = v
    return pose


class T3Renderer:
    """Renders one mesh under the T3Bench protocol. Use as a context manager."""

    def __init__(self, mesh_path: str | Path):
        import pyrender
        self._pr = pyrender
        parts = load_parts(mesh_path)
        _t3_normalize(parts)
        self.scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
        for p in parts:
            m = pyrender.Mesh.from_trimesh(p, smooth=False)
            for prim in m.primitives:
                if prim.material is not None:
                    prim.material.baseColorFactor = [1., 1., 1., 1.]
            self.scene.add(m, pose=np.eye(4))
        self.renderer = pyrender.OffscreenRenderer(T3_SIZE, T3_SIZE)

    def _shot(self, yfov: float, pose: np.ndarray) -> Image.Image:
        node = self.scene.add(self._pr.PerspectiveCamera(yfov=yfov), pose=pose)
        try:
            color, _ = self.renderer.render(self.scene, flags=self._pr.constants.RenderFlags.FLAT)
        finally:
            self.scene.remove_node(node)
        return Image.fromarray(color.astype(np.uint8))

    def quality_views(self):
        """Yield (vertex_index, [5 PIL images]) over icosphere(subdivisions=2)."""
        ico = trimesh.creation.icosphere(subdivisions=2, radius=T3_RADIUS)
        for idx, v in enumerate(ico.vertices):
            pose = _t3_pose(v)
            yield idx, [self._shot(fov, pose) for fov in T3_QUALITY_YFOVS]

    def caption_views(self) -> list[Image.Image]:
        """12 views over icosphere(subdivisions=0). T3Bench scales the radius
        twice (radius=R, then vertices *= R), so the distance is R*R = 4.84."""
        ico = trimesh.creation.icosphere(subdivisions=0, radius=T3_RADIUS)
        verts = ico.vertices * T3_RADIUS
        return [self._shot(np.pi / 3.0, _t3_pose(v)) for v in verts]

    def close(self):
        self.renderer.delete()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def t3_icosphere():
    """The icosphere T3Bench smooths quality scores over (vertex_neighbors)."""
    return trimesh.creation.icosphere(subdivisions=2, radius=T3_RADIUS)


# ---------------------------------------------------------------------------
# Turntable (Gen3DEval-protocol judge)
# ---------------------------------------------------------------------------

def _look_at(eye, target=(0, 0, 0), up=(0, 1, 0)) -> np.ndarray:
    eye, target, up = (np.asarray(x, float) for x in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, true_up, -fwd, eye
    return pose


def render_turntable(mesh_path: str | Path, out_dir: str | Path, n_views: int = 4,
                     elevation_deg: float = 20.0, size: int = 512,
                     distance: float = 3.0) -> dict[str, list[str]]:
    """Render n_views RGB + normal-map views around +Y (glTF up), azimuth 0 =
    front (+Z). Cached: existing files are reused. Returns {"rgb": [...],
    "normal": [...]} PNG paths in azimuth order."""
    out_dir = Path(out_dir)
    names = [f"az{int(round(360 * i / n_views)):03d}" for i in range(n_views)]
    rgb_paths = [out_dir / f"rgb_{n}.png" for n in names]
    nrm_paths = [out_dir / f"normal_{n}.png" for n in names]
    if all(p.exists() for p in rgb_paths + nrm_paths):
        return {"rgb": [str(p) for p in rgb_paths], "normal": [str(p) for p in nrm_paths]}

    import pyrender
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = load_parts(mesh_path)
    allv = np.concatenate([p.vertices for p in parts])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    center, scale = (lo + hi) / 2, 2.0 / float((hi - lo).max())
    for p in parts:
        p.vertices = (p.vertices - center) * scale

    el = np.radians(elevation_deg)
    poses = []
    for i in range(n_views):
        az = 2 * np.pi * i / n_views
        eye = distance * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        poses.append(_look_at(eye))

    renderer = pyrender.OffscreenRenderer(size, size)
    try:
        # RGB: textured, ambient + camera headlight, white background
        scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[1., 1., 1., 1.])
        for p in parts:
            scene.add(pyrender.Mesh.from_trimesh(p, smooth=False))
        cam = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0))
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), parent_node=cam)
        for pose, path in zip(poses, rgb_paths):
            scene.set_pose(cam, pose)
            color, _ = renderer.render(scene)
            Image.fromarray(color).save(path)

        # Normal maps: per-face normals in CAMERA space, unlit
        flat = []
        for p in parts:
            q = trimesh.Trimesh(vertices=p.vertices[p.faces].reshape(-1, 3),
                                faces=np.arange(len(p.faces) * 3).reshape(-1, 3),
                                process=False)
            flat.append((q, np.repeat(p.face_normals, 3, axis=0)))
        for pose, path in zip(poses, nrm_paths):
            rot_w2c = pose[:3, :3].T
            nscene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=[0., 0., 0., 1.])
            for q, vn in flat:
                nc = vn @ rot_w2c.T
                q.visual = trimesh.visual.ColorVisuals(
                    q, vertex_colors=np.clip((nc + 1) / 2 * 255, 0, 255).astype(np.uint8))
                nscene.add(pyrender.Mesh.from_trimesh(q, smooth=False))
            nscene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=pose)
            color, _ = renderer.render(nscene, flags=pyrender.constants.RenderFlags.FLAT)
            Image.fromarray(color).save(path)
    finally:
        renderer.delete()
    return {"rgb": [str(p) for p in rgb_paths], "normal": [str(p) for p in nrm_paths]}
