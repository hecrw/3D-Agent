"""
Modal deployment for Hunyuan3D-Part (https://github.com/Tencent-Hunyuan/Hunyuan3D-Part).

Two sub-models from the same repo:
    P3-SAM    native 3D part segmentation       (mesh -> labelled mesh + AABBs)
    X-Part    structure-coherent decomposition  (mesh -> exploded part GLBs)

Async submit/poll, same shape as the trellis2/partcrafter/hunyuan3d-2 apps.

Routes:
    POST /segment      mesh        -> {job_id}    (P3-SAM segmentation)
    POST /decompose    mesh        -> {job_id}    (X-Part decomposition, exploded parts)
    GET  /jobs/{id}                -> 202 / 200 GLB / 5xx

Deploy:
    pip install modal
    modal setup
    modal deploy modal_app.py
"""

import modal

APP_NAME = "hunyuan3d-part"
REPO_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-Part.git"
REPO_COMMIT = "main"
SONATA_URL = "https://github.com/facebookresearch/sonata.git"

# Build for L40S/A100/H100. flash-attn + chamfer3D + sonata kernels JIT for
# the active GPU; keeping this list tight cuts build time meaningfully.
CUDA_ARCHS = "8.0;8.6;8.9;9.0"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": CUDA_ARCHS,
        "MAX_JOBS": "4",
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .apt_install(
        "git", "wget", "build-essential", "ninja-build", "clang",
        "libglib2.0-0", "libgl1", "libsm6", "libxext6", "libxrender1",
        "libegl1", "libgles2-mesa-dev",
        "ffmpeg",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install ninja packaging pybind11",
    )
    # Sonata's package-mode requirements (sparse 3D backbone used by both
    # P3-SAM and X-Part).
    .pip_install(
        "spconv-cu124",
        "huggingface_hub[hf_transfer]",
        "timm",
        "numpy<2",
    )
    .pip_install(
        "torch-scatter",
        find_links="https://data.pyg.org/whl/torch-2.5.0+cu124.html",
    )
    .run_commands(
        "python -m pip install flash-attn==2.7.3 --no-build-isolation",
    )
    # Clone + install Sonata (no PyPI release).
    # Pin setuptools<81 — Sonata's setup.py uses `pkg_resources`, which
    # setuptools 81 removed.
    .run_commands(
        f"git clone {SONATA_URL} /tmp/sonata",
        "python -m pip install --force-reinstall --no-deps 'setuptools<81' wheel",
        "python -m pip install --no-build-isolation /tmp/sonata",
    )
    # P3-SAM + X-Part Python deps (consolidated from both READMEs).
    .pip_install(
        "viser",
        "fpsample",
        "trimesh",
        "numba",
        "addict",
        "scikit-learn",
        "scikit-image",
        "pymeshlab==2023.12.post3",
        "easydict",
        "omegaconf",
        "diffusers==0.32.2",
        "transformers==4.49.0",
        "accelerate",
        "einops",
        "opencv-python-headless",
        "pygltflib",
        "xatlas",
        "tqdm>=4.66.3",
        # API surface.
        "fastapi",
        "python-multipart",
    )
    # Clone Hunyuan3D-Part + build the chamfer3D CUDA extension.
    .run_commands(
        f"git clone {REPO_URL} /app",
        f"cd /app && git checkout {REPO_COMMIT}" if REPO_COMMIT != "main" else "true",
        "cd /app/P3-SAM/utils/chamfer3D && python setup.py install",
    )
    .workdir("/app")
)

app = modal.App(APP_NAME, image=image)

hf_cache_vol = modal.Volume.from_name("hunyuan3d-part-hf-cache", create_if_missing=True)
jobs_vol = modal.Volume.from_name("hunyuan3d-part-jobs", create_if_missing=True)

JOBS_DIR = "/jobs"
HF_REPO = "tencent/Hunyuan3D-Part"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _job_paths(job_id: str):
    return (
        f"{JOBS_DIR}/{job_id}.glb",
        f"{JOBS_DIR}/{job_id}.err",
    )


async def _serve_job(job_id: str, download_name: str):
    import os
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, JSONResponse

    await jobs_vol.reload.aio()
    out_path, err_path = _job_paths(job_id)
    if os.path.exists(out_path):
        return FileResponse(
            out_path,
            media_type="model/gltf-binary",
            filename=download_name,
        )
    if os.path.exists(err_path):
        with open(err_path) as f:
            raise HTTPException(500, f.read())
    return JSONResponse(status_code=202, content={"status": "pending", "job_id": job_id})


def _write_temp_mesh(mesh_bytes: bytes, suffix: str) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(mesh_bytes)
        return tmp.name


# ---------------------------------------------------------------------------
# P3-SAM: native 3D part segmentation
# ---------------------------------------------------------------------------
@app.cls(
    gpu="L40S",
    volumes={"/cache": hf_cache_vol, JOBS_DIR: jobs_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    timeout=3600,
    max_containers=2,
)
class Segmenter:
    @modal.enter()
    def load(self):
        import sys
        from huggingface_hub import hf_hub_download

        sys.path.insert(0, "/app/P3-SAM")
        sys.path.insert(0, "/app/P3-SAM/demo")
        from auto_mask import AutoMask

        ckpt = hf_hub_download(
            repo_id=HF_REPO,
            filename="p3sam/p3sam.safetensors",
            cache_dir="/cache/huggingface",
        )
        self.auto_mask = AutoMask(ckpt)

    @modal.method()
    def _do_segment(
        self,
        mesh_bytes: bytes,
        mesh_suffix: str,
        job_id: str,
        point_num: int,
        prompt_num: int,
    ):
        import os, shutil, tempfile, traceback
        import trimesh

        out_path, err_path = _job_paths(job_id)
        tmp_path = None
        workdir = tempfile.mkdtemp(prefix=f"p3sam_{job_id}_")
        try:
            tmp_path = _write_temp_mesh(mesh_bytes, mesh_suffix)
            mesh = trimesh.load(tmp_path, force="mesh")

            self.auto_mask.predict_aabb(
                mesh,
                point_num=point_num,
                prompt_num=prompt_num,
                save_path=workdir,
            )

            # auto_mask writes many artifacts; the labelled mesh is the canonical one.
            final = os.path.join(workdir, "auto_mask_mesh_final.glb")
            if not os.path.exists(final):
                # Fall back to the post-processed variant if present.
                final = os.path.join(workdir, "auto_mask_mesh_final_post.glb")
            if not os.path.exists(final):
                raise RuntimeError(
                    f"P3-SAM produced no final mesh in {workdir}: {os.listdir(workdir)}"
                )
            shutil.copyfile(final, out_path)
        except Exception:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            shutil.rmtree(workdir, ignore_errors=True)
            jobs_vol.commit()

    @modal.asgi_app()
    def web(self):
        import os, uuid
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="Hunyuan3D-Part / P3-SAM segmentation")

        @api.get("/")
        def root():
            return {
                "service": "hunyuan3d-part-segmenter",
                "submit": "POST /segment (multipart: mesh)",
                "poll":   "GET /jobs/{job_id}",
                "supported_mesh_formats": [".glb", ".obj", ".ply", ".stl"],
            }

        @api.post("/segment")
        async def segment(
            mesh: UploadFile = File(...),
            point_num: int = Form(100000),
            prompt_num: int = Form(400),
        ):
            mesh_bytes = await mesh.read()
            suffix = os.path.splitext(mesh.filename or "")[1].lower() or ".glb"
            if not mesh_bytes:
                raise HTTPException(400, "empty mesh upload")

            job_id = uuid.uuid4().hex
            await self._do_segment.spawn.aio(
                mesh_bytes, suffix, job_id, point_num, prompt_num,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return await _serve_job(job_id, download_name=f"{job_id}.glb")

        return api


# ---------------------------------------------------------------------------
# X-Part: structure-coherent shape decomposition
# ---------------------------------------------------------------------------
@app.cls(
    gpu="L40S",
    volumes={"/cache": hf_cache_vol, JOBS_DIR: jobs_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    timeout=3600,
    max_containers=2,
)
class Decomposer:
    @modal.enter()
    def load(self):
        import sys
        import torch

        sys.path.insert(0, "/app/XPart")
        from partgen.partformer_pipeline import PartFormerPipeline

        self.pipeline = PartFormerPipeline.from_pretrained(
            model_path=HF_REPO,
            verbose=True,
        )
        self.pipeline.to(device="cuda", dtype=torch.float32)

    @modal.method()
    def _do_decompose(
        self,
        mesh_bytes: bytes,
        mesh_suffix: str,
        job_id: str,
        octree_resolution: int,
    ):
        import os, glob, shutil, tempfile, traceback

        out_path, err_path = _job_paths(job_id)
        tmp_path = None
        workdir = tempfile.mkdtemp(prefix=f"xpart_{job_id}_")
        try:
            tmp_path = _write_temp_mesh(mesh_bytes, mesh_suffix)

            # X-Part's pipeline writes its outputs as a side-effect of demo.py
            # rather than returning meshes; re-create that flow here.
            outputs = self.pipeline(
                mesh_path=tmp_path,
                octree_resolution=octree_resolution,
                output_type="trimesh",
            )
            obj_mesh, out_bbox, gt_bbox, exploded = outputs[:4]

            # The exploded variant is the most useful single artifact —
            # parts visible as separate sub-meshes in one GLB.
            exploded.export(out_path)
        except Exception:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            shutil.rmtree(workdir, ignore_errors=True)
            jobs_vol.commit()

    @modal.asgi_app()
    def web(self):
        import os, uuid
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="Hunyuan3D-Part / X-Part decomposition")

        @api.get("/")
        def root():
            return {
                "service": "hunyuan3d-part-decomposer",
                "submit": "POST /decompose (multipart: mesh)",
                "poll":   "GET /jobs/{job_id}",
                "supported_mesh_formats": [".glb", ".obj", ".ply", ".stl"],
            }

        @api.post("/decompose")
        async def decompose(
            mesh: UploadFile = File(...),
            octree_resolution: int = Form(512),
        ):
            mesh_bytes = await mesh.read()
            suffix = os.path.splitext(mesh.filename or "")[1].lower() or ".glb"
            if not mesh_bytes:
                raise HTTPException(400, "empty mesh upload")

            job_id = uuid.uuid4().hex
            await self._do_decompose.spawn.aio(
                mesh_bytes, suffix, job_id, octree_resolution,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return await _serve_job(job_id, download_name=f"{job_id}.glb")

        return api
