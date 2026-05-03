"""
Modal deployment for Hunyuan3D-2 (https://github.com/Tencent-Hunyuan/Hunyuan3D-2).

Two pipelines:
    Hunyuan3D-DiT      shape generation       ~6 GB VRAM
    Hunyuan3D-Paint    PBR texture synthesis  ~10 GB VRAM (16 GB combined)

Async submit/poll, same shape as the trellis2/partcrafter/paint3d apps.

Routes:
    POST /generate     image                    -> {job_id}    (shape + texture)
    POST /shape        image                    -> {job_id}    (shape only)
    POST /texture      image + mesh             -> {job_id}    (retexture an existing mesh)
    GET  /jobs/{id}                             -> 202 / 200 GLB / 5xx

Deploy:
    pip install modal
    modal setup
    modal deploy modal_app.py
"""

import modal

APP_NAME = "hunyuan3d-2"
REPO_URL = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
REPO_COMMIT = "main"

# A10/A10G/L4/L40S/A100/H100. Hunyuan3D-2 ships pure-Python CUDA extensions,
# so keeping the arch list tight lowers build time.
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
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # rembg / onnxruntime use this for the U2Net session.
        "U2NET_HOME": "/cache/u2net",
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
    .pip_install(
        # hy3dgen requirements (matches their setup.py install_requires).
        "diffusers==0.32.2",
        "transformers==4.49.0",
        "accelerate",
        "einops",
        "omegaconf",
        "trimesh",
        "pymeshlab",
        "pygltflib",
        "xatlas",
        "opencv-python-headless",
        "numpy<2",
        "tqdm>=4.66.3",
        "rembg",
        "onnxruntime-gpu",
        "huggingface_hub[hf_transfer]",
        # API surface.
        "fastapi",
        "python-multipart",
    )
    # Clone repo + install hy3dgen + build the two CUDA extensions.
    .run_commands(
        f"git clone {REPO_URL} /app",
        f"cd /app && git checkout {REPO_COMMIT}" if REPO_COMMIT != "main" else "true",
        "cd /app && pip install -e . --no-deps",
        "cd /app/hy3dgen/texgen/custom_rasterizer && python3 setup.py install",
        "cd /app/hy3dgen/texgen/differentiable_renderer && python3 setup.py install",
    )
    .workdir("/app")
)

app = modal.App(APP_NAME, image=image)

hf_cache_vol = modal.Volume.from_name("hunyuan3d-2-hf-cache", create_if_missing=True)
jobs_vol = modal.Volume.from_name("hunyuan3d-2-jobs", create_if_missing=True)

JOBS_DIR = "/jobs"
SHAPE_MODEL = "tencent/Hunyuan3D-2"
PAINT_MODEL = "tencent/Hunyuan3D-2"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _read_image(upload_bytes: bytes):
    import io
    from PIL import Image
    return Image.open(io.BytesIO(upload_bytes)).convert("RGBA")


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


# ---------------------------------------------------------------------------
# Image-to-3D (shape + texture)
# ---------------------------------------------------------------------------
@app.cls(
    gpu="L40S",
    volumes={"/cache": hf_cache_vol, JOBS_DIR: jobs_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    timeout=3600,
    max_containers=2,
)
class Generator:
    @modal.enter()
    def load(self):
        from hy3dgen.shapegen import (
            Hunyuan3DDiTFlowMatchingPipeline,
            FloaterRemover,
            DegenerateFaceRemover,
            FaceReducer,
        )
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from hy3dgen.rembg import BackgroundRemover

        self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            SHAPE_MODEL,
            use_safetensors=True,
            device="cuda",
        )
        self.shape.enable_flashvdm(mc_algo="mc")
        self.paint = Hunyuan3DPaintPipeline.from_pretrained(PAINT_MODEL)
        self.rembg = BackgroundRemover()
        self.floater_remover = FloaterRemover()
        self.degen_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()

    @modal.method()
    def _do_generate(
        self,
        image_bytes: bytes,
        job_id: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        octree_resolution: int,
        face_count: int,
        with_texture: bool,
        rembg: bool,
    ):
        import traceback
        import torch

        out_path, err_path = _job_paths(job_id)
        try:
            img = _read_image(image_bytes)
            if rembg:
                img = self.rembg(img)

            generator = torch.Generator(device="cuda").manual_seed(seed)
            mesh = self.shape(
                image=img,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                octree_resolution=octree_resolution,
                generator=generator,
                mc_algo="mc",
            )[0]

            if with_texture:
                mesh = self.floater_remover(mesh)
                mesh = self.degen_remover(mesh)
                mesh = self.face_reducer(mesh, max_facenum=face_count)
                mesh = self.paint(mesh, image=img)

            mesh.export(out_path)
        except Exception:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
        finally:
            jobs_vol.commit()

    @modal.asgi_app()
    def web(self):
        import uuid
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="Hunyuan3D-2 image-to-3D")

        @api.get("/")
        def root():
            return {
                "service": "hunyuan3d-2-generator",
                "endpoints": {
                    "POST /generate": "image -> textured GLB (shape + paint)",
                    "POST /shape":    "image -> untextured GLB (shape only)",
                    "GET /jobs/{id}": "poll for result",
                },
            }

        @api.post("/generate")
        async def generate(
            image: UploadFile = File(...),
            seed: int = Form(42),
            steps: int = Form(50),
            guidance_scale: float = Form(5.5),
            octree_resolution: int = Form(256),
            face_count: int = Form(40000),
            rembg: bool = Form(True),
        ):
            try:
                image_bytes = await image.read()
                _read_image(image_bytes)
            except Exception as e:
                raise HTTPException(400, f"could not decode image: {e}")

            job_id = uuid.uuid4().hex
            await self._do_generate.spawn.aio(
                image_bytes, job_id, seed, steps, guidance_scale,
                octree_resolution, face_count, True, rembg,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.post("/shape")
        async def shape(
            image: UploadFile = File(...),
            seed: int = Form(42),
            steps: int = Form(50),
            guidance_scale: float = Form(5.5),
            octree_resolution: int = Form(256),
            rembg: bool = Form(True),
        ):
            try:
                image_bytes = await image.read()
                _read_image(image_bytes)
            except Exception as e:
                raise HTTPException(400, f"could not decode image: {e}")

            job_id = uuid.uuid4().hex
            await self._do_generate.spawn.aio(
                image_bytes, job_id, seed, steps, guidance_scale,
                octree_resolution, 40000, False, rembg,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return await _serve_job(job_id, download_name=f"{job_id}.glb")

        return api


# ---------------------------------------------------------------------------
# Texture-only (retexture an arbitrary mesh)
# ---------------------------------------------------------------------------
@app.cls(
    gpu="L40S",
    volumes={"/cache": hf_cache_vol, JOBS_DIR: jobs_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    timeout=3600,
    max_containers=2,
)
class Texturer:
    @modal.enter()
    def load(self):
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from hy3dgen.rembg import BackgroundRemover

        self.paint = Hunyuan3DPaintPipeline.from_pretrained(PAINT_MODEL)
        self.rembg = BackgroundRemover()

    @modal.method()
    def _do_texture(
        self,
        image_bytes: bytes,
        mesh_bytes: bytes,
        mesh_suffix: str,
        job_id: str,
        rembg: bool,
    ):
        import os, tempfile, traceback
        import trimesh

        out_path, err_path = _job_paths(job_id)
        tmp_path = None
        try:
            img = _read_image(image_bytes)
            if rembg:
                img = self.rembg(img)

            with tempfile.NamedTemporaryFile(suffix=mesh_suffix, delete=False) as tmp:
                tmp.write(mesh_bytes)
                tmp_path = tmp.name

            mesh = trimesh.load(tmp_path, force="mesh")
            mesh = self.paint(mesh, image=img)
            mesh.export(out_path)
        except Exception:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            jobs_vol.commit()

    @modal.asgi_app()
    def web(self):
        import os, uuid
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="Hunyuan3D-2 PBR texturing")

        @api.get("/")
        def root():
            return {
                "service": "hunyuan3d-2-texturer",
                "submit": "POST /texture (multipart: image, mesh)",
                "poll":   "GET /jobs/{job_id}",
                "supported_mesh_formats": [".glb", ".obj", ".ply", ".stl"],
            }

        @api.post("/texture")
        async def texture(
            image: UploadFile = File(...),
            mesh: UploadFile = File(...),
            rembg: bool = Form(True),
        ):
            try:
                image_bytes = await image.read()
                _read_image(image_bytes)
            except Exception as e:
                raise HTTPException(400, f"could not decode image: {e}")

            mesh_bytes = await mesh.read()
            suffix = os.path.splitext(mesh.filename or "")[1].lower() or ".glb"

            job_id = uuid.uuid4().hex
            await self._do_texture.spawn.aio(
                image_bytes, mesh_bytes, suffix, job_id, rembg,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return await _serve_job(job_id, download_name=f"{job_id}.glb")

        return api
