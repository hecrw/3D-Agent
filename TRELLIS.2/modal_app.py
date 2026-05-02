"""
Modal deployment for TRELLIS.2.

Async job pattern — endpoints return a job_id immediately and you poll for
the GLB. This is the only reliable shape for long generations (>~5 min)
because Modal's web proxy caps open HTTP connections around that mark.

Routes (on each service):
    POST /generate   (Generator)  image            -> {job_id}
    POST /texture    (Texturer)   image + mesh     -> {job_id}
    GET  /jobs/{id}                                -> 202 pending / 200 GLB / 4xx-5xx

Deploy:
    pip install modal
    modal setup                  # first time only
    modal deploy modal_app.py
"""

import modal

APP_NAME = "trellis2"

# GPU architectures to compile CUDA extensions for:
#   8.0 = A100, 8.6 = A10/A10G/A40, 8.9 = L4/L40, 9.0 = H100
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
        "OPENCV_IO_ENABLE_OPENEXR": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .apt_install(
        "git", "wget", "build-essential", "clang",
        "libglib2.0-0", "libgl1", "libsm6", "libxext6", "libxrender1",
        "libjpeg-dev", "ffmpeg",
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install packaging ninja psutil einops",
    )
    .pip_install(
        "imageio", "imageio-ffmpeg", "tqdm", "easydict",
        "opencv-python-headless", "trimesh", "transformers==4.57.3",
        "zstandard", "kornia", "timm", "numpy<2",
        "huggingface_hub[hf_transfer]",
        "rembg", "onnxruntime-gpu",
        "fastapi", "python-multipart",
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
    )
    .run_commands(
        "python -m pip install wheel packaging && "
        "python -m pip install flash-attn==2.7.3 --no-build-isolation"
    )
    .run_commands(
        "mkdir -p /tmp/extensions",
        "git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast",
        "pip install /tmp/extensions/nvdiffrast --no-build-isolation",
        "git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec",
        "pip install /tmp/extensions/nvdiffrec --no-build-isolation",
        "git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh",
        "pip install /tmp/extensions/CuMesh --no-build-isolation",
        "git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM",
        "pip install /tmp/extensions/FlexGEMM --no-build-isolation",
    )
    .add_local_dir(
        ".",
        "/app",
        copy=True,
        ignore=[
            ".git/**", "tmp/**", "results/**", "datasets/**",
            "__pycache__", "**/__pycache__", "*.pyc",
            "*.mp4", "*.glb", "modal_app.py",
        ],
    )
    .run_commands(
        "rm -rf /app/o-voxel/third_party/eigen && "
        "git clone --depth 1 https://gitlab.com/libeigen/eigen.git "
        "/app/o-voxel/third_party/eigen",
        "pip install /app/o-voxel --no-build-isolation",
    )
    .workdir("/app")
)

app = modal.App(APP_NAME, image=image)

# HF weights persist across cold starts.
hf_cache_vol = modal.Volume.from_name("trellis2-hf-cache", create_if_missing=True)
# Per-job scratch space shared between worker + API containers.
jobs_vol = modal.Volume.from_name("trellis2-jobs", create_if_missing=True)

HDRI_PATH = "/app/assets/hdri/forest.exr"
JOBS_DIR = "/jobs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_image(upload_bytes: bytes):
    import io
    from PIL import Image
    return Image.open(io.BytesIO(upload_bytes))


def _job_paths(job_id: str):
    return (
        f"{JOBS_DIR}/{job_id}.glb",
        f"{JOBS_DIR}/{job_id}.err",
    )


async def _serve_job(job_id: str, download_name: str):
    """Shared GET /jobs/{id} handler body."""
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
            err = f.read()
        raise HTTPException(500, err)
    return JSONResponse(status_code=202, content={"status": "pending", "job_id": job_id})


# ---------------------------------------------------------------------------
# Image-to-3D generator
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
        import cv2, torch
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.renderers import EnvMap

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B"
        )
        self.pipeline.cuda()

        exr = cv2.cvtColor(
            cv2.imread(HDRI_PATH, cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB,
        )
        self.envmap = EnvMap(
            torch.tensor(exr, dtype=torch.float32, device="cuda")
        )

    @modal.method()
    def _do_generate(
        self,
        image_bytes: bytes,
        job_id: str,
        seed: int,
        pipeline_type: str,
        decimation_target: int,
        texture_size: int,
        remesh: bool,
    ):
        import traceback
        import o_voxel

        out_path, err_path = _job_paths(job_id)
        try:
            img = _read_image(image_bytes)
            mesh = self.pipeline.run(
                img, seed=seed, pipeline_type=pipeline_type
            )[0]
            mesh.simplify(16_777_216)

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=remesh,
                remesh_band=1,
                remesh_project=0,
                verbose=True,
            )
            glb.export(out_path, extension_webp=True)
        except Exception:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
        finally:
            jobs_vol.commit()

    @modal.asgi_app()
    def web(self):
        import uuid
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="TRELLIS.2 image-to-3D")

        @api.get("/")
        def root():
            return {
                "service": "trellis2-generator",
                "submit": "POST /generate (multipart: image)",
                "poll": "GET /jobs/{job_id}",
                "pipeline_types": ["512", "1024", "1024_cascade", "1536_cascade"],
            }

        @api.post("/generate")
        async def generate(
            image: UploadFile = File(...),
            seed: int = Form(42),
            pipeline_type: str = Form("1024_cascade"),
            decimation_target: int = Form(1_000_000),
            texture_size: int = Form(4096),
            remesh: bool = Form(True),
        ):
            try:
                image_bytes = await image.read()
                _read_image(image_bytes)  # validate
            except Exception as e:
                raise HTTPException(400, f"could not decode image: {e}")

            job_id = uuid.uuid4().hex
            await self._do_generate.spawn.aio(
                image_bytes,
                job_id,
                seed,
                pipeline_type,
                decimation_target,
                texture_size,
                remesh,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return await _serve_job(job_id, download_name=f"{job_id}.glb")

        return api


# ---------------------------------------------------------------------------
# Texturer (shape-conditioned PBR texture generation)
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
        from trellis2.pipelines import Trellis2TexturingPipeline

        self.pipeline = Trellis2TexturingPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B",
            config_file="texturing_pipeline.json",
        )
        self.pipeline.cuda()

    @modal.method()
    def _do_texture(
        self,
        image_bytes: bytes,
        mesh_bytes: bytes,
        mesh_suffix: str,
        job_id: str,
        seed: int,
        resolution: int,
        texture_size: int,
    ):
        import os, tempfile, traceback
        import trimesh

        out_path, err_path = _job_paths(job_id)
        try:
            img = _read_image(image_bytes)
            with tempfile.NamedTemporaryFile(suffix=mesh_suffix, delete=False) as tmp:
                tmp.write(mesh_bytes)
                mesh_path = tmp.name
            try:
                tm = trimesh.load(mesh_path, force="mesh")
                out = self.pipeline.run(
                    tm, img,
                    seed=seed,
                    resolution=resolution,
                    texture_size=texture_size,
                )
                out.export(out_path, extension_webp=True)
            finally:
                if os.path.exists(mesh_path):
                    os.unlink(mesh_path)
        except Exception:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
        finally:
            jobs_vol.commit()

    @modal.asgi_app()
    def web(self):
        import os, uuid
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="TRELLIS.2 PBR texturing")

        @api.get("/")
        def root():
            return {
                "service": "trellis2-texturer",
                "submit": "POST /texture (multipart: image, mesh)",
                "poll": "GET /jobs/{job_id}",
                "supported_mesh_formats": [".ply", ".obj", ".glb", ".stl"],
            }

        @api.post("/texture")
        async def texture(
            image: UploadFile = File(...),
            mesh: UploadFile = File(...),
            seed: int = Form(42),
            resolution: int = Form(1024),
            texture_size: int = Form(2048),
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
                image_bytes,
                mesh_bytes,
                suffix,
                job_id,
                seed,
                resolution,
                texture_size,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return await _serve_job(job_id, download_name=f"{job_id}.glb")

        return api
