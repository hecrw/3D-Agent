"""
Modal deployment for PartCrafter (https://github.com/wgsxm/PartCrafter).

Mirrors the TRELLIS.2 deployment: async job pattern, each job returns a
single composite GLB (all parts baked into one mesh with per-part colors).

Routes:
    POST /generate         image -> {job_id}   (object,  1-16 parts)
    POST /generate-scene   image -> {job_id}   (scene,   1-16 parts)
    GET  /jobs/{id}                            -> 202 pending / 200 GLB / 5xx

Deploy:
    pip install modal
    modal setup
    modal deploy partcrafter_modal.py
"""

import modal

APP_NAME = "partcrafter"
REPO_URL = "https://github.com/wgsxm/PartCrafter.git"
REPO_COMMIT = "main"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PYOPENGL_PLATFORM": "egl",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .apt_install(
        "git", "wget", "build-essential",
        "libegl1", "libegl1-mesa", "libgl1-mesa-dev",
        "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "ffmpeg",
    )
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "torch-cluster",
        find_links="https://data.pyg.org/whl/torch-2.5.1+cu124.html",
    )
    .pip_install(
        "scikit-learn", "diffusers", "transformers", "einops",
        "huggingface_hub[hf_transfer]", "opencv-python-headless",
        "trimesh", "omegaconf", "scikit-image", "numpy==1.26.4",
        "peft", "jaxtyping", "typeguard", "matplotlib",
        "imageio", "imageio-ffmpeg", "pyrender", "colormaps",
        "accelerate", "pillow",
        "fastapi", "python-multipart",
    )
    .run_commands(
        f"git clone {REPO_URL} /app",
        f"cd /app && git checkout {REPO_COMMIT}" if REPO_COMMIT != "main" else "true",
    )
    .workdir("/app")
)

app = modal.App(APP_NAME, image=image)

hf_cache_vol = modal.Volume.from_name("partcrafter-hf-cache", create_if_missing=True)
jobs_vol = modal.Volume.from_name("partcrafter-jobs", create_if_missing=True)

JOBS_DIR = "/jobs"
WEIGHTS_DIR = "/cache/partcrafter_weights"


# ---------------------------------------------------------------------------
# Helpers (match TRELLIS layout)
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


def _serve_job(job_id: str, download_name: str):
    import os
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, JSONResponse

    jobs_vol.reload()
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


def _run_pipeline(
    pipe, rmbg_net, image_bytes, job_id, num_parts, seed,
    num_tokens, num_inference_steps, guidance_scale, rmbg,
):
    """Shared worker body for object + scene."""
    import io, os, tempfile, traceback
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from accelerate.utils import set_seed

    import sys; sys.path.insert(0, "/app")
    from src.utils.data_utils import get_colored_mesh_composition
    from src.utils.image_utils import prepare_image

    out_path, err_path = _job_paths(job_id)
    tmpdir = tempfile.mkdtemp(prefix=f"pc_{job_id}_")
    try:
        set_seed(seed)
        img_path = os.path.join(tmpdir, "input.png")
        Image.open(io.BytesIO(image_bytes)).convert("RGB").save(img_path)

        if rmbg:
            pil = prepare_image(
                img_path,
                bg_color=np.array([1.0, 1.0, 1.0]),
                rmbg_net=rmbg_net,
            )
        else:
            pil = Image.open(img_path)

        with torch.no_grad():
            outputs = pipe(
                image=[pil] * num_parts,
                attention_kwargs={"num_parts": num_parts},
                num_tokens=num_tokens,
                generator=torch.Generator(device=pipe.device).manual_seed(seed),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                max_num_expanded_coords=int(1e9),
                use_flash_decoder=False,
            ).meshes

        for i, m in enumerate(outputs):
            if m is None:
                outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])

        composite = get_colored_mesh_composition(outputs)
        composite.export(out_path)
    except Exception:
        with open(err_path, "w") as f:
            f.write(traceback.format_exc())
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        jobs_vol.commit()


# ---------------------------------------------------------------------------
# Object generator
# ---------------------------------------------------------------------------
@app.cls(
    gpu="L4",
    volumes={"/cache": hf_cache_vol, JOBS_DIR: jobs_vol},
    scaledown_window=300,
    timeout=3600,
    max_containers=2,
)
class ObjectGenerator:
    @modal.enter()
    def load(self):
        import sys; sys.path.insert(0, "/app")
        import torch
        from huggingface_hub import snapshot_download
        from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
        from src.models.briarmbg import BriaRMBG

        pc_dir = f"{WEIGHTS_DIR}/PartCrafter"
        rmbg_dir = f"{WEIGHTS_DIR}/RMBG-1.4"
        snapshot_download("wgsxm/PartCrafter", local_dir=pc_dir)
        snapshot_download("briaai/RMBG-1.4", local_dir=rmbg_dir)
        hf_cache_vol.commit()

        self.rmbg_net = BriaRMBG.from_pretrained(rmbg_dir).to("cuda").eval()
        self.pipe = PartCrafterPipeline.from_pretrained(pc_dir).to("cuda", torch.float16)

    @modal.method()
    def _do_generate(self, image_bytes, job_id, num_parts, seed,
                     num_tokens, num_inference_steps, guidance_scale, rmbg):
        _run_pipeline(
            self.pipe, self.rmbg_net, image_bytes, job_id, num_parts, seed,
            num_tokens, num_inference_steps, guidance_scale, rmbg,
        )

    @modal.asgi_app()
    def web(self):
        import uuid
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="PartCrafter object generator")

        @api.get("/")
        def root():
            return {
                "service": "partcrafter-object",
                "submit": "POST /generate (multipart: image; form: num_parts 1-16)",
                "poll": "GET /jobs/{job_id}",
            }

        @api.post("/generate")
        async def generate(
            image: UploadFile = File(...),
            num_parts: int = Form(...),
            seed: int = Form(0),
            num_tokens: int = Form(1024),
            num_inference_steps: int = Form(50),
            guidance_scale: float = Form(7.0),
            rmbg: bool = Form(True),
        ):
            if not (1 <= num_parts <= 16):
                raise HTTPException(400, "num_parts must be in [1, 16]")
            try:
                image_bytes = await image.read()
                _read_image(image_bytes)
            except Exception as e:
                raise HTTPException(400, f"could not decode image: {e}")

            job_id = uuid.uuid4().hex
            self._do_generate.spawn(
                image_bytes, job_id, num_parts, seed,
                num_tokens, num_inference_steps, guidance_scale, rmbg,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return _serve_job(job_id, download_name=f"{job_id}.glb")

        return api


# ---------------------------------------------------------------------------
# Scene generator
# ---------------------------------------------------------------------------
@app.cls(
    gpu="L4",
    volumes={"/cache": hf_cache_vol, JOBS_DIR: jobs_vol},
    scaledown_window=300,
    timeout=3600,
    max_containers=2,
)
class SceneGenerator:
    @modal.enter()
    def load(self):
        import sys; sys.path.insert(0, "/app")
        import torch
        from huggingface_hub import snapshot_download
        from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
        from src.models.briarmbg import BriaRMBG

        pc_dir = f"{WEIGHTS_DIR}/PartCrafter-Scene"
        rmbg_dir = f"{WEIGHTS_DIR}/RMBG-1.4"
        snapshot_download("wgsxm/PartCrafter-Scene", local_dir=pc_dir)
        snapshot_download("briaai/RMBG-1.4", local_dir=rmbg_dir)
        hf_cache_vol.commit()

        self.rmbg_net = BriaRMBG.from_pretrained(rmbg_dir).to("cuda").eval()
        self.pipe = PartCrafterPipeline.from_pretrained(pc_dir).to("cuda", torch.float16)

    @modal.method()
    def _do_generate(self, image_bytes, job_id, num_parts, seed,
                     num_tokens, num_inference_steps, guidance_scale, rmbg):
        _run_pipeline(
            self.pipe, self.rmbg_net, image_bytes, job_id, num_parts, seed,
            num_tokens, num_inference_steps, guidance_scale, rmbg,
        )

    @modal.asgi_app()
    def web(self):
        import uuid
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="PartCrafter scene generator")

        @api.get("/")
        def root():
            return {
                "service": "partcrafter-scene",
                "submit": "POST /generate-scene (multipart: image; form: num_parts 1-16)",
                "poll": "GET /jobs/{job_id}",
            }

        @api.post("/generate-scene")
        async def generate_scene(
            image: UploadFile = File(...),
            num_parts: int = Form(...),
            seed: int = Form(0),
            num_tokens: int = Form(1024),
            num_inference_steps: int = Form(150),
            guidance_scale: float = Form(7.0),
            rmbg: bool = Form(True),            
        ):
            if not (1 <= num_parts <= 16):
                raise HTTPException(400, "num_parts must be in [1, 16]")
            try:
                image_bytes = await image.read()
                _read_image(image_bytes)
            except Exception as e:
                raise HTTPException(400, f"could not decode image: {e}")

            job_id = uuid.uuid4().hex
            self._do_generate.spawn(
                image_bytes, job_id, num_parts, seed,
                num_tokens, num_inference_steps, guidance_scale, rmbg,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return _serve_job(job_id, download_name=f"{job_id}.glb")

        return api
