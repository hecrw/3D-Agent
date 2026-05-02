"""
Modal deployment for Paint3D (https://github.com/OpenTexture/Paint3D).

Paint3D paints a high-resolution, lighting-less UV texture map onto an
*untextured* mesh given a text prompt (and optionally an IP-adapter image).
Two-stage: coarse multi-view fusion -> UV inpaint/illumination refine.

Async submit/poll, same shape as your trellis2/threestudio/etc apps.

Routes:
    POST /paint                multipart: mesh, prompt, [ip_image]   -> {job_id}
    GET  /jobs/{id}                              -> 202 / 200 .glb / 5xx

Deploy:
    pip install modal
    modal setup
    modal secret create huggingface-secret HF_TOKEN=hf_xxx   # if not done
    modal deploy modal_app.py
"""

import modal

APP_NAME = "paint3d"
REPO_URL = "https://github.com/OpenTexture/Paint3D.git"
REPO_COMMIT = "main"

# T4 / A100 / A10/L4/L40S — H100 (sm_90) JITs from PTX.
CUDA_ARCHS = "7.5 8.0 8.6 8.9+PTX"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.6.2-devel-ubuntu20.04",
        add_python="3.10",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": CUDA_ARCHS,
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "MAX_JOBS": "4",
        "HF_HOME": "/hf_cache",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TRANSFORMERS_CACHE": "/hf_cache",
        "DEBIAN_FRONTEND": "noninteractive",
        "TZ": "Etc/UTC",
    })
    .run_commands(
        "ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime",
        "echo 'Etc/UTC' > /etc/timezone",
    )
    .apt_install(
        "tzdata",
        "git", "wget", "curl", "build-essential", "ninja-build",
        "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "libegl1", "ffmpeg",
        "clang",
    )
    # PyTorch 1.12.1 + cu116 — Paint3D's pinned baseline.
    .run_commands(
        "pip install --upgrade pip setuptools==69.5.1 wheel ninja packaging",
        "pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 "
        "torchaudio==0.12.1+cu116 "
        "--extra-index-url https://download.pytorch.org/whl/cu116",
    )
    # Kaolin 0.13.0. NVIDIA's precompiled wheels at
    # nvidia-kaolin.s3.us-east-2.amazonaws.com only cover cp37/cp38/cp39 for
    # this torch/cuda combo — Python 3.10 has no wheel, so we build from
    # source. ~5-8 min compile, then cached in the layer.
    .run_commands(
        "pip install cython",
        "pip install git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.13.0 "
        "--no-build-isolation",
    )
    # Python deps from environment.yaml's pip section, pinned.
    .run_commands(
        "pip install "
        "albumentations==1.3.0 opencv-python-headless==4.6.0.66 "
        "imageio==2.9.0 imageio-ffmpeg==0.4.2 "
        "pytorch-lightning==1.4.2 omegaconf==2.1.1 "
        "einops==0.3.0 transformers==4.27.1 "
        "kornia==0.6.12 open_clip_torch==2.0.2 "
        "torchmetrics==0.6.0 diffusers==0.25.0 accelerate==0.29.2 "
        "loguru==0.7.2 trimesh==3.20.2 xatlas==0.0.7 "
        "huggingface_hub==0.22.2 hf_transfer "
        "test-tube webdataset==0.2.5 invisible-watermark "
        "fastapi 'pydantic>=1.10,<2' python-multipart",
    )
    # Clone Paint3D at build time.
    .run_commands(
        f"git clone {REPO_URL} /app",
        f"cd /app && git checkout {REPO_COMMIT}",
    )
    # Pre-fetch the UV-position ControlNet (Paint3D's only project-specific
    # checkpoint). Standard ControlNets pull lazily on first job.
    # HF_HOME override + no symlinks: keep blobs out of /hf_cache (which is a
    # runtime volume mount — non-empty mount target = boot failure).
    .run_commands(
        "HF_HOME=/tmp/hfbuild huggingface-cli download GeorgeQi/Paint3d_UVPos_Control "
        "--local-dir /app/controlnet/UV_Pos_Control "
        "--local-dir-use-symlinks False || true",
        "rm -rf /tmp/hfbuild /hf_cache",
    )
    .workdir("/app")
)

app = modal.App(APP_NAME, image=image)

hf_cache_vol = modal.Volume.from_name("paint3d-hf-cache", create_if_missing=True)
jobs_vol = modal.Volume.from_name("paint3d-jobs", create_if_missing=True)

JOBS_DIR = "/jobs"


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
            err = f.read()
        raise HTTPException(500, err)
    return JSONResponse(status_code=202, content={"status": "pending", "job_id": job_id})


@app.cls(
    gpu="A10G",  # Paint3D fits comfortably in 24 GB. Bump to A100 for speed.
    volumes={"/hf_cache": hf_cache_vol, JOBS_DIR: jobs_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    timeout=3600,
    max_containers=2,
)
class Painter:
    @modal.enter()
    def load(self):
        """Pre-warm by touching SD 1.5 + ControlNet weights so they're cached."""
        import os
        from huggingface_hub import snapshot_download
        for repo in [
            "runwayml/stable-diffusion-v1-5",
            "lllyasviel/control_v11f1p_sd15_depth",
            "lllyasviel/control_v11p_sd15_inpaint",
        ]:
            try:
                snapshot_download(
                    repo,
                    cache_dir=os.environ["HF_HOME"],
                    allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"],
                )
            except Exception as e:
                print(f"[warn] prefetch {repo} failed: {e}")

    @modal.method()
    def _do_paint(
        self,
        mesh_bytes: bytes,
        mesh_suffix: str,
        prompt: str,
        ip_image_bytes: bytes | None,
        job_id: str,
        seed: int,
        sd_config: str,
        render_config: str,
    ):
        """Run the two-stage Paint3D pipeline.

        Stage 1: pipeline_paint3d_stage1.py  (coarse multi-view texture)
        Stage 2: pipeline_paint3d_stage2.py  (UV inpaint + illumination fix)
        Then: bundle the textured OBJ + albedo into a GLB via trimesh.
        """
        import os, glob, subprocess, tempfile, traceback
        import trimesh

        out_path, err_path = _job_paths(job_id)
        workdir = f"/work/{job_id}"
        os.makedirs(workdir, exist_ok=True)

        try:
            # 1. Persist the input mesh to disk (Paint3D reads paths only).
            mesh_path = f"{workdir}/input{mesh_suffix}"
            with open(mesh_path, "wb") as f:
                f.write(mesh_bytes)
            # Paint3D's pipeline expects OBJ. If the user uploaded GLB/PLY,
            # convert via trimesh.
            if mesh_suffix.lower() != ".obj":
                tm = trimesh.load(mesh_path, force="mesh")
                obj_path = f"{workdir}/input.obj"
                tm.export(obj_path)
                mesh_path = obj_path

            # Optional IP-adapter image (style transfer-ish conditioning).
            ip_image_path = None
            if ip_image_bytes:
                ip_image_path = f"{workdir}/ip.png"
                with open(ip_image_path, "wb") as f:
                    f.write(ip_image_bytes)

            stage1_out = f"{workdir}/stage1"
            stage2_out = f"{workdir}/stage2"

            # Stage 1
            cmd1 = [
                "python", "pipeline_paint3d_stage1.py",
                "--mesh_path", mesh_path,
                "--prompt", prompt,
                "--outdir", stage1_out,
                "--sd_config", sd_config,
                "--render_config", render_config,
                "--seed", str(seed),
            ]
            if ip_image_path:
                cmd1 += ["--ip_adapter_image_path", ip_image_path]
            print(f"[paint3d] stage1: {' '.join(cmd1)}")
            subprocess.run(cmd1, check=True, cwd="/app")

            # Stage 2 — refine. Find stage 1's albedo output to feed in.
            stage1_albedo = sorted(
                glob.glob(f"{stage1_out}/**/albedo.png", recursive=True))[-1]
            cmd2 = [
                "python", "pipeline_paint3d_stage2.py",
                "--mesh_path", mesh_path,
                "--texture_path", stage1_albedo,
                "--prompt", prompt,
                "--outdir", stage2_out,
                "--sd_config", sd_config,
                "--render_config", render_config,
                "--seed", str(seed),
            ]
            print(f"[paint3d] stage2: {' '.join(cmd2)}")
            subprocess.run(cmd2, check=True, cwd="/app")

            # 2. Bundle textured OBJ + final albedo into a single GLB.
            final_albedo = sorted(
                glob.glob(f"{stage2_out}/**/albedo.png", recursive=True))[-1]
            tm = trimesh.load(mesh_path, force="mesh")
            from PIL import Image
            tex = Image.open(final_albedo).convert("RGB")
            tm.visual = trimesh.visual.TextureVisuals(
                uv=tm.visual.uv if hasattr(tm.visual, "uv") else None,
                image=tex,
            )
            tm.export(out_path)  # .glb
            print(f"[paint3d] wrote {out_path}")
        except Exception:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
        finally:
            jobs_vol.commit()

    @modal.asgi_app()
    def web(self):
        import uuid, os
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="Paint3D")

        @api.get("/")
        def root():
            return {
                "service": "paint3d",
                "submit": "POST /paint (multipart: mesh, prompt, [ip_image])",
                "poll": "GET /jobs/{job_id}",
            }

        @api.post("/paint")
        async def paint(
            mesh: UploadFile = File(...),
            prompt: str = Form(...),
            ip_image: UploadFile | None = File(None),
            seed: int = Form(0),
            sd_config: str = Form("controlnet/config/depth_based_inpaint_template.yaml"),
            render_config: str = Form("paint3d/config/train_config_paint3d.py"),
        ):
            import gzip
            mesh_bytes = await mesh.read()
            if not mesh_bytes:
                raise HTTPException(400, "empty mesh upload")
            # Modal's ASGI proxy caps request bodies at ~20 MiB, so the client
            # may have gzipped the mesh. Detect the gzip magic and inflate.
            filename = mesh.filename or ""
            if mesh_bytes[:2] == b"\x1f\x8b":
                mesh_bytes = gzip.decompress(mesh_bytes)
                if filename.endswith(".gz"):
                    filename = filename[:-3]
            ip_bytes = await ip_image.read() if ip_image else None
            suffix = os.path.splitext(filename)[1].lower() or ".obj"

            job_id = uuid.uuid4().hex
            await self._do_paint.spawn.aio(
                mesh_bytes, suffix, prompt, ip_bytes, job_id,
                seed, sd_config, render_config,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return await _serve_job(job_id, download_name=f"{job_id}.glb")

        return api