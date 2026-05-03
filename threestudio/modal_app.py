"""
Modal deployment for threestudio (Fantasia3D texture stage).

Takes (mesh, prompt) and runs optimization-based SDS refinement of a neural
PBR texture field on the input geometry. Exports OBJ+MTL, converts to GLB.

Async job pattern matches trellis2/modal_app.py — submit returns a job_id,
you poll /jobs/{id} for the GLB.

Routes:
    POST /refine     image-like multipart: mesh, prompt  -> {job_id}
    GET  /jobs/{id}                                      -> 202 / 200 GLB / 5xx

Deploy:
    modal deploy modal_app.py
"""

import modal

APP_NAME = "threestudio-refiner"

# A100/A10/L4/H100
CUDA_ARCHS = "8.0;8.6;8.9;9.0"

# Mirrors threestudio/docker/Dockerfile — their canonical install recipe.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python=None,  # use system python3.10 from Ubuntu 22.04
    )
    .env({
        # CUDA 11.8's nvcc segfaults compiling nvdiffrast for sm_90. Build for
        # sm_89 with PTX; H100 (sm_90) will JIT-compile from PTX at runtime.
        "TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.9+PTX",
        "TCNN_CUDA_ARCHITECTURES": "89;86;80",
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "LIBRARY_PATH": "/usr/local/cuda/lib64/stubs",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "MAX_JOBS": "4",
    })
    .apt_install(
        "build-essential", "curl", "git", "wget",
        "libegl1-mesa-dev", "libgl1-mesa-dev", "libgles2-mesa-dev",
        "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "python-is-python3", "python3.10-dev", "python3-pip",
        "ffmpeg",
    )
    .run_commands(
        # Exactly what their Dockerfile does.
        "pip install --upgrade pip setuptools==69.5.1 ninja",
        "pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 "
        "--index-url https://download.pytorch.org/whl/cu118",
        # Pre-install common build-time deps — required because we use
        # --no-build-isolation for the CUDA/torch extensions below.
        "pip install pybind11 Cython wheel packaging",
        "pip install git+https://github.com/KAIR-BAIR/nerfacc.git@v0.5.2 --no-build-isolation",
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch --no-build-isolation",
    )
    .add_local_file(
        "requirements.txt", "/tmp/requirements.txt", copy=True,
    )
    .run_commands(
        # Strip xformers + bitsandbytes from requirements.txt: both unpinned,
        # both pull in a modern torch that breaks the ABI for the already-compiled
        # nerfacc/tiny-cuda-nn. Neither is imported at threestudio module-load.
        # Also strip libigl: unpinned resolves to >=2.5, which dropped top-level
        # fast_winding_number_for_meshes / point_mesh_squared_distance / read_obj
        # that threestudio/utils/ops.py imports. Reinstall 2.4.1 below.
        "sed -i -E '/^(xformers|bitsandbytes|libigl)(\\b|==|>=|<=)/d' /tmp/requirements.txt",
        # requirements.txt includes git+ deps (nvdiffrast, envlight) that import
        # torch in setup.py, so build-isolation would hide it. --no-build-isolation
        # lets them see the installed torch.
        "pip install -r /tmp/requirements.txt --no-build-isolation",
        # Safety net: force torch back to the pinned version in case any
        # requirements.txt transitive dep upgraded it.
        "pip install --force-reinstall --no-deps "
        "torch==2.0.1+cu118 torchvision==0.15.2+cu118 "
        "--index-url https://download.pytorch.org/whl/cu118",
        # pytorch-lightning resolves too new otherwise — newer versions import
        # torch.utils.flop_counter (a torch 2.1+ symbol we don't have).
        "pip install --force-reinstall --no-deps "
        "'pytorch-lightning==2.0.0' 'lightning==2.0.0'",
        # diffusers<0.20 in requirements.txt still imports `cached_download`,
        # which was removed from huggingface_hub>=0.26. Downgrade.
        "pip install 'huggingface_hub<0.26' 'hf_transfer'",
        # libigl 2.4.1 is the last release that exposes fast_winding_number_for_meshes,
        # point_mesh_squared_distance, and read_obj at the top level.
        "pip install 'libigl==2.4.1'",
        # Modal-specific extras for the FastAPI endpoint. Force-upgrade because
        # something in requirements.txt resolves an ancient fastapi that imports
        # pydantic v1's `Schema` (removed in pydantic 2).
        "pip install --upgrade 'fastapi>=0.103.0,<0.110' 'pydantic>=2.0,<3' "
        "python-multipart",
    )
    .add_local_dir(
        ".",
        "/app",
        copy=True,
        ignore=[
            ".git/**", "outputs/**",
            "__pycache__", "**/__pycache__", "*.pyc",
            "*.mp4", "*.glb", "modal_app.py",
        ],
    )
    .workdir("/app")
)

app = modal.App(APP_NAME, image=image)

hf_cache_vol = modal.Volume.from_name("threestudio-hf-cache", create_if_missing=True)
jobs_vol = modal.Volume.from_name("threestudio-jobs", create_if_missing=True)

JOBS_DIR = "/jobs"
WORKROOT = "/work"  # per-run scratch


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
            out_path, media_type="model/gltf-binary", filename=download_name,
        )
    if os.path.exists(err_path):
        with open(err_path) as f:
            err = f.read()
        raise HTTPException(500, err)
    return JSONResponse(status_code=202, content={"status": "pending", "job_id": job_id})


@app.cls(
    gpu="L40S",
    volumes={"/cache": hf_cache_vol, JOBS_DIR: jobs_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=300,
    timeout=7200,
    max_containers=2,
)
class Refiner:
    @modal.method()
    def _do_refine(
        self,
        mesh_bytes: bytes,
        mesh_suffix: str,
        prompt: str,
        negative_prompt: str,
        job_id: str,
        seed: int,
        max_steps: int,
        guidance_scale: float,
        mesh_up: str,
        mesh_front: str,
    ):
        import os, subprocess, shutil, tempfile, traceback, glob
        import trimesh
        from omegaconf import OmegaConf

        out_path, err_path = _job_paths(job_id)
        workdir = f"{WORKROOT}/{job_id}"
        os.makedirs(workdir, exist_ok=True)

        try:
            # 1. Stash the input mesh
            in_mesh = f"{workdir}/input{mesh_suffix}"
            with open(in_mesh, "wb") as f:
                f.write(mesh_bytes)

            # threestudio's custom-mesh loader expects a file; GLBs load fine via trimesh.
            outputs_root = f"{workdir}/outputs"
            os.makedirs(outputs_root, exist_ok=True)

            # 2. Build a patched config. The stock fantasia3d-texture.yaml's
            # `system.geometry` block is tetrahedra-sdf-grid-specific (requires
            # a prev-stage ckpt via geometry_convert_from=??? and carries keys
            # CustomMesh.Config rejects: isosurface_resolution,
            # isosurface_deformable_grid, fix_geometry). threestudio's CLI
            # override parser is dotlist-only — no `~key` delete support — so
            # instead of overriding, we load the yaml, rewrite the geometry
            # block for custom-mesh, and pass the patched file.
            cfg = OmegaConf.load("/app/configs/fantasia3d-texture.yaml")
            cfg.system.geometry_convert_from = None
            cfg.system.geometry_type = "custom-mesh"
            cfg.system.geometry = {
                "radius": 1.0,
                "shape_init": f"mesh:{in_mesh}",
                "shape_init_params": 1.0,
                "shape_init_mesh_up": mesh_up,
                "shape_init_mesh_front": mesh_front,
                "pos_encoding_config": {
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": 1.4472692374403782,
                },
                "n_feature_dims": 8,  # albedo3 + roughness1 + metallic1 + bump3
            }
            patched_cfg = f"{workdir}/config.yaml"
            OmegaConf.save(cfg, patched_cfg)

            # 3. Train: SDS on neural PBR texture field, geometry frozen
            train_cmd = [
                "python", "launch.py",
                "--config", patched_cfg,
                "--train", "--gpu", "0",
                f"exp_root_dir={outputs_root}",
                f"seed={seed}",
                f'system.prompt_processor.prompt={prompt!r}',
                f'system.prompt_processor.negative_prompt={negative_prompt!r}',
                f"system.guidance.guidance_scale={guidance_scale}",
                f"trainer.max_steps={max_steps}",
                "trainer.val_check_interval=99999",  # skip eval renders
            ]
            print("[refiner] training:", " ".join(train_cmd), flush=True)
            subprocess.run(train_cmd, check=True, cwd="/app")

            # Find the trial dir (outputs/fantasia3d-texture/<rmspace-prompt>@<timestamp>)
            trial_dirs = sorted(
                glob.glob(f"{outputs_root}/fantasia3d-texture/*"),
                key=os.path.getmtime,
            )
            if not trial_dirs:
                raise RuntimeError("training produced no trial dir")
            trial_dir = trial_dirs[-1]
            ckpt = f"{trial_dir}/ckpts/last.ckpt"
            parsed_cfg = f"{trial_dir}/configs/parsed.yaml"

            # 3. Export to OBJ+MTL
            export_cmd = [
                "python", "launch.py",
                "--config", parsed_cfg,
                "--export", "--gpu", "0",
                f"resume={ckpt}",
                "system.exporter_type=mesh-exporter",
            ]
            print("[refiner] exporting:", " ".join(export_cmd), flush=True)
            subprocess.run(export_cmd, check=True, cwd="/app")

            # 4. Find the exported OBJ and convert to GLB
            save_dir = f"{trial_dir}/save"
            obj_candidates = glob.glob(f"{save_dir}/**/*.obj", recursive=True)
            if not obj_candidates:
                raise RuntimeError(f"no .obj produced under {save_dir}")
            obj_path = obj_candidates[0]

            scene = trimesh.load(obj_path, force="scene")
            scene.export(out_path)
            print(f"[refiner] wrote {out_path}", flush=True)

        except Exception:
            with open(err_path, "w") as f:
                f.write(traceback.format_exc())
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            jobs_vol.commit()

    @modal.asgi_app()
    def web(self):
        import os, uuid
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException

        api = FastAPI(title="threestudio SDS refiner")

        @api.get("/")
        def root():
            return {
                "service": "threestudio-refiner",
                "submit": "POST /refine (multipart: mesh, prompt)",
                "poll": "GET /jobs/{job_id}",
                "notes": "Re-textures input geometry with SDS. Input texture is discarded.",
            }

        @api.post("/refine")
        async def refine(
            mesh: UploadFile = File(...),
            prompt: str = Form(...),
            negative_prompt: str = Form(
                "blurry, low quality, distorted, oversaturated, unrealistic"
            ),
            seed: int = Form(0),
            max_steps: int = Form(1500),
            guidance_scale: float = Form(100.0),
            mesh_up: str = Form("+y"),     # GLB convention
            mesh_front: str = Form("+z"),
        ):
            import gzip
            mesh_bytes = await mesh.read()
            # Modal's ASGI proxy caps request bodies at ~20 MiB, so the client
            # may have gzipped the mesh. Detect the gzip magic and inflate.
            filename = mesh.filename or ""
            if mesh_bytes[:2] == b"\x1f\x8b":
                mesh_bytes = gzip.decompress(mesh_bytes)
                if filename.endswith(".gz"):
                    filename = filename[:-3]
            suffix = os.path.splitext(filename)[1].lower() or ".glb"
            if suffix not in (".glb", ".obj", ".ply", ".stl"):
                raise HTTPException(400, f"unsupported mesh format: {suffix}")

            job_id = uuid.uuid4().hex
            await self._do_refine.spawn.aio(
                mesh_bytes, suffix, prompt, negative_prompt, job_id,
                seed, max_steps, guidance_scale, mesh_up, mesh_front,
            )
            return {"job_id": job_id, "poll_url": f"/jobs/{job_id}"}

        @api.get("/jobs/{job_id}")
        async def get_job(job_id: str):
            return _serve_job(job_id, download_name=f"{job_id}.glb")

        return api
