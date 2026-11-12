#!/usr/bin/env python
"""Local TRELLIS.2 HTTP server — drop-in replacement for the Modal endpoint.

Runs in the `trellis` conda env on the GPU box, loads the pipeline ONCE, and
serves image->GLB over HTTP. The agent (running in its own Django/langchain env)
posts to it instead of Modal:

    # terminal 1 (trellis env):
    conda activate trellis
    export OPENCV_IO_ENABLE_OPENEXR=1
    python eval/local_trellis_server.py --repo ~/TRELLIS.2 --port 8200

    # terminal 2 (agent/eval env):
    export LOCAL_TRELLIS_URL=http://127.0.0.1:8200
    python eval/run_pilot.py --input retrieved --limit 10 --conditions loo_view ...

Needs:  pip install fastapi uvicorn python-multipart  (in the trellis env)
"""
import argparse
import io
import os
import sys
import tempfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/TRELLIS.2"))
    ap.add_argument("--port", type=int, default=8200)
    args = ap.parse_args()

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        sys.exit(f"TRELLIS.2 repo not found at {repo}")
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from PIL import Image
    import uvicorn
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import FileResponse
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    import o_voxel

    print("loading TRELLIS.2-4B (downloads weights on first run)...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()
    print(f"ready — serving on http://0.0.0.0:{args.port}")

    app = FastAPI()

    import subprocess
    import torch
    _commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                             capture_output=True, text=True).stdout.strip()
    _info = {"status": "ok", "model": "microsoft/TRELLIS.2-4B",
             "repo_commit": _commit, "torch": torch.__version__,
             "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}

    @app.get("/")
    def health():
        return _info

    import gc
    import threading
    from fastapi.responses import Response
    gpu_lock = threading.Lock()   # one generation at a time on this GPU

    # Plain `def` (not async): FastAPI runs it in a worker thread, so the
    # health check GET / still answers while a generation is running.
    @app.post("/generate")
    def generate(
        image: UploadFile = File(...),
        seed: int = Form(42),
        decimation_target: int = Form(1_000_000),
        texture_size: int = Form(4096),
        remesh: str = Form("true"),
    ):
        data = image.file.read()
        with gpu_lock:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            mesh = pipeline.run(img, seed=seed)[0]  # honor the client's seed
            mesh.simplify(16777216)  # nvdiffrast limit
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target, texture_size=texture_size,
                remesh=(remesh.lower() == "true"), remesh_band=1, remesh_project=0,
                verbose=False)
            # Export, read back, and DELETE the temp file — leaving it behind
            # leaked ~40 MB per request into /tmp. Then free memory: a server
            # that runs for days must not grow request by request.
            with tempfile.NamedTemporaryFile(suffix=".glb") as out:
                glb.export(out.name, extension_webp=True)
                payload = Path(out.name).read_bytes()
            del mesh, glb, img
            gc.collect()
            torch.cuda.empty_cache()
        return Response(content=payload, media_type="model/gltf-binary",
                        headers={"Content-Disposition": 'attachment; filename="model.glb"'})

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
