"""ULIP-2 point-cloud↔text scorer on Modal GPU.

Why Modal: ULIP-2's colored PointBERT encoder needs CUDA ops (pointnet2_ops
furthest-point-sampling) that don't build on macOS. On a real GPU it runs
exactly as the authors intended — no CPU hacks — giving a number directly
comparable to Twist & Compute (which reports ULIP).

Split of work:
  * CLIENT (run_pilot / backfill_ulip) samples a 10k xyz+rgb point cloud from the
    .glb with trimesh and POSTs the array here.
  * THIS APP encodes the point cloud + caption and returns cosine similarity.

Model: ULIP-2 PointBERT (10k, xyz+rgb), open_clip ViT-bigG-14 text encoder.
Checkpoint: HF dataset SFXX/ulip ->
  ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt

Deploy (from a workspace you're authed to):
    modal deploy eval/ulip_modal.py
Endpoint:
    https://<workspace>--ulip2-scorer-web.modal.run/score   (POST)
"""
import modal

APP_NAME = "ulip2-scorer"
CKPT_REPO = "SFXX/ulip"
CKPT_FILE = ("ULIP-2/pretrained_models/"
             "ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-objaverse_shapenet-pretrained.pt")
CUDA_ARCHS = "8.0;8.6;8.9;9.0"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10",
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": CUDA_ARCHS,
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "OPEN_CLIP_CACHE_DIR": "/cache/open_clip",
    })
    .apt_install("git", "build-essential", "wget")
    .pip_install(
        "torch==2.4.0", "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "open_clip_torch==2.24.0", "timm", "easydict", "pyyaml",
        "numpy<2", "ninja", "huggingface_hub[hf_transfer]",
        "fastapi", "python-multipart",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/salesforce/ULIP.git /ULIP",
        # Colored PointBERT only uses a pure-torch KNN at inference; the CUDA
        # knn_cuda import (and its module-level instance) are unused but would
        # crash the import. Neutralise both.
        "python -c \""
        "import pathlib;"
        "p=pathlib.Path('/ULIP/models/pointbert/dvae.py');"
        "s=p.read_text();"
        "s=s.replace('from knn_cuda import KNN','try:\\n    from knn_cuda import KNN\\nexcept Exception:\\n    KNN=None');"
        "s=s.replace('knn = KNN(k=4, transpose_mode=False)','knn = None');"
        "p.write_text(s)\"",
        # CUDA furthest-point-sampling used inside PointBERT's grouping.
        "pip install 'git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#subdirectory=pointnet2_ops_lib'",
    )
    .workdir("/ULIP")  # so the relative colored-pointbert config yaml resolves
)

app = modal.App(APP_NAME, image=image)
cache_vol = modal.Volume.from_name("ulip2-cache", create_if_missing=True)


@app.cls(gpu="L4", volumes={"/cache": cache_vol},
         scaledown_window=300, timeout=600)
class Scorer:
    @modal.enter()
    def load(self):
        import sys, torch
        from easydict import EasyDict
        from huggingface_hub import hf_hub_download
        sys.path.insert(0, "/ULIP")
        from models.ULIP_models import ULIP2_PointBERT_Colored

        self.torch = torch
        self.device = "cuda"
        # evaluate_3d=True skips the train-only head inside the encoder.
        args = EasyDict(evaluate_3d=True, npoints=10000)
        self.model = ULIP2_PointBERT_Colored(args).to(self.device).eval()

        ckpt_path = hf_hub_download(
            repo_id=CKPT_REPO, repo_type="dataset", filename=CKPT_FILE,
            cache_dir="/cache/hf_ckpt",
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        report = self.model.load_state_dict(sd, strict=False)
        print(f"[ulip] ckpt loaded; missing={len(report.missing_keys)} "
              f"unexpected={len(report.unexpected_keys)}")
        cache_vol.commit()

    def _score(self, pc, caption: str) -> float:
        torch = self.torch
        pc_t = torch.as_tensor(pc, dtype=torch.float32,
                               device=self.device).unsqueeze(0)  # (1,10000,6)
        with torch.no_grad():
            pc_embed = self.model.encode_pc(pc_t)
            pc_embed = pc_embed / pc_embed.norm(dim=-1, keepdim=True)
            tokens = self.model.tokenizer([caption]).to(self.device)
            txt_embed = self.model.encode_text(tokens)
            txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)
            return float((pc_embed @ txt_embed.T).item())

    @modal.asgi_app()
    def web(self):
        import io
        import numpy as np
        from fastapi import FastAPI, UploadFile, File, Form
        from fastapi.responses import JSONResponse

        api = FastAPI()

        @api.get("/")
        def health():
            return {"service": APP_NAME, "ok": True}

        @api.post("/score")
        async def score(caption: str = Form(...), pc: UploadFile = File(...)):
            arr = np.load(io.BytesIO(await pc.read()))
            if arr.ndim != 2 or arr.shape[1] != 6:
                return JSONResponse({"error": f"bad pc shape {arr.shape}"},
                                    status_code=400)
            return {"ulip": self._score(arr.tolist(), caption)}

        return api
