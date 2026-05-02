import argparse
import gzip
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

WORKSPACE = os.environ.get("TRELLIS_WORKSPACE", "yousefarafa40612")
TRELLIS_GEN_URL = f"https://{WORKSPACE}--trellis2-generator-web.modal.run"
TRELLIS_TEX_URL = f"https://{WORKSPACE}--trellis2-texturer-web.modal.run"
PARTCRAFTER_OBJ_URL = f"https://{WORKSPACE}--partcrafter-objectgenerator-web.modal.run"
PARTCRAFTER_SCENE_URL = f"https://{WORKSPACE}--partcrafter-scenegenerator-web.modal.run"
THREESTUDIO_URL = f"https://{WORKSPACE}--threestudio-refiner-refiner-web.modal.run"
DREAMEDITOR_URL = f"https://{WORKSPACE}--dreameditor-editor-web.modal.run"
PAINT3D_URL = f"https://{WORKSPACE}--paint3d-painter-web.modal.run"


OBJAVERSE_STYLE_PROMPT = (
    "Restyle this image as a single 3D asset rendered in the style of the "
    "Objaverse dataset: one centered object on a plain neutral background, "
    "even studio lighting, no shadows on the ground, no scene context, no "
    "text, clean matte materials, orthographic-feeling three-quarter view, "
    "the object fully visible and uncropped."
)

GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"


_gemini_client: Optional[genai.Client] = None


def _gemini() -> genai.Client:
    """Lazy singleton — only constructed when a Gemini stage actually runs."""
    global _gemini_client
    if _gemini_client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set (check your .env)")
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


def _extract_image_bytes(response) -> Optional[bytes]:
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    return None


def _save_png(data: bytes, out_path: str | Path) -> str:
    out_path = str(out_path)
    Image.open(io.BytesIO(data)).save(out_path)
    return out_path


def generate_concept_image(prompt: str,
                           out_path: str | Path | None = None) -> str:
    """Generate a concept image from a text prompt via Gemini.

    Returns the path to the saved PNG.
    """
    out_path = out_path or f"concept_{int(time.time())}.png"
    print(f"[gemini] concept: {prompt!r}")
    resp = _gemini().models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    data = _extract_image_bytes(resp)
    if not data:
        raise RuntimeError("Gemini returned no image for concept prompt")
    path = _save_png(data, out_path)
    print(f"[gemini] saved {path}")
    return path


def restyle_to_objaverse(image_path: str | Path,
                         out_path: str | Path | None = None,
                         style_prompt: str = OBJAVERSE_STYLE_PROMPT) -> str:
    """Restyle an image to look like an Objaverse-dataset asset.

    Returns the path to the saved PNG.
    """
    image_path = str(image_path)
    out_path = out_path or f"objaverse_{int(time.time())}.png"
    print(f"[gemini] restyle: {image_path}")
    src = _gemini().files.upload(file=image_path)
    resp = _gemini().models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=[src, style_prompt],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    data = _extract_image_bytes(resp)
    if not data:
        raise RuntimeError("Gemini returned no image for restyle")
    path = _save_png(data, out_path)
    print(f"[gemini] saved {path}")
    return path


# Modal's ASGI proxy caps request bodies at ~20 MiB. Anything larger and the
# proxy aborts the connection mid-upload (looks like a 400 to the client). For
# texture-overwriting pipelines (Paint3D, threestudio) the input texture is
# discarded anyway, so stripping it is free; gzip on the resulting geometry
# typically halves it again. The server detects the gzip magic and inflates.
_BODY_BUDGET = 18 * 1024 * 1024


def _compact_mesh_for_upload(mesh_path: str | Path,
                             strip_textures: bool = True) -> str:
    """Return a path whose bytes will fit through the Modal proxy.

    Strips embedded textures (lossless for pipelines that re-texture from
    scratch), gzips, and returns a temp file path. Falls back to the original
    file if it already fits.
    """
    mesh_path = str(mesh_path)
    if os.path.getsize(mesh_path) <= _BODY_BUDGET and not strip_textures:
        return mesh_path

    import trimesh
    m = trimesh.load(mesh_path, force="mesh")
    if strip_textures:
        m.visual = trimesh.visual.ColorVisuals(mesh=m)
    buf = io.BytesIO()
    m.export(buf, file_type="glb")
    raw = buf.getvalue()

    if len(raw) <= _BODY_BUDGET:
        payload, suffix = raw, ".glb"
    else:
        payload, suffix = gzip.compress(raw, 6), ".glb.gz"
        if len(payload) > _BODY_BUDGET:
            raise RuntimeError(
                f"mesh still {len(payload)/1e6:.1f} MB after strip+gzip "
                f"(budget {_BODY_BUDGET/1e6:.0f} MB). Decimate the mesh first.")

    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="upload_")
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    print(f"[upload] {mesh_path} -> {tmp} "
          f"({os.path.getsize(mesh_path)/1e6:.1f} MB -> {len(payload)/1e6:.1f} MB)")
    return tmp


def _download_via_volume(volume_name: str, remote_name: str,
                         out_path: str) -> int:
    """Download a file from a Modal volume by shelling out to the CLI.

    The Python `Volume.read_file` API only works from inside a running
    Modal function; from a local script you have to use the CLI, which
    is also what `modal volume get` does. Returns bytes written.
    """
    import shutil
    import subprocess
    if shutil.which("modal") is None:
        raise RuntimeError("modal CLI not found on PATH")
    if os.path.exists(out_path):
        os.remove(out_path)
    subprocess.run(
        ["modal", "volume", "get", volume_name, remote_name, out_path],
        check=True,
    )
    return os.path.getsize(out_path)


def _run_modal_job(tag: str,
                   submit_url: str,
                   base_url: str,
                   files: dict,
                   form: dict,
                   out_path: str,
                   volume_name: str | None = None,
                   remote_suffix: str = ".glb",
                   poll_every: int = 10,
                   submit_retries: int = 8,
                   submit_read_timeout: int = 600,
                   download_read_timeout: int = 600,
                   min_bytes: int = 1024,
                   cold_start_budget_s: int = 900) -> str:
    """Generic submit -> poll -> download for any TRELLIS-style Modal app.

    `files` is a dict of {field: open_path_str}. We open and stream each.
    Returns the path to the downloaded artifact.
    """
    print(f"[{tag}] submit -> {submit_url}")
    job_id: Optional[str] = None
    last_err: Optional[Exception] = None
    for attempt in range(1, submit_retries + 1):
        opened = {k: open(v, "rb") for k, v in files.items()}
        try:
            r = requests.post(
                submit_url, files=opened, data=form,
                timeout=(10, submit_read_timeout),
            )
            if r.status_code >= 500:
                raise requests.exceptions.HTTPError(
                    f"{r.status_code}: {r.text[:200]}", response=r)
            r.raise_for_status()
            job_id = r.json()["job_id"]
            break
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            last_err = e
            wait = min(60, 10 * (2 ** (attempt - 1)))
            print(f"[{tag}] submit {attempt}/{submit_retries} failed "
                  f"({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
        finally:
            for fh in opened.values():
                fh.close()
    if job_id is None:
        raise RuntimeError(f"{tag} submit failed after {submit_retries}: {last_err}")
    print(f"[{tag}] job={job_id}")

    job_url = f"{base_url}/jobs/{job_id}"
    tiny_count = 0
    transient_since: Optional[float] = None 
    while True:
        try:
            resp = requests.get(job_url, timeout=(10, download_read_timeout),
                                stream=True)
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            if transient_since is None:
                transient_since = time.time()
            elapsed = time.time() - transient_since
            if elapsed > cold_start_budget_s:
                raise RuntimeError(
                    f"{tag} poll kept failing for {elapsed:.0f}s "
                    f"({type(e).__name__}); giving up. Last err: {e}")
            print(f"[{tag}] poll error ({type(e).__name__}); "
                  f"retrying ({elapsed:.0f}s/{cold_start_budget_s}s)")
            time.sleep(poll_every)
            continue

        if resp.status_code == 200:
            content_length = resp.headers.get("content-length")
            # Drain/close HTTP body — we only used it as a "ready" signal.
            # The HTTP FileResponse path is much slower than reading the
            # volume directly, so prefer the SDK if a volume name is given.
            if volume_name is not None:
                resp.close()
                print(f"[{tag}] downloading from volume {volume_name}")
                try:
                    written = _download_via_volume(
                        volume_name, f"{job_id}{remote_suffix}", out_path)
                except Exception as e:
                    print(f"[{tag}] volume download failed ({e}); "
                          f"falling back to HTTP")
                    written = 0
                if written < min_bytes:
                    with open(out_path, "wb") as out:
                        with requests.get(job_url, stream=True,
                                          timeout=(10, download_read_timeout)
                                          ) as r2:
                            for chunk in r2.iter_content(chunk_size=1 << 20):
                                if chunk:
                                    out.write(chunk)
                                    written += len(chunk)
            else:
                written = 0
                with open(out_path, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            out.write(chunk)
                            written += len(chunk)
                resp.close()
            if written < min_bytes:
                # Server says done but body is empty/tiny. Common causes:
                #   - FileResponse opened the path before the worker
                #     finished flushing/committing (race condition).
                #   - Worker crashed after creating an empty file.
                # Treat as transient and re-poll a few times before bailing.
                tiny_count += 1
                print(f"[{tag}] WARN: 200 with only {written} bytes "
                      f"(content-length={content_length}); retry {tiny_count}/5")
                if tiny_count >= 5:
                    raise RuntimeError(
                        f"{tag} kept returning empty artifact "
                        f"({written} bytes); job may have crashed silently. "
                        f"Inspect with `modal volume ls trellis2-jobs` and "
                        f"check worker logs in the Modal dashboard.")
                time.sleep(poll_every)
                continue
            print(f"[{tag}] done -> {out_path} ({written/1e6:.2f} MB)")
            return out_path
        if resp.status_code == 202:
            resp.close()
            transient_since = None  # got a real response, reset budget
            print(f"[{tag}] pending...")
            time.sleep(poll_every)
            continue
        # 5xx during cold start = Modal edge / container not ready.
        # Treat as transient until cold_start_budget_s elapses since first
        # bad response. 4xx is a real client/server error — fail fast.
        if resp.status_code >= 500:
            body = resp.text[:500]
            resp.close()
            if transient_since is None:
                transient_since = time.time()
            elapsed = time.time() - transient_since
            if elapsed > cold_start_budget_s:
                raise RuntimeError(
                    f"{tag} {resp.status_code} for {elapsed:.0f}s "
                    f"(>{cold_start_budget_s}s budget): {body}")
            print(f"[{tag}] {resp.status_code} (cold start?); "
                  f"retrying ({elapsed:.0f}s/{cold_start_budget_s}s): {body[:120]}")
            time.sleep(poll_every)
            continue
        body = resp.text[:500]
        resp.close()
        raise RuntimeError(f"{tag} error {resp.status_code}: {body}")


def image_to_3d(image_path: str | Path,
                out_path: str | Path | None = None,
                pipeline_type: str = "1024_cascade",
                remesh: bool = True,
                seed: int = 42,
                decimation_target: int = 1_000_000,
                texture_size: int = 4096,
                **job_kwargs) -> str:
    """TRELLIS.2: image -> GLB."""
    out_path = str(out_path or f"model_{int(time.time())}.glb")
    return _run_modal_job(
        tag="trellis2",
        submit_url=f"{TRELLIS_GEN_URL}/generate",
        base_url=TRELLIS_GEN_URL,
        files={"image": str(image_path)},
        form={
            "pipeline_type": pipeline_type,
            "remesh": str(remesh).lower(),
            "seed": str(seed),
            "decimation_target": str(decimation_target),
            "texture_size": str(texture_size),
        },
        out_path=out_path,
        volume_name="trellis2-jobs",
        **job_kwargs,
    )


def texture_mesh(image_path: str | Path,
                 mesh_path: str | Path,
                 out_path: str | Path | None = None,
                 
                 seed: int = 42,
                 resolution: int = 1024,
                 texture_size: int = 2048,
                 **job_kwargs) -> str:
    """TRELLIS.2 texturer: image + mesh -> textured GLB."""
    out_path = str(out_path or f"textured_{int(time.time())}.glb")
    return _run_modal_job(
        tag="trellis2-tex",
        submit_url=f"{TRELLIS_TEX_URL}/texture",
        base_url=TRELLIS_TEX_URL,
        files={"image": str(image_path), "mesh": str(mesh_path)},
        form={
            "seed": str(seed),
            "resolution": str(resolution),
            "texture_size": str(texture_size),
        },
        out_path=out_path,
        volume_name="trellis2-jobs",
        **job_kwargs,
    )


def partcrafter(image_path: str | Path,
                out_path: str | Path | None = None,
                num_parts: int = 3,
                seed: int = 0,
                num_tokens: int = 1024,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.0,
                rmbg: bool = True,
                scene: bool = False,
                **job_kwargs) -> str:
    """PartCrafter: image -> N-part GLB.

    Set `scene=True` to hit the scene-generator endpoint instead of the
    object generator.
    """
    base = PARTCRAFTER_SCENE_URL if scene else PARTCRAFTER_OBJ_URL
    route = "/generate-scene" if scene else "/generate"
    out_path = str(out_path or f"partcrafter_{int(time.time())}.glb")
    return _run_modal_job(
        tag="partcrafter-scene" if scene else "partcrafter",
        submit_url=f"{base}{route}",
        base_url=base,
        files={"image": str(image_path)},
        form={
            "num_parts": str(num_parts),
            "seed": str(seed),
            "num_tokens": str(num_tokens),
            "num_inference_steps": str(num_inference_steps),
            "guidance_scale": str(guidance_scale),
            "rmbg": str(rmbg).lower(),
        },
        out_path=out_path,
        volume_name="partcrafter-jobs",
        **job_kwargs,
    )


def threestudio_refine(mesh_path: str | Path,
                       prompt: str,
                       out_path: str | Path | None = None,
                       negative_prompt: str = (
                           "ugly, blurry, low quality, distorted"),
                       seed: int = 0,
                       max_steps: int = 1500,
                       guidance_scale: float = 100.0,
                       mesh_up: str = "+y",
                       mesh_front: str = "+z",
                       **job_kwargs) -> str:
    """threestudio SDS refiner: mesh + prompt -> refined GLB.

    Long-running (often 30+ min for max_steps=1500). Default timeouts in
    `_run_modal_job` are tuned for this; override via job_kwargs if needed.
    Input texture is discarded (SDS re-textures from scratch), so we strip
    it client-side to fit the Modal proxy's request body cap.
    """
    out_path = str(out_path or f"refined_{int(time.time())}.glb")
    return _run_modal_job(
        tag="threestudio",
        submit_url=f"{THREESTUDIO_URL}/refine",
        base_url=THREESTUDIO_URL,
        files={"mesh": _compact_mesh_for_upload(mesh_path)},
        form={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": str(seed),
            "max_steps": str(max_steps),
            "guidance_scale": str(guidance_scale),
            "mesh_up": mesh_up,
            "mesh_front": mesh_front,
        },
        out_path=out_path,
        volume_name="threestudio-jobs",
        **job_kwargs,
    )


def paint3d_texture(mesh_path: str | Path,
                    prompt: str,
                    out_path: str | Path | None = None,
                    ip_image_path: str | Path | None = None,
                    seed: int = 0,
                    sd_config: str = (
                        "controlnet/config/depth_based_inpaint_template.yaml"),
                    render_config: str = (
                        "paint3d/config/train_config_paint3d.py"),
                    **job_kwargs) -> str:
    """Paint3D: paint a high-res lighting-less texture onto an untextured mesh.

    Accepts .obj/.glb/.ply (non-OBJ auto-converted server-side via trimesh).
    Input texture is discarded (Paint3D paints from scratch), so we strip
    it client-side to fit the Modal proxy's request body cap. Returns a
    path to the textured GLB.
    """
    out_path = str(out_path or f"painted_{int(time.time())}.glb")
    files = {"mesh": _compact_mesh_for_upload(mesh_path)}
    if ip_image_path is not None:
        files["ip_image"] = str(ip_image_path)
    return _run_modal_job(
        tag="paint3d",
        submit_url=f"{PAINT3D_URL}/paint",
        base_url=PAINT3D_URL,
        files=files,
        form={
            "prompt": prompt,
            "seed": str(seed),
            "sd_config": sd_config,
            "render_config": render_config,
        },
        out_path=out_path,
        volume_name="paint3d-jobs",
        **job_kwargs,
    )

paint3d_texture("test.glb", prompt="a cat with an octopus head")