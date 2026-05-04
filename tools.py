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
HUNYUAN3D2_GEN_URL = f"https://{WORKSPACE}--hunyuan3d-2-generator-web.modal.run"


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


TAVILY_URL = "https://api.tavily.com/search"


def _tavily(payload: dict) -> dict:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set (check your .env)")
    r = requests.post(TAVILY_URL, json={"api_key": key, **payload}, timeout=30)
    r.raise_for_status()
    return r.json()


def web_search(query: str, max_results: int = 5) -> str:
    """Web search via Tavily. Returns markdown-formatted results."""
    print(f"[tavily] search: {query!r}")
    data = _tavily({
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    })
    out = [f"# Results for: {query}"]
    if answer := data.get("answer"):
        out.append(f"\n**Quick answer:** {answer}\n")
    for hit in data.get("results", []):
        snippet = (hit.get("content") or "")[:300].strip()
        out.append(f"- **{hit.get('title','')}** — {hit.get('url','')}\n  {snippet}")
    return "\n".join(out) if len(out) > 1 else "No results."


def image_search(query: str, max_results: int = 5) -> list[str]:
    """Image search via Tavily. Returns list of image URLs."""
    print(f"[tavily] image search: {query!r}")
    data = _tavily({
        "query": query,
        "max_results": max_results,
        "include_images": True,
    })
    return data.get("images", []) or []


def download_image(url: str, out_path: str | Path | None = None) -> str:
    """Download an image URL to a local file. Returns the saved path."""
    out_path = str(out_path or f"downloaded_{int(time.time())}.jpg")
    print(f"[download] {url}")
    r = requests.get(
        url, stream=True, timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    Image.open(out_path).verify()
    print(f"[download] saved {out_path}")
    return out_path


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


def _stream_to_file(resp, out_path: str) -> int:
    """Stream an HTTP response body to a file. Returns bytes written."""
    written = 0
    with open(out_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
                written += len(chunk)
    return written


def _download_artifact(tag: str, job_url: str, out_path: str,
                       volume_name: Optional[str], job_id: str,
                       remote_suffix: str, read_timeout: int) -> int:
    """Pull the finished artifact. Volume-first (faster), HTTP fallback."""
    if volume_name:
        try:
            print(f"[{tag}] downloading from volume {volume_name}")
            return _download_via_volume(
                volume_name, f"{job_id}{remote_suffix}", out_path)
        except Exception as e:
            print(f"[{tag}] volume download failed ({e}); falling back to HTTP")
    with requests.get(job_url, stream=True, timeout=(10, read_timeout)) as r:
        return _stream_to_file(r, out_path)


def _check_transient_budget(tag: str, started_at: Optional[float],
                            budget_s: int, what: str) -> float:
    """Track how long we've been seeing transient errors. Raises if over budget.

    Returns the (possibly newly-set) start timestamp.
    """
    started_at = started_at or time.time()
    elapsed = time.time() - started_at
    if elapsed > budget_s:
        raise RuntimeError(
            f"{tag} transient errors for {elapsed:.0f}s (>{budget_s}s): {what}")
    print(f"[{tag}] {what}; retrying ({elapsed:.0f}s/{budget_s}s)")
    return started_at


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
    # ── 1. Submit with exponential backoff ────────────────────────────────────
    print(f"[{tag}] submit -> {submit_url}")
    job_id: Optional[str] = None
    last_err: Optional[Exception] = None
    for attempt in range(1, submit_retries + 1):
        opened = {k: open(v, "rb") for k, v in files.items()}
        try:
            r = requests.post(submit_url, files=opened, data=form,
                              timeout=(10, submit_read_timeout))
            if r.status_code >= 500:
                raise requests.exceptions.HTTPError(
                    f"{r.status_code}: {r.text[:200]}", response=r)
            r.raise_for_status()
            job_id = r.json()["job_id"]
            break
        except requests.exceptions.RequestException as e:
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

    # ── 2. Poll until 200 or budget exhausted ─────────────────────────────────
    job_url = f"{base_url}/jobs/{job_id}"
    transient_since: Optional[float] = None
    empty_count = 0

    while True:
        try:
            resp = requests.get(job_url, timeout=(10, download_read_timeout))
        except requests.exceptions.RequestException as e:
            transient_since = _check_transient_budget(
                tag, transient_since, cold_start_budget_s,
                f"poll error ({type(e).__name__})")
            time.sleep(poll_every)
            continue

        status = resp.status_code

        if status == 202:
            resp.close()
            transient_since = None
            print(f"[{tag}] pending...")
            time.sleep(poll_every)
            continue

        if status >= 500:
            body = resp.text[:200]
            resp.close()
            transient_since = _check_transient_budget(
                tag, transient_since, cold_start_budget_s,
                f"{status} (cold start?): {body[:120]}")
            time.sleep(poll_every)
            continue

        if status != 200:
            body = resp.text[:500]
            resp.close()
            raise RuntimeError(f"{tag} error {status}: {body}")

        # 200 — download. Volume-first if available, HTTP fallback inside.
        resp.close()
        written = _download_artifact(
            tag, job_url, out_path, volume_name, job_id,
            remote_suffix, download_read_timeout)

        if written < min_bytes:
            empty_count += 1
            print(f"[{tag}] WARN: 200 with only {written} bytes; "
                  f"retry {empty_count}/5")
            if empty_count >= 5:
                raise RuntimeError(
                    f"{tag} kept returning empty artifact ({written} bytes); "
                    f"job may have crashed silently. Check worker logs.")
            time.sleep(poll_every)
            continue

        print(f"[{tag}] done -> {out_path} ({written/1e6:.2f} MB)")
        return out_path


def trellis2(image_path: str | Path,
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


def trellis2_texture(image_path: str | Path,
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


def hunyuan3d2(image_path: str | Path,
               out_path: str | Path | None = None,
               seed: int = 42,
               steps: int = 50,
               guidance_scale: float = 5.5,
               octree_resolution: int = 256,
               rembg: bool = True,
               **job_kwargs) -> str:
    """Hunyuan3D-2: image -> textured GLB (shape + paint)."""
    out_path = str(out_path or f"hunyuan3d2_{int(time.time())}.glb")
    return _run_modal_job(
        tag="hunyuan3d2",
        submit_url=f"{HUNYUAN3D2_GEN_URL}/generate",
        base_url=HUNYUAN3D2_GEN_URL,
        files={"image": str(image_path)},
        form={
            "seed": str(seed),
            "steps": str(steps),
            "guidance_scale": str(guidance_scale),
            "octree_resolution": str(octree_resolution),
            "rembg": str(rembg).lower(),
        },
        out_path=out_path,
        volume_name="hunyuan3d-2-jobs",
        **job_kwargs,
    )


def test():
    generate_concept_image("a cat with a hat and boots", "media/2d_outputs/cat.jpeg")
    restyle_to_objaverse("media/2d_outputs/cat.jpeg", "media/2d_outputs/restylized_cat.jpeg")
    hunyuan3d2("media/2d_outputs/restylized_cat.jpeg", "media/3d_outputs/cat.glb")