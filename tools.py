from dotenv import load_dotenv
load_dotenv()

import io
import os
import random
import time
from pathlib import Path
from typing import Optional, Literal
from PIL import Image
from pydantic import BaseModel

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google import genai
from google.genai import types
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch

import trimesh

WORKSPACE = os.environ.get("TRELLIS_WORKSPACE", "")
TRELLIS_GEN_URL = f"https://{WORKSPACE}--trellis2-generator-web.modal.run"
TRELLIS_TEX_URL = f"https://{WORKSPACE}--trellis2-texturer-web.modal.run"
PARTCRAFTER_OBJ_URL = f"https://{WORKSPACE}--partcrafter-objectgenerator-web.modal.run"
PARTCRAFTER_SCENE_URL = f"https://{WORKSPACE}--partcrafter-scenegenerator-web.modal.run"
HUNYUAN3D2_GEN_URL = f"https://{WORKSPACE}--hunyuan3d-2-generator-web.modal.run"


# --- Objaverse restyle, decomposed into independent axes ---
#
# The restyle intervention is the subject of the preprocessing ablation: we
# measure the marginal contribution of each axis to image-to-3D quality. Each
# axis is a self-contained clause so we can toggle any subset (e.g. leave-one-out)
# via build_restyle_prompt(). The order here is the canonical "all-on" order.
RESTYLE_AXES: dict[str, str] = {
    "background":      "place it on a plain neutral background with no scene context",
    "framing":         "center the object so it is closed up, almost filling the frame",
    "view":            ("show it from a two-quarter view, strictly from the horizontal; "
                        "do not tilt the view of the object"),
    "lighting":        "use even studio lighting with no shadows on the ground",
    "isolation":       "show a single isolated object only, with no text",
    "part_visibility": ("make the parts of the object distinctly visible, twisting the "
                        "object slightly from its natural pose if needed"),
}

RESTYLE_PREAMBLE = (
    "Restyle this image as a single 3D asset rendered in the style of the "
    "Objaverse dataset"
)


def build_restyle_prompt(axes: "list[str] | None" = None) -> str:
    """Compose the restyle prompt from a subset of RESTYLE_AXES.

    axes: ordered list of axis names to enable. None means all axes (the
          canonical "all-on" condition). An empty list yields the bare
          preamble (style transfer with no normalization axes).

    Unknown axis names raise ValueError so a typo in an experiment config
    fails loudly instead of silently dropping an axis.
    """
    if axes is None:
        axes = list(RESTYLE_AXES)
    unknown = [a for a in axes if a not in RESTYLE_AXES]
    if unknown:
        raise ValueError(
            f"unknown restyle axis/axes: {unknown}; valid: {list(RESTYLE_AXES)}")
    clauses = [RESTYLE_AXES[a] for a in axes]
    if not clauses:
        return RESTYLE_PREAMBLE + "."
    return RESTYLE_PREAMBLE + ": " + ", ".join(clauses) + "."


OBJAVERSE_STYLE_PROMPT = build_restyle_prompt()

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


def edit_image(image_path: str | Path,
               instruction: str,
               out_path: str | Path | None = None) -> str:
    """Edit an image per a natural-language instruction (Gemini image editing).

    Unlike restyle (which normalizes to Objaverse style), this applies an
    arbitrary targeted change the user asked for, e.g. "make the background
    white", "remove the text", "make it look more realistic". Returns the saved
    PNG path.
    """
    image_path = str(image_path)
    out_path = out_path or f"edited_{int(time.time())}.png"
    print(f"[gemini] edit: {image_path} :: {instruction!r}")
    src = _gemini().files.upload(file=image_path)
    resp = _gemini().models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=[src, instruction],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    data = _extract_image_bytes(resp)
    if not data:
        raise RuntimeError("Gemini returned no image for edit")
    path = _save_png(data, out_path)
    print(f"[gemini] saved {path}")
    return path


TAVILY_URL = "https://api.tavily.com/search"


def _tavily(payload: dict) -> dict:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set (check your .env)")
    r = _session.post(TAVILY_URL, json={"api_key": key, **payload}, timeout=30)
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
    r = _session.get(
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


SUBMIT_MAX_RETRIES = 8
SUBMIT_BACKOFF_BASE_S = 10.0
SUBMIT_BACKOFF_MAX_S = 60.0
RETRY_JITTER = 0.2
POLL_INTERVAL_S = 10.0
POLL_TRANSIENT_BUDGET_S = 900
TOTAL_DEADLINE_S = 3600
EMPTY_RETRY_MAX = 5
MIN_ARTIFACT_BYTES = 1024
RETRYABLE_4XX = {408, 429}


def _jittered(seconds: float, frac: float = RETRY_JITTER) -> float:
    """Add ±frac random jitter to a sleep duration. Floor at 0.1s."""
    return max(0.1, seconds * (1 + random.uniform(-frac, frac)))


def _retry_after(resp) -> Optional[float]:
    """Honor `Retry-After` header (seconds form). Returns None if absent."""
    if resp is None:
        return None
    val = resp.headers.get("Retry-After")
    if val and val.strip().isdigit():
        return float(val.strip())
    return None


def _is_retryable_status(status: int) -> bool:
    """5xx is always retryable; 4xx is terminal except 408/429."""
    return status >= 500 or status in RETRYABLE_4XX


def _build_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=Retry(
            total=2,
            connect=2,
            read=0,
            backoff_factor=0.5,
            status_forcelist=[],
            allowed_methods=frozenset(["GET", "POST"]),
        ),
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


_session = _build_session()


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
    with _session.get(job_url, stream=True, timeout=(10, read_timeout)) as r:
        return _stream_to_file(r, out_path)


def _backoff(attempt: int) -> float:
    """Exponential backoff with cap, jittered."""
    return _jittered(min(SUBMIT_BACKOFF_MAX_S,
                         SUBMIT_BACKOFF_BASE_S * 2 ** (attempt - 1)))


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
                   poll_every: float = POLL_INTERVAL_S,
                   submit_retries: int = SUBMIT_MAX_RETRIES,
                   submit_read_timeout: int = 600,
                   download_read_timeout: int = 600,
                   min_bytes: int = MIN_ARTIFACT_BYTES,
                   cold_start_budget_s: int = POLL_TRANSIENT_BUDGET_S,
                   total_deadline_s: int = TOTAL_DEADLINE_S) -> str:
    """Generic submit -> poll -> download for any TRELLIS-style Modal app.

    retry behavior:
      * Jittered exponential backoff on submit; respects `Retry-After`.
      * 4xx (except 408/429) fails immediately - auth/payload errors don't retry.
      * 5xx + connection errors share a transient budget (cold-start window).
      * Hard total deadline from submit time prevents pathological infinite polls.
      * Empty artifact (200 with < min_bytes) retried up to EMPTY_RETRY_MAX
        - covers the volume eventual-consistency window after worker commit.

    `files` is a dict of {field: open_path_str}. We open and stream each.
    Returns the path to the downloaded artifact.
    """
    deadline = time.time() + total_deadline_s

    print(f"[{tag}] submit -> {submit_url}")
    job_id: Optional[str] = None
    last_err: Optional[Exception] = None
    for attempt in range(1, submit_retries + 1):
        if time.time() >= deadline:
            raise RuntimeError(f"{tag} submit deadline reached ({total_deadline_s}s)")
        opened = {k: open(v, "rb") for k, v in files.items()}
        wait: float = 0.0
        try:
            r = _session.post(submit_url, files=opened, data=form,
                              timeout=(10, submit_read_timeout))
            if r.status_code == 200:
                job_id = r.json()["job_id"]
                break
            if not _is_retryable_status(r.status_code):
                raise RuntimeError(
                    f"{tag} submit non-retryable {r.status_code}: {r.text[:200]}")
            wait = _retry_after(r) or _backoff(attempt)
            last_err = RuntimeError(f"{r.status_code}: {r.text[:200]}")
        except requests.exceptions.RequestException as e:
            last_err = e
            wait = _backoff(attempt)
        finally:
            for fh in opened.values():
                fh.close()
        print(f"[{tag}] submit {attempt}/{submit_retries} failed: "
              f"{last_err}; retrying in {wait:.1f}s")
        time.sleep(wait)
    if job_id is None:
        raise RuntimeError(f"{tag} submit failed after {submit_retries}: {last_err}")
    print(f"[{tag}] job={job_id}")

    job_url = f"{base_url}/jobs/{job_id}"
    transient_since: Optional[float] = None
    empty_count = 0

    while True:
        if time.time() >= deadline:
            raise RuntimeError(
                f"{tag} poll deadline reached ({total_deadline_s}s) - job {job_id}")

        try:
            resp = _session.get(job_url, timeout=(10, download_read_timeout))
        except requests.exceptions.RequestException as e:
            transient_since = _check_transient_budget(
                tag, transient_since, cold_start_budget_s,
                f"poll error ({type(e).__name__})")
            time.sleep(_jittered(poll_every))
            continue

        status = resp.status_code

        if status == 202:
            resp.close()
            transient_since = None
            print(f"[{tag}] pending...")
            time.sleep(_jittered(poll_every))
            continue

        if _is_retryable_status(status):
            wait = _retry_after(resp) or _jittered(poll_every)
            body = resp.text[:200]
            resp.close()
            transient_since = _check_transient_budget(
                tag, transient_since, cold_start_budget_s,
                f"{status}: {body[:120]}")
            time.sleep(wait)
            continue

        if status != 200:
            body = resp.text[:500]
            resp.close()
            raise RuntimeError(f"{tag} error {status}: {body}")

        resp.close()
        written = _download_artifact(
            tag, job_url, out_path, volume_name, job_id,
            remote_suffix, download_read_timeout)

        if written < min_bytes:
            empty_count += 1
            print(f"[{tag}] WARN: 200 with only {written} bytes; "
                  f"retry {empty_count}/{EMPTY_RETRY_MAX}")
            if empty_count >= EMPTY_RETRY_MAX:
                raise RuntimeError(
                    f"{tag} kept returning empty artifact ({written} bytes); "
                    f"job may have crashed silently. Check worker logs.")
            time.sleep(_jittered(poll_every))
            continue

        print(f"[{tag}] done -> {out_path} ({written/1e6:.2f} MB)")
        return out_path


def _run_local_trellis(base_url: str, image_path, out_path: str,
                       seed: int, decimation_target: int,
                       texture_size: int, remesh: bool) -> str:
    """Post the image to a local TRELLIS HTTP server (eval/local_trellis_server.py)
    and save the returned GLB. Synchronous — no Modal submit/poll/volume dance."""
    print(f"[trellis2] local -> {base_url}/generate ({image_path})")
    with open(str(image_path), "rb") as fh:
        resp = _session.post(
            f"{base_url.rstrip('/')}/generate",
            files={"image": fh},
            data={"seed": str(seed), "decimation_target": str(decimation_target),
                  "texture_size": str(texture_size), "remesh": str(remesh).lower()},
            timeout=1800,
        )
    resp.raise_for_status()
    with open(out_path, "wb") as out:
        out.write(resp.content)
    print(f"[trellis2] done -> {out_path} ({len(resp.content)/1e6:.2f} MB)")
    return out_path


def trellis2(image_path: str | Path,
             out_path: str | Path | None = None,
             pipeline_type: str = "1024_cascade",
             remesh: bool = True,
             seed: int = 42,
             decimation_target: int = 1_000_000,
             texture_size: int = 4096,
             **job_kwargs) -> str:
    """TRELLIS.2: image -> GLB. Uses a local TRELLIS server if LOCAL_TRELLIS_URL
    is set (the GPU box), otherwise submits a Modal job."""
    out_path = str(out_path or f"model_{int(time.time())}.glb")
    local_url = os.environ.get("LOCAL_TRELLIS_URL")
    if local_url:
        return _run_local_trellis(local_url, image_path, out_path,
                                  seed, decimation_target, texture_size, remesh)
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


_FACE_DIRS = {
    "front":  ( 0,  0,  1),
    "back":   ( 0,  0, -1),
    "left":   (-1,  0,  0),
    "right":  ( 1,  0,  0),
    "top":    ( 0,  1,  0),
    "bottom": ( 0, -1,  0),
}
_CORNER_DIRS = {
    "front_top_right":    ( 1,  1,  1),
    "front_top_left":     (-1,  1,  1),
    "front_bottom_right": ( 1, -1,  1),
    "front_bottom_left":  (-1, -1,  1),
    "back_top_right":     ( 1,  1, -1),
    "back_top_left":      (-1,  1, -1),
    "back_bottom_right":  ( 1, -1, -1),
    "back_bottom_left":   (-1, -1, -1),
}
_VIEW_PRESETS = {
    "default": list(_FACE_DIRS),                       # 6 axis-aligned
    "corners": list(_CORNER_DIRS),                     # 8 diagonals
    "all":     list(_FACE_DIRS) + list(_CORNER_DIRS),  # 14 total
}


def _look_at(camera_position, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """Build a camera-to-world matrix (OpenGL convention: looking down -Z)."""
    import numpy as np
    cam = np.array(camera_position, dtype=float)
    tgt = np.array(target,          dtype=float)
    u   = np.array(up,              dtype=float)

    forward = tgt - cam
    forward /= np.linalg.norm(forward)

    if abs(np.dot(forward, u)) > 0.999:
        u = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, u)) > 0.999:
            u = np.array([1.0, 0.0, 0.0])

    right = np.cross(forward, u); right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward
    mat[:3, 3] = cam
    return mat


def render_mesh_views(mesh_path: str | Path,
                      out_dir: str | Path,
                      views: str | list[str] = "default",
                      image_size: int = 512,
                      distance: float = 3.0) -> dict[str, str]:
    """Render named camera views of a mesh as PNGs (thread-safe wrapper).

    pyrender's pyglet backend must own the *main* thread on macOS. The agent
    calls this from a Django/LangGraph worker thread, where pyglet's attempt to
    touch the AppKit main menu crashes the whole process. To stay safe from any
    thread we run the actual render in a short-lived subprocess (which owns its
    own main thread) when we're not already on the main thread; on the main
    thread we render in-process with no overhead.

    Args:
        mesh_path:  any format trimesh can load (.glb/.obj/.ply/.stl).
        out_dir:    directory to write `<view>.png` files into. Created if missing.
        views:      one of "default" (6 face views), "corners" (8 diagonals),
                    "all" (14), or a custom list of view names. Names come from
                    {front, back, left, right, top, bottom, front_top_right, ...}.
        image_size: square render resolution.
        distance:   camera distance from origin (mesh is normalized to fit a unit
                    cube, so 3.0 keeps the whole mesh comfortably in frame).

    Returns: dict mapping view name -> saved PNG path.
    """
    import threading

    if threading.current_thread() is threading.main_thread():
        return _render_mesh_views_local(mesh_path, out_dir, views, image_size, distance)

    import json
    import subprocess
    import sys

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--render-views",
         str(mesh_path), str(out_dir), json.dumps(views),
         str(image_size), str(distance)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"render subprocess failed (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:]}")
    # The render impl logs "[views] ..." lines; the result dict is the final
    # non-empty stdout line, emitted as JSON by the __main__ entrypoint below.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    try:
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"render subprocess produced no result ({e}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-2000:]}")


def _render_mesh_views_local(mesh_path: str | Path,
                             out_dir: str | Path,
                             views: str | list[str] = "default",
                             image_size: int = 512,
                             distance: float = 3.0) -> dict[str, str]:
    """In-process render. ONLY safe on the main thread (see render_mesh_views)."""
    # Lazy imports — keeps tools.py importable on machines without GL.
    import numpy as np
    import pyrender

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve view selection.
    if isinstance(views, str):
        if views not in _VIEW_PRESETS:
            raise ValueError(
                f"unknown views preset {views!r}; "
                f"use one of {list(_VIEW_PRESETS)} or pass a list of names")
        view_names = _VIEW_PRESETS[views]
    else:
        view_names = list(views)
    all_dirs = {**_FACE_DIRS, **_CORNER_DIRS}
    unknown = [v for v in view_names if v not in all_dirs]
    if unknown:
        raise ValueError(f"unknown view name(s): {unknown}; "
                         f"valid: {list(all_dirs)}")

    # Load mesh, normalize into a unit cube around the origin.
    loaded = trimesh.load(str(mesh_path))
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])

    if isinstance(loaded, trimesh.Scene):
        geometries = [g.copy() for g in loaded.geometry.values()]
        combined = trimesh.util.concatenate(geometries)
        center = combined.bounds.mean(axis=0)
        scale = 2.0 / max(combined.extents)
        for g in geometries:
            g.apply_translation(-center)
            g.apply_scale(scale)
            scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))
    else:
        mesh = loaded.copy()
        center = mesh.bounds.mean(axis=0)
        scale = 2.0 / max(mesh.extents)
        mesh.apply_translation(-center)
        mesh.apply_scale(scale)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_node = scene.add(camera)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
        parent_node=cam_node,
    )

    renderer = pyrender.OffscreenRenderer(image_size, image_size)
    paths: dict[str, str] = {}
    try:
        for name in view_names:
            pos = tuple(c * distance for c in all_dirs[name])
            scene.set_pose(cam_node, _look_at(pos))
            color, _ = renderer.render(scene)
            out_path = str(out_dir / f"{name}.png")
            Image.fromarray(color).save(out_path)
            paths[name] = out_path
            print(f"[views] {name} -> {out_path}")
    finally:
        renderer.delete()
    return paths


def compose_scene(placements: "list[dict] | str",
                  out_path: str | Path | None = None,
                  gap: float = 0.15) -> str:
    """Place existing meshes into ONE scene in a shared coordinate space.

    Deterministic composition — no model, no GPU. Each input mesh keeps its own
    geometry + material (we do NOT concatenate, so textures survive).

    placements: list of dicts (or a JSON string of that list), one per object:
        {"mesh_path": str,            # required, a .glb/.obj/...
         "x","y","z": float,          # optional world coords of the object's
                                      #   center-bottom (meters)
         "scale": float,              # optional, default 1.0
         "rot_z_deg": float}          # optional yaw about the vertical (Z) axis
    If NO item supplies x/y/z, objects are auto-arranged left-to-right in a row,
    bottoms on the ground (z=0), centered on y=0, separated by `gap`. Otherwise
    each object's center-bottom is placed at its (x,y,z) (missing coords -> 0).

    Returns the saved combined-GLB path.
    """
    import json
    import numpy as np
    from trimesh.transformations import (translation_matrix, scale_matrix,
                                         rotation_matrix, concatenate_matrices)

    if isinstance(placements, str):
        placements = json.loads(placements)
    out_path = str(out_path or f"scene_{int(time.time())}.glb")

    scene = trimesh.Scene()
    cursor_x = 0.0
    for i, pl in enumerate(placements):
        loaded = trimesh.load(str(pl["mesh_path"]))
        # dump() bakes each sub-mesh's scene transform into world space and keeps
        # per-part visuals; a single Trimesh becomes a 1-element list.
        parts = loaded.dump() if isinstance(loaded, trimesh.Scene) else [loaded.copy()]

        mins = np.min([p.bounds[0] for p in parts], axis=0)
        maxs = np.max([p.bounds[1] for p in parts], axis=0)
        center = (mins + maxs) / 2.0
        ext = maxs - mins

        s = float(pl.get("scale", 1.0))
        rz = float(pl.get("rot_z_deg", 0.0))

        if any(k in pl for k in ("x", "y", "z")):
            tx, ty, tz = float(pl.get("x", 0.0)), float(pl.get("y", 0.0)), float(pl.get("z", 0.0))
        else:
            tx, ty, tz = cursor_x + ext[0] * s / 2.0, 0.0, 0.0
            cursor_x += ext[0] * s + gap

        # Move center-bottom to origin, scale, yaw, then translate to the target.
        m = concatenate_matrices(
            translation_matrix([tx, ty, tz]),
            rotation_matrix(np.radians(rz), [0, 0, 1]),
            scale_matrix(s),
            translation_matrix([-center[0], -center[1], -mins[2]]),
        )
        for j, p in enumerate(parts):
            pc = p.copy()
            pc.apply_transform(m)
            scene.add_geometry(pc, geom_name=f"obj{i}_{j}")
        print(f"[compose] obj{i} {Path(pl['mesh_path']).name} "
              f"size={np.round(ext, 2).tolist()} -> ({tx:.2f},{ty:.2f},{tz:.2f})")

    scene.export(out_path)
    print(f"[compose] {len(placements)} objects -> {out_path}")
    return out_path


class AlignmentReport(BaseModel):
    accept: bool
    score: float
    summary: str
    next_action: Literal["proceed", "regenerate"]
    worst_view: str | None
    per_view: dict[str, float]


_MODEL = None
_PROCESSOR = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load(model_name: str = "openai/clip-vit-large-patch14"):
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        _MODEL = CLIPModel.from_pretrained(model_name).to(_DEVICE).eval()
        _PROCESSOR = CLIPProcessor.from_pretrained(model_name)
    return _MODEL, _PROCESSOR


@torch.no_grad()
def check_alignment(
    view_paths: dict[str, str] | list[str | Path],
    prompt: str,
    *,
    accept_threshold: float = 0.22,
    min_view_threshold: float = 0.15,
) -> AlignmentReport:
    if isinstance(view_paths, dict):
        items = list(view_paths.items())
    else:
        items = [(Path(p).stem, str(p)) for p in view_paths]

    if not items:
        return AlignmentReport(
            accept=False, score=0.0,
            summary="No views provided.",
            next_action="regenerate",
            worst_view=None, per_view={},
        )

    model, processor = _load()
    images = [Image.open(p).convert("RGB") for _, p in items]

    inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True).to(_DEVICE)
    out = model(**inputs)
    img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    sims = (img_emb @ txt_emb.T).squeeze(-1).cpu().tolist()

    per_view = {name: round(float(s), 3) for (name, _), s in zip(items, sims)}
    mean = sum(per_view.values()) / len(per_view)
    worst_name = min(per_view, key=per_view.get)
    worst_score = per_view[worst_name]

    if mean < accept_threshold or worst_score < min_view_threshold:
        if worst_score < min_view_threshold:
            summary = (
                f"The '{worst_name}' view does not match the prompt "
                f"(score {worst_score:.2f}); likely a Janus-style inconsistency."
            )
        else:
            summary = f"Mesh weakly matches the prompt (mean score {mean:.2f})."
        return AlignmentReport(
            accept=False,
            score=round(mean, 2),
            summary=summary,
            next_action="regenerate",
            worst_view=worst_name,
            per_view=per_view,
        )

    return AlignmentReport(
        accept=True,
        score=round(mean, 2),
        summary=f"Mesh matches the prompt across all views (mean {mean:.2f}).",
        next_action="proceed",
        worst_view=worst_name,
        per_view=per_view,
    )


if __name__ == "__main__":
    # Subprocess entrypoint used by render_mesh_views() when it is called off the
    # main thread (see that function). Renders on this fresh process's main
    # thread and prints the {view: path} dict as JSON on the final stdout line.
    import sys as _sys

    if len(_sys.argv) >= 2 and _sys.argv[1] == "--render-views":
        import json as _json

        _mesh_path = _sys.argv[2]
        _out_dir = _sys.argv[3]
        _views = _json.loads(_sys.argv[4]) if len(_sys.argv) > 4 else "default"
        _image_size = int(_sys.argv[5]) if len(_sys.argv) > 5 else 512
        _distance = float(_sys.argv[6]) if len(_sys.argv) > 6 else 3.0
        _paths = _render_mesh_views_local(
            _mesh_path, _out_dir, _views, _image_size, _distance)
        print(_json.dumps(_paths))