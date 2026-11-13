#!/usr/bin/env python
"""Restyle-preprocessing ablation sweep — agent-level evaluation.

Drives the 3D agent (agent.process_chat_stream) with each prompt under each
restyle condition and scores the resulting mesh.

For each (prompt, condition) it:
  1. builds a system instruction that forces the restyle axes for the condition,
  2. sends the prompt to the agent and waits for a mesh to be produced,
  3. renders multi-view PNGs from the mesh,
  4. scores with CLIP, a VLM judge (Gemini, 1-10; NOT the published VLM judge), and ULIP-2,
and appends one row to a CSV. Already-present rows are skipped so the sweep is
resumable after a crash.

Run from the repo root:
    .venv/bin/python eval/run_pilot.py --run-id R1 --input retrieved

Outputs go to eval/runs/<run-id>/ (results_<input>.csv, run_meta.json,
traces/, concepts/). Rerunning the same --run-id resumes; a NEW --run-id never
reuses meshes or images from another run.

Integrity guarantees (rerun for camera-ready, 2026-11):
  * The condition's restyle prompt is ENFORCED at the tool layer: the agent's
    tool_restyle_to_objaverse has no style argument, so before this fix every
    restyled condition silently used the all-on default prompt. `raw` makes
    the restyle tool an identity pass-through.
  * The user message sent to the agent is IDENTICAL across conditions, so any
    routing difference comes from the image, not the instruction text.
  * Generated mode: one concept image per prompt per run, shared by all 8
    conditions (paired design) and generated from the verbatim caption.
  * Every row logs run id, pinned + resolved model versions, seeds, the full
    tool-call trace (routing decision, endpoint, args) and HITL interrupts.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Django is required by agent.py (LangGraph checkpointer uses the DB).
import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

import tools  # noqa: E402
import agent  # noqa: E402
from agent import process_chat_stream, resume_chat_stream  # noqa: E402

# Monkey-patch render_mesh_views to always run in a subprocess.
# The agent calls tool_render_mesh_views during its pipeline, which lazily
# imports pyrender — and on macOS importing pyrender/pyglet from a LangGraph
# background thread crashes AppKit ("setting the main menu on a non-main
# thread"). Routing every render call through a subprocess gives pyrender its
# own clean main thread.
#
# Both tools.render_mesh_views AND agent.render_mesh_views must be patched:
# agent.py did `from tools import render_mesh_views`, so it holds its own
# binding that the tools.* patch would not touch.
_RENDER_HELPER = REPO_ROOT / "eval" / "_render_helper.py"

def _safe_render(mesh_path, out_dir, views="default", **kwargs):
    import subprocess
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(_RENDER_HELPER), str(mesh_path), str(out_dir)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"render subprocess failed:\n{result.stderr}")
    paths = [p for p in result.stdout.strip().splitlines() if p.endswith(".png")]
    return {Path(p).stem: p for p in paths}

tools.render_mesh_views = _safe_render
agent.render_mesh_views = _safe_render

# Shorten Modal job deadlines FOR THE EVAL so a flaky/wedged backend (e.g. a
# 0-byte volume download + ChunkedEncodingError poll loop) fails fast and the
# sweep keeps moving, instead of burning the default 900s transient budget /
# 3600s hard deadline on one cell. Failed cells are retried on the next resume
# pass, where the transient backend usually succeeds. A healthy generation
# runs in ~1-3 min, so these caps leave ample headroom.
_orig_run_modal_job = tools._run_modal_job

def _fastfail_run_modal_job(*args, **kwargs):
    # Generous deadlines: a cold-start TRELLIS generation can legitimately take
    # many minutes, so a tight cap kills real jobs mid-generation (symptom:
    # restyle succeeds but no .glb is ever produced). The actual anti-hang
    # protection is the bounded download below — NOT a short job deadline.
    kwargs.setdefault("cold_start_budget_s", 400)   # tolerate slow cold starts
    kwargs.setdefault("total_deadline_s", 1200)     # 20 min safety net
    kwargs.setdefault("download_read_timeout", 120) # bound HTTP-fallback reads
    return _orig_run_modal_job(*args, **kwargs)

tools._run_modal_job = _fastfail_run_modal_job

# Bound the volume download. tools._download_via_volume shells out to
# `modal volume get` with NO timeout, so a stalled modal CLI hangs the sweep
# forever. Re-implement with a timeout; on TimeoutExpired the caller
# (_download_artifact) catches it and falls back to the HTTP path, which now
# also has a bounded read timeout (above).
def _download_via_volume_timeout(volume_name, remote_name, out_path, _timeout=180):
    import os as _os, shutil, subprocess
    if shutil.which("modal") is None:
        raise RuntimeError("modal CLI not found on PATH")
    if _os.path.exists(out_path):
        _os.remove(out_path)
    subprocess.run(
        ["modal", "volume", "get", volume_name, remote_name, out_path],
        check=True, timeout=_timeout,
    )
    return _os.path.getsize(out_path)

tools._download_via_volume = _download_via_volume_timeout

# Tag eval-produced media with the prompt stem + condition so files are
# identifiable instead of bare timestamps. Set per run_one(); the agent's
# _stamp() reads it via the patch below. Prefix stays first so the
# 'objaverse_*.png' and '*.glb' detection globs still match.
_EVAL_TAG = ""

def _eval_stamp(prefix: str, ext: str) -> str:
    ts = int(time.time() * 1000)  # ms granularity avoids same-second collisions
    name = (f"{prefix}_{_EVAL_TAG}_{ts}.{ext}" if _EVAL_TAG
            else f"{prefix}_{ts}.{ext}")
    return str(agent.OUT / name)

agent._stamp = _eval_stamp

# ---------------------------------------------------------------------------
# Condition enforcement at the tool layer
# ---------------------------------------------------------------------------
# agent.tool_restyle_to_objaverse(image_path) calls the module-level name
# agent.restyle_to_objaverse with the DEFAULT (all-on) prompt and the agent has
# no way to pass a different one. So we patch that binding: it uses the
# current condition's prompt, or is an identity pass-through for `raw`.
_EVAL_STYLE_PROMPT: str | None = None   # None => raw (identity)
_EVAL_CAPTION: str = ""
_CONCEPT_CACHE_PATH: Path | None = None  # per (run, prompt) shared concept image
_TOOL_LOG: list[dict] = []               # what the patched tools actually did
_REVIEW: bool = False                    # review conditions: agent may score + regenerate
MAX_GENS_REVIEW = 3                      # 1 generation + at most 2 regenerations
_GEN_COUNT = 0                           # 3D generation calls in the current cell
_LAST_MESH: str | None = None
_AGENT_ERRORS: list[str] = []            # "Agent error: ..." texts from the current cell


class GeminiOutOfCredit(SystemExit):
    """Out of credit: every remaining cell would fail too — stop the sweep."""

_orig_restyle = tools.restyle_to_objaverse
_orig_concept = tools.generate_concept_image

# ---------------------------------------------------------------------------
# Per-cell artifact cache: a rerun of the same cell (crash, retry pass) reuses
# anything identical it already produced instead of paying Gemini / the GPU
# again. Keyed by WHAT was computed — input image CONTENT + exact parameters —
# so a review retry with a new seed or an edited image is still generated
# fresh. Scoped per OBJECT within a run (run id + input mode + object), so
# identical work is shared across conditions — e.g. all_on_review starts from
# exactly the restyled image and seed-42 mesh of all_on, and the review effect
# is purely what the review loop adds. Conditions never collide because the
# key includes the input content and every parameter (raw's input is the photo,
# all_on's the restyle with the full prompt, loo_lighting's a different
# prompt). Nothing is ever reused across runs or input modes.
# ---------------------------------------------------------------------------
_CACHE_DIR = agent.OUT / ".eval_cache"


def _file_hash(path) -> str:
    import hashlib
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return f"missing:{path}"


def _cache_file() -> Path:
    obj_scope = (_EVAL_TAG or "untagged").split("__")[0]   # <run>_<mode><stem>
    return _CACHE_DIR / f"{obj_scope}.json"


def _trace_file(tag: str) -> Path:
    """One record per CELL ({mesh, restyled, trace}). Not next to the mesh:
    with per-object reuse, two cells can share the same mesh file."""
    return agent.OUT / ".eval_traces" / f"{tag}.json"


def _cache_get(key: str) -> str | None:
    f = _cache_file()
    if not f.exists():
        return None
    hit = json.loads(f.read_text()).get(key)
    if hit and Path(hit).exists() and Path(hit).stat().st_size > 1024:
        return hit
    return None


def _cache_put(key: str, path: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = _cache_file()
    d = json.loads(f.read_text()) if f.exists() else {}
    d[key] = str(path)
    f.write_text(json.dumps(d, indent=1))


def _eval_restyle(image_path, out_path=None, style_prompt=None):
    if _EVAL_STYLE_PROMPT is None:
        _TOOL_LOG.append({"tool": "restyle", "mode": "identity(raw)",
                          "input": str(image_path)})
        print(f"[eval] raw condition: restyle is identity for {image_path}")
        return str(image_path)
    key = f"restyle|{_file_hash(image_path)}|{_EVAL_STYLE_PROMPT}"
    hit = _cache_get(key)
    entry = {"tool": "restyle", "input": str(image_path), "prompt": _EVAL_STYLE_PROMPT}
    if hit:
        print(f"[eval] reusing restyled image {Path(hit).name}")
        _TOOL_LOG.append({**entry, "mode": "enforced", "reused": True, "path": hit})
        return hit
    out = _orig_restyle(image_path, out_path=out_path, style_prompt=_EVAL_STYLE_PROMPT)
    _cache_put(key, out)
    _TOOL_LOG.append({**entry, "mode": "enforced", "path": out})
    return out


def _eval_concept(prompt, out_path=None):
    """Concept generation pinned to the verbatim caption and shared across all
    conditions of a prompt within a run (paired design)."""
    import shutil
    out_path = str(out_path or agent._stamp("concept", "png"))
    entry = {"tool": "concept", "agent_prompt": prompt, "used_prompt": _EVAL_CAPTION}
    n_prev = sum(1 for e in _TOOL_LOG if e.get("tool") == "concept")
    if _REVIEW and n_prev > 0:
        # a review retry asked for a NEW concept image: generate one with a
        # different (logged) seed instead of returning the shared cached image
        base = int(os.environ.get("GEMINI_SEED", "0") or 0)
        old = os.environ.get("GEMINI_SEED")
        os.environ["GEMINI_SEED"] = str(base + n_prev)
        try:
            _orig_concept(_EVAL_CAPTION or prompt, out_path=out_path)
        finally:
            if old is None:
                os.environ.pop("GEMINI_SEED", None)
            else:
                os.environ["GEMINI_SEED"] = old
        entry.update(mode="review_regenerated", seed=base + n_prev, path=out_path)
        _TOOL_LOG.append(entry)
        return out_path
    if _CONCEPT_CACHE_PATH and _CONCEPT_CACHE_PATH.exists():
        shutil.copyfile(_CONCEPT_CACHE_PATH, out_path)
        entry["mode"] = "cached"
    else:
        _orig_concept(_EVAL_CAPTION or prompt, out_path=out_path)
        if _CONCEPT_CACHE_PATH:
            _CONCEPT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out_path, _CONCEPT_CACHE_PATH)
        entry["mode"] = "generated"
    entry["path"] = out_path
    _TOOL_LOG.append(entry)
    return out_path


agent.restyle_to_objaverse = _eval_restyle
agent.generate_concept_image = _eval_concept


def _wrap_generator(name: str, fn):
    """Cap 3D generations per cell (1 normally, MAX_GENS_REVIEW in review
    conditions) and give each retry a NEW seed (default + k): with a fixed
    seed, "regenerate" from the same image would rebuild the identical mesh.
    Past the cap the call returns the last mesh unchanged (logged)."""
    import inspect
    default_seed = inspect.signature(fn).parameters["seed"].default

    def wrapped(*args, **kwargs):
        global _GEN_COUNT, _LAST_MESH
        cap = MAX_GENS_REVIEW if _REVIEW else 1
        if _GEN_COUNT >= cap and _LAST_MESH:
            _TOOL_LOG.append({"tool": "generate", "backbone": name, "mode": "capped",
                              "returned": _LAST_MESH})
            print(f"[eval] generation cap ({cap}) reached — returning last mesh")
            return _LAST_MESH
        # EVAL_SEED_OFFSET: rerun cells whose seed-42 input reproducibly hangs
        # the backbone with a different (logged) seed
        kwargs["seed"] = default_seed + int(os.environ.get("EVAL_SEED_OFFSET", "0")) + _GEN_COUNT
        _GEN_COUNT += 1
        params = {k: v for k, v in kwargs.items() if k not in ("out_path", "image_path", "mesh_path")}
        key = (f"gen|{name}|{_file_hash(kwargs.get('image_path', ''))}|"
               f"{_file_hash(kwargs['mesh_path']) if 'mesh_path' in kwargs else ''}|"
               f"{json.dumps(params, sort_keys=True, default=str)}")
        hit = _cache_get(key)
        if hit:
            print(f"[eval] reusing generated mesh {Path(hit).name} (same input + {name} params)")
            out = hit
        else:
            out = fn(*args, **kwargs)
            _cache_put(key, out)
        _LAST_MESH = out
        _TOOL_LOG.append({"tool": "generate", "backbone": name, "mode": "run",
                          "seed": kwargs["seed"], "input": kwargs.get("image_path"),
                          "path": out, "reused": bool(hit)})
        return out
    return wrapped


for _n in ("trellis2", "trellis2_texture", "hunyuan3d2", "partcrafter"):
    setattr(agent, _n, _wrap_generator(_n, getattr(tools, _n)))

_orig_edit = tools.edit_image


def _eval_edit(image_path, instruction, out_path=None):
    _TOOL_LOG.append({"tool": "edit_image", "input": str(image_path), "instruction": instruction})
    return _orig_edit(image_path, instruction, out_path=out_path)


agent.edit_image = _eval_edit

# VLM judge: its own pinned model at temperature 0 (deterministic as the
# API allows), independent of the agent's sampling temperature.
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", agent.AGENT_LLM_MODEL)
judge_llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0.0)

DATASET_DIR  = REPO_ROOT / "eval" / "dataset"
IMAGES_DIR   = DATASET_DIR / "images"   # real photos for --input photo mode
CAPTIONS_CSV = DATASET_DIR / "captions.csv"
WORK_DIR     = REPO_ROOT / "eval" / "work"
RUNS_DIR     = REPO_ROOT / "eval" / "runs"

CSV_FIELDS = [
    "prompt", "condition",
    "clip_mean", "clip_accept", "worst_view", "per_view",
    "vlm_score", "ulip_mean",
    "restyled_path", "mesh_path", "status", "error", "seconds",
    # --- provenance (added for camera-ready rerun) ---
    "run_id", "input_mode", "stem", "started_at",
    "agent_model", "agent_model_resolved", "judge_model", "image_model",
    "gemini_seed", "backbone_tool", "backbone_endpoint", "backbone_args",
    "backbone_seed", "tool_sequence", "n_tool_calls", "n_restyle_calls",
    "n_hitl_interrupts", "condition_enforced", "concept_path",
    "input_image_path", "vlm_score_raw", "reused_mesh", "trace_path",
    # --- review study ---
    "review", "n_gen_calls", "n_edit_calls", "n_concept_regens",
    "review_scores", "review_verdicts",
]

GEN_TOOLS = {"tool_trellis2", "tool_trellis2_texture", "tool_hunyuan3d2",
             "tool_partcrafter"}

# ---------------------------------------------------------------------------
# Condition → agent instruction
# ---------------------------------------------------------------------------
# For each condition we prepend a one-line system instruction to the user
# prompt that tells the agent which restyle axes to use (or none for raw).
# The agent's own judgment about which backbone to use is left untouched.

_ALL_AXES = list(tools.RESTYLE_AXES)

def conditions() -> list[tuple[str, str | None]]:
    """(condition_name, restyle_instruction_or_None)"""
    def _instruction(axes: list[str]) -> str:
        prompt = tools.build_restyle_prompt(axes)
        return (
            f"[EVAL] When restyling the input image use ONLY these axes: "
            f"{', '.join(axes)}. "
            f"Use this exact restyle prompt: {prompt}"
        )

    conds: list[tuple[str, str | None]] = [
        ("raw",    "[EVAL] Do NOT restyle the input image. Feed it directly to the 3D backbone."),
        ("all_on", _instruction(_ALL_AXES)),
    ]
    for ax in _ALL_AXES:
        remaining = [a for a in _ALL_AXES if a != ax]
        conds.append((f"loo_{ax}", _instruction(remaining)))
    # review study (2x2 with raw / all_on): same preprocessing, but the agent's
    # review loop is ON (inspect input, score result, regenerate if needed)
    conds.append(("raw_review", None))
    conds.append(("all_on_review", None))
    return conds


def base_condition(cond: str) -> str:
    return cond.removesuffix("_review")


REVIEW_SUBSET = "submission"   # review conditions run only on these objects


STYLE_PROMPTS: dict[str, str] = {"all_on": tools.build_restyle_prompt(_ALL_AXES)}
STYLE_PROMPTS["all_on_review"] = STYLE_PROMPTS["all_on"]
for _ax in _ALL_AXES:
    STYLE_PROMPTS[f"loo_{_ax}"] = tools.build_restyle_prompt(
        [a for a in _ALL_AXES if a != _ax])


def _write_run_meta(run_dir: Path, run_id: str, args) -> None:
    """Everything needed to reproduce / audit the run, written once per run
    (a resumed run appends a new entry instead of overwriting)."""
    import hashlib
    import platform
    import subprocess
    from importlib import metadata

    def _git(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    pkgs = {}
    for name in ("langchain", "langgraph", "langchain-google-genai", "google-genai",
                 "torch", "open_clip_torch", "transformers", "trimesh", "pyrender"):
        try:
            pkgs[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            pkgs[name] = None
    # resolve what the pinned agent model actually serves today
    resolved = None
    try:
        r = tools._gemini().models.generate_content(
            model=agent.AGENT_LLM_MODEL, contents="Reply with: ok")
        resolved = r.model_version
    except Exception as e:  # noqa: BLE001
        # Every cell needs Gemini (agent, concept, restyle, judge). If the
        # account is out of credit or the key is bad, stop now instead of
        # writing a whole sweep of "agent did not produce a .glb" errors.
        sys.exit(f"Gemini is not usable — stopping before any generation:\n{str(e)[:400]}")
    local_trellis = os.environ.get("LOCAL_TRELLIS_URL")
    trellis_health = None
    if local_trellis:
        try:
            import requests
            trellis_health = requests.get(local_trellis + "/", timeout=10).text[:300]
        except Exception as e:  # noqa: BLE001
            trellis_health = f"error: {e}"
    meta = {
        "run_id": run_id,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "input_mode": args.input,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_files": [l for l in _git("status", "--porcelain").splitlines()
                            if l.endswith(".py")],
        "agent_model": agent.AGENT_LLM_MODEL,
        "agent_model_resolved": resolved,
        "agent_temperature": agent.AGENT_LLM_TEMPERATURE,
        "judge_model": JUDGE_MODEL,
        "judge_temperature": 0.0,
        "image_model": tools.GEMINI_IMAGE_MODEL,
        "gemini_seed": os.environ.get("GEMINI_SEED"),
        "clip_accept_threshold": 0.22,
        "local_trellis_url_set": bool(local_trellis),
        "local_trellis_health": trellis_health,
        "style_prompts": STYLE_PROMPTS,
        "captions_sha1": hashlib.sha1(CAPTIONS_CSV.read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": pkgs,
    }
    path = run_dir / "run_meta.json"
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(meta)
    path.write_text(json.dumps(history, indent=1))
    print(f"[run] {run_id}  agent={agent.AGENT_LLM_MODEL} (resolves to {resolved})  "
          f"judge={JUDGE_MODEL}  image={tools.GEMINI_IMAGE_MODEL}")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_vlm(view_paths: list[str], caption: str,
                     raw_out: dict | None = None) -> float:
    """VLM-as-judge quality score 1–10 using the agent's Gemini model.

    Uses agent.llm (the same vision-capable Gemini that tool_inspect_image
    uses), so no extra account/key is needed beyond GEMINI_API_KEY. Sends up
    to 4 rendered views + the prompt and parses a 1–10 score from the reply.
    """
    import re
    from langchain_core.messages import HumanMessage

    content: list = [{"type": "text", "text": (
        f'These are rendered views of a 3D mesh generated from the prompt: '
        f'"{caption}". Rate the overall 3D generation quality from 1 to 10 '
        "considering: (1) geometric accuracy and completeness, (2) texture and "
        "appearance quality, (3) semantic alignment with the prompt. Reply with "
        "a single integer from 1 to 10 and nothing else."
    )}]
    for p in view_paths[:4]:
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{b64}",
        })
    try:
        resp = judge_llm.invoke([HumanMessage(content=content)])
        raw = resp.content
        text = (raw if isinstance(raw, str)
                else "".join(b.get("text", "") for b in raw
                             if isinstance(b, dict)))
        if raw_out is not None:
            raw_out["text"] = text.strip()[:200]
        m = re.search(r"\d+(?:\.\d+)?", text)  # robust to extra words
        if not m:
            return math.nan
        return max(1.0, min(10.0, float(m.group())))  # clamp to 1–10
    except Exception:  # noqa: BLE001
        return math.nan


def _score_ulip(mesh_path: str, caption: str) -> float:
    """ULIP-2 point-cloud↔text similarity via the Modal scorer (eval/ulip_modal.py).

    Samples a colored point cloud from the MESH (not the rendered views — ULIP's
    value is its 3D encoder) and scores it on GPU. Returns nan if the ULIP Modal
    app isn't deployed/reachable, so the sweep is unaffected.
    """
    # On the GPU box ULIP-2 is scored afterwards, locally, by `rerun_all.sh ulip`
    # (export_pointclouds -> local_ulip -> merge_ulip); skip the Modal call here.
    if os.environ.get("SKIP_INLINE_ULIP") == "1":
        return math.nan
    from ulip_client import score_ulip  # noqa: PLC0415
    return score_ulip(mesh_path, caption)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_captions() -> list[tuple[str, str]]:
    """Returns list of (filename, caption) in CSV order."""
    if not CAPTIONS_CSV.exists():
        sys.exit(f"missing {CAPTIONS_CSV}")
    rows = []
    with open(CAPTIONS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["filename"].strip(), row["caption"].strip()))
    return rows


def review_subset() -> set[str]:
    """Filenames of the objects the review conditions run on."""
    with open(CAPTIONS_CSV, newline="") as fh:
        return {r["filename"].strip() for r in csv.DictReader(fh)
                if r.get("source", "").strip() == REVIEW_SUBSET}


def ensure_retrieved_photos(captions: list[tuple[str, str]]) -> None:
    """For each caption missing a photo, retrieve ONE real web image and cache it.

    Uses the project's own Tavily image search + downloader. The cached photo
    in eval/dataset/images/<filename> is then reused across ALL 8 conditions
    for that prompt (so the only variable is the restyle axes, not the input
    image) and across reruns. Tries multiple URLs until one downloads + verifies.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for filename, caption in captions:
        out = IMAGES_DIR / filename
        if out.exists():
            continue
        print(f"[retrieve] searching for a real photo of {caption!r}")
        try:
            urls = tools.image_search(f"{caption}, real photograph", max_results=8)
        except Exception as e:  # noqa: BLE001
            print(f"[retrieve] search failed for {filename}: {e}")
            continue
        for url in urls:
            try:
                tools.download_image(url, str(out))
                print(f"[retrieve] {filename} <- {url[:70]}")
                break
            except Exception:  # noqa: BLE001 — try the next URL
                continue
        else:
            print(f"[retrieve] !! no downloadable image for {filename} "
                  f"({caption}) — leave a file there manually or rerun")


def load_done(csv_path: Path) -> set[tuple[str, str]]:
    """(prompt, condition) pairs that SUCCEEDED — skipped on resume.

    Only `ok` rows are treated as done. `error` rows are retried on the next
    run so transient/backend failures don't permanently drop a cell.
    """
    done: set[tuple[str, str]] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "status" not in reader.fieldnames:
            sys.exit(f"{csv_path} has no/!bad header (got {reader.fieldnames}). "
                     "Resume can't work — fix or delete the file.")
        for row in reader:
            if row.get("status") == "ok":
                done.add((row["prompt"], row["condition"]))
    return done


# ---------------------------------------------------------------------------
# Agent driver
# ---------------------------------------------------------------------------

def _newest_after(pattern: str, since: float) -> str | None:
    """Newest file matching agent.OUT/pattern with mtime >= since, else None.

    The agent strips artifact paths out of its event/status stream (the
    '[trellis2] done -> /x.glb' stdout line becomes the friendly status
    'TRELLIS finished.'), so we detect outputs by scanning the output dir
    for files the run just created instead of parsing events.
    """
    candidates = [
        (p.stat().st_mtime, str(p))
        for p in agent.OUT.glob(pattern)
        if p.stat().st_mtime >= since
    ]
    if not candidates:
        return None
    return max(candidates)[1]


def _run_agent(user_message: str, tag: str = "") -> dict:
    """Drive the agent and return {'mesh': path, 'restyled': path}.

    Auto-approves HITL gates so the eval runs unattended. Detects the produced
    mesh (.glb) and restyled input (objaverse_*.png) by newest-file-in-output-
    dir rather than event parsing, since the agent does not surface paths.
    """
    session_id = str(uuid.uuid4())
    # mtime fence: only count files created from here on. Subtract 1s of slack
    # for filesystem timestamp granularity.
    since = time.time() - 1.0

    interrupts: list[dict] = []

    def _consume(stream) -> None:
        for event in stream:
            text = str(event.get("content", ""))
            if event.get("type") == "text" and text.startswith("Agent error"):
                _AGENT_ERRORS.append(text)
            if event.get("type") == "interrupt":
                interrupts.append({k: event.get(k) for k in
                                   ("tool", "image_path", "allowed_decisions")})
                requests = event.get("action_requests", [])
                decisions = [{"type": "approve"} for _ in requests] or [{"type": "approve"}]
                _consume(resume_chat_stream(session_id, decisions))
                return

    _consume(process_chat_stream(user_message, [], session_id=session_id))
    # Match on this cell's tag (not just "newest file") so two sweeps running
    # in parallel on one box (e.g. generated + retrieved arms on separate
    # GPUs) can never pick up each other's meshes.
    t = f"_{tag}_" if tag else "_"
    # Prefer what this cell's (wrapped) tools actually returned: a reused mesh
    # or restyled image predates `since`, so mtime alone would miss it.
    restyles = [e["path"] for e in _TOOL_LOG if e.get("tool") == "restyle" and e.get("path")]
    return {
        "mesh": _LAST_MESH or _newest_after(f"*{t}*.glb", since),
        "restyled": (restyles[-1] if restyles else None) or _newest_after(f"objaverse{t}*.png", since),
        "trace": _extract_trace(session_id, interrupts),
    }


def _extract_trace(session_id: str, interrupts: list[dict]) -> dict:
    """Pull the full tool-call trace + resolved model versions out of the
    LangGraph checkpoint for this thread (the stream hides tool args)."""
    calls, models, results = [], [], []
    try:
        state = agent.agent.get_state(agent._thread_config(session_id))
        for m in state.values.get("messages", []):
            meta = getattr(m, "response_metadata", None) or {}
            mv = meta.get("model_version") or meta.get("model_name")
            if mv:
                models.append(mv)
            for tc in getattr(m, "tool_calls", None) or []:
                calls.append({"name": tc.get("name"), "args": tc.get("args")})
            if getattr(m, "type", "") == "tool":
                content = m.content if isinstance(m.content, str) else str(m.content)
                results.append({"name": getattr(m, "name", ""), "content": content[:400]})
    except Exception as e:  # noqa: BLE001
        calls.append({"name": "__trace_error__", "args": repr(e)})
    return {"session_id": session_id, "tool_calls": calls,
            "models_resolved": sorted(set(models)), "interrupts": interrupts,
            "tool_log": list(_TOOL_LOG), "tool_results": results}


def _backbone_info(trace: dict) -> dict:
    """Routing decision = the LAST 3D-generation tool call the agent made."""
    import inspect
    gen = [c for c in trace.get("tool_calls", []) if c["name"] in GEN_TOOLS]
    if not gen:
        return {}
    c = gen[-1]
    args = c.get("args") or {}
    name = c["name"]
    if name == "tool_trellis2":
        ep = (os.environ.get("LOCAL_TRELLIS_URL", "") or tools.TRELLIS_GEN_URL) + "/generate"
        fn = tools.trellis2
    elif name == "tool_trellis2_texture":
        ep, fn = tools.TRELLIS_TEX_URL + "/texture", tools.trellis2_texture
    elif name == "tool_hunyuan3d2":
        ep, fn = tools.HUNYUAN3D2_GEN_URL + "/generate", tools.hunyuan3d2
    else:
        ep = (tools.PARTCRAFTER_SCENE_URL + "/generate-scene" if args.get("scene")
              else tools.PARTCRAFTER_OBJ_URL + "/generate")
        fn = tools.partcrafter
    seed = inspect.signature(fn).parameters.get("seed")
    if name.startswith("tool_trellis2") and os.environ.get("TRELLIS_REMESH", "").lower() == "false":
        args = {**args, "remesh": False, "remesh_fallback": True}   # logged fallback
    if name.startswith("tool_trellis2"):
        for env, key in (("TRELLIS_DECIMATION", "decimation_target"), ("TRELLIS_TEXTURE_SIZE", "texture_size")):
            if os.environ.get(env):
                args = {**args, key: int(os.environ[env]), "export_fallback": True}
    # strip the workspace host so the CSV doesn't leak the account name
    ep_short = ep.split(".modal.run")[-1] if ".modal.run" in ep else ep
    return {"backbone_tool": name.replace("tool_", ""),
            "backbone_endpoint": f"{name.replace('tool_', '')}:{ep_short}",
            "backbone_args": json.dumps(args),
            "backbone_seed": "" if seed is None else str(seed.default),
            "input_image_path": args.get("image_path", "")}


def _existing_artifacts(tag: str) -> dict[str, str | None]:
    """Find an already-generated mesh (+ restyled input) for this tag on disk.

    A hard crash (e.g. during render) writes no CSV row, so the cell isn't
    marked done — but the expensive .glb is already saved. Reusing it skips
    regenerating the concept, restyle, and mesh on the next run. Returns the
    NEWEST valid (>=MIN bytes) mesh matching the tag, or {mesh: None}.
    """
    MIN = 1024  # a real .glb is far bigger; filters 0-byte failed downloads
    meshes = [
        (p.stat().st_mtime, str(p))
        for p in agent.OUT.glob(f"*{tag}_*.glb")
        if p.stat().st_size >= MIN
    ]
    if not meshes:
        return {"mesh": None, "restyled": None}
    mesh = max(meshes)[1]
    restyled = [str(p) for p in agent.OUT.glob(f"objaverse_{tag}_*.png")]
    return {"mesh": mesh, "restyled": restyled[-1] if restyled else None}


def _existing_input_image(tag: str, cond_name: str,
                          photo_path: str | None) -> tuple[str | None, str | None]:
    """Find the image that would be fed to the 3D backbone for this cell, if a
    prior run already produced it — so we can skip the (expensive) Gemini concept
    + restyle calls and just re-run the backbone.

    Returns (image_path, kind) or (None, None). For restyled conditions the input
    is the restyled image; for `raw` it's the concept image (generated mode) or
    the photo (retrieved/photo mode).
    """
    out = agent.OUT
    restyled = sorted(out.glob(f"objaverse_{tag}_*.png"), key=lambda p: p.stat().st_mtime)
    if restyled:
        return str(restyled[-1]), "restyled"
    if cond_name == "raw":
        if photo_path:
            return photo_path, "photo"
        concept = sorted(out.glob(f"concept_{tag}_*.png"), key=lambda p: p.stat().st_mtime)
        if concept:
            return str(concept[-1]), "concept"
    return None, None


_SKIP_VLM = False


def _fill_provenance(row: dict, trace: dict, cond_name: str,
                     style_prompt: str | None, mesh: str | None) -> None:
    calls = trace.get("tool_calls", [])
    log = trace.get("tool_log", [])
    restyles = [e for e in log if e.get("tool") == "restyle"]
    concepts = [e for e in log if e.get("tool") == "concept"]
    row["agent_model_resolved"] = ";".join(trace.get("models_resolved", []))
    row["tool_sequence"] = ">".join(c["name"].replace("tool_", "") for c in calls)
    row["n_tool_calls"] = str(len(calls))
    row["n_restyle_calls"] = str(len(restyles))
    row["n_hitl_interrupts"] = str(len(trace.get("interrupts", [])))
    row["concept_path"] = concepts[-1]["path"] if concepts else ""
    row.update(_backbone_info(trace))
    gens = [e for e in log if e.get("tool") == "generate" and e.get("mode") == "run"]
    row["n_gen_calls"] = str(len(gens))
    row["n_edit_calls"] = str(sum(1 for e in log if e.get("tool") == "edit_image"))
    row["n_concept_regens"] = str(sum(1 for e in log if e.get("mode") == "review_regenerated"))
    scores, verdicts = [], []
    for r in trace.get("tool_results", []):
        if r.get("name") == "tool_score_alignment":
            m1 = re.search(r"Alignment score:\s*([\d.]+)", r["content"])
            m2 = re.search(r"next_action:\s*(\w+)", r["content"])
            scores.append(m1.group(1) if m1 else "?")
            verdicts.append(m2.group(1) if m2 else "?")
    row["review_scores"] = ";".join(scores)
    row["review_verdicts"] = ";".join(verdicts)
    row["review"] = str(cond_name.endswith("_review"))
    cond_name = base_condition(cond_name)
    # Was the condition actually applied to the image the backbone received?
    if cond_name == "raw":
        ok = all(e["mode"] == "identity(raw)" for e in restyles)
    else:
        ok = bool(restyles) and all(e.get("prompt") == style_prompt for e in restyles)
    row["condition_enforced"] = str(ok)
    if mesh:
        row["trace_path"] = str(_trace_file(_EVAL_TAG))


def _wait_for_trellis(limit_s: float = 900) -> None:
    """Block until the local TRELLIS.2 server answers. After a watchdog exit
    the server needs 1-2 min to reload weights; without this wait every cell
    started in that window fails instantly with "Connection refused"."""
    import urllib.request
    url = os.environ.get("LOCAL_TRELLIS_URL")
    if not url:
        return
    t0, warned = time.time(), False
    while True:
        try:
            urllib.request.urlopen(url.rstrip("/") + "/", timeout=5).read()
            if warned:
                print(f"[eval] TRELLIS server back after {time.time() - t0:.0f} s", flush=True)
            return
        except Exception:
            if time.time() - t0 > limit_s:
                raise SystemExit(f"[eval] TRELLIS server {url} down for {limit_s:.0f} s — "
                                 "stopping the sweep (check logs/trellis_*.log)")
            if not warned:
                print(f"[eval] waiting for TRELLIS server {url} ...", flush=True)
                warned = True
            time.sleep(10)


def run_one(caption: str, cond_name: str, instruction: str | None,
            tag: str = "", photo_path: str | None = None,
            style_prompt: str | None = None, concept_cache: Path | None = None,
            prov: dict | None = None) -> dict:
    global _EVAL_TAG, _EVAL_STYLE_PROMPT, _EVAL_CAPTION, _CONCEPT_CACHE_PATH
    global _REVIEW, _GEN_COUNT, _LAST_MESH
    _wait_for_trellis()
    _REVIEW = cond_name.endswith("_review")
    _GEN_COUNT, _LAST_MESH = 0, None
    _EVAL_TAG = tag  # the patched agent._stamp() reads this to name media
    _EVAL_STYLE_PROMPT = style_prompt
    _EVAL_CAPTION = caption
    _CONCEPT_CACHE_PATH = concept_cache
    row = {f: "" for f in CSV_FIELDS}
    row.update(prompt=caption, condition=cond_name, **(prov or {}))
    row["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    try:
        # Reuse an existing mesh for this cell if a prior (crashed) run already
        # generated one — skips the expensive concept/restyle/backbone steps.
        # Reuse ONLY a mesh from this same run (tag carries the run id) whose
        # trace sidecar exists — i.e. a cell whose generation finished but
        # whose scoring crashed. Never force a backbone: that would override
        # the agent's routing decision (the old trellis2-only fallback did).
        rec_f = _trace_file(tag) if tag else None
        rec = json.loads(rec_f.read_text()) if rec_f and rec_f.exists() else {}
        if rec.get("mesh") and Path(rec["mesh"]).exists():
            print(f"    reusing this cell's mesh {Path(rec['mesh']).name}")
            produced = {"mesh": rec["mesh"], "restyled": rec.get("restyled"),
                        "trace": rec.get("trace", {})}
            row["reused_mesh"] = "True"
        else:
            # Build the user message. The eval instruction sets the restyle axes
            # for this condition; the STOP directive keeps the pipeline fixed
            # (no re-texture / no agent-side render). In photo mode we hand the
            # agent a REAL photo via the [Uploaded Image Local Path: ...] tag it
            # understands, so it restyles+generates from the photo instead of
            # generating a concept image first.
            stop = ("[EVAL] After the 3D model (.glb) is generated, STOP. Do "
                    "NOT re-texture, render views, or inspect the result — "
                    "just report the saved .glb path.")
            upload = (f"\n[Uploaded Image Local Path: {photo_path}]"
                      if photo_path else "")
            # Generated mode = strategy 3 ("generate without extra information"):
            # pin the concept image to the bare caption so the agent does NOT
            # embellish it with style/lighting/background terms (that embellishment
            # is what made the first generated pilot already-clean and inert).
            verbatim = ("" if photo_path else
                        "\n[EVAL] Generate the concept image from this EXACT "
                        "prompt verbatim — add NO extra detail, style, lighting, "
                        f"background, or framing terms: {caption}")
            # Review conditions replace the STOP directive with the agent's
            # review loop; the cap/seed logic is enforced in _wrap_generator.
            review = (
                "[EVAL] REVIEW MODE: before sending an image to a 3D tool, check it "
                "with tool_inspect_image. After the 3D model (.glb) is generated, "
                f"call tool_score_alignment on it with the prompt \"{caption}\". "
                "If it recommends 'regenerate', fix the likely cause (for example a "
                "better or edited input image, or a different 3D tool) and generate "
                "again, then score again — at most 2 regenerations in total. Do NOT "
                "re-texture. Finish by reporting the final .glb path.")
            lead = review if _REVIEW else stop
            # NOTE: `instruction` is deliberately NOT put in the message — the
            # condition is enforced at the tool layer, so the agent sees the
            # same text in every condition of the same review setting.
            message = f"{lead}{verbatim}{upload}\n\nGenerate a 3D model of: {caption}"
            _TOOL_LOG.clear()
            _AGENT_ERRORS.clear()
            produced = _run_agent(message, tag)
            if any("credits are depleted" in e or "RESOURCE_EXHAUSTED" in e for e in _AGENT_ERRORS):
                raise GeminiOutOfCredit(
                    "Gemini prepaid credit is depleted — stopping the sweep. Top up at "
                    "ai.studio/projects, then rerun the same command (only unfinished cells run).")
            if produced.get("mesh"):
                rec_f = _trace_file(tag)
                rec_f.parent.mkdir(parents=True, exist_ok=True)
                rec_f.write_text(json.dumps(
                    {"mesh": produced["mesh"], "restyled": produced.get("restyled"),
                     "trace": produced["trace"]}, indent=1, default=str))

        trace = produced.get("trace") or {}
        _fill_provenance(row, trace, cond_name, style_prompt, mesh=produced.get("mesh"))
        mesh = produced["mesh"]
        if not mesh:
            # surface the backend's own error (the agent only saw it as a tool message)
            errs = [r["content"] for r in trace.get("tool_results", [])
                    if r.get("name") in GEN_TOOLS and "error" in r.get("content", "").lower()]
            agent_err = f" — agent: {_AGENT_ERRORS[-1][:300]}" if _AGENT_ERRORS else ""
            raise RuntimeError("agent did not produce a .glb mesh"
                               + (f" — 3D tool error: {errs[-1][:300]}" if errs else "")
                               + agent_err)
        row["mesh_path"] = mesh
        row["restyled_path"] = produced.get("restyled") or ""

        # render views (tools.render_mesh_views is patched to use subprocess)
        views_dir = Path(mesh).with_suffix(".views")
        render_result = tools.render_mesh_views(mesh, views_dir)
        view_paths = list(render_result.values())

        # CLIP
        clip_rep = tools.check_alignment(view_paths, caption)
        row.update(
            clip_mean=f"{clip_rep.score:.4f}",
            clip_accept=str(clip_rep.accept),
            worst_view=clip_rep.worst_view or "",
            per_view=json.dumps(clip_rep.per_view),
        )

        # VLM judge
        if _SKIP_VLM:
            vlm, raw = math.nan, {}
        else:
            raw = {}
            vlm = _score_vlm(view_paths, caption, raw_out=raw)
        row["vlm_score"] = "" if math.isnan(vlm) else f"{vlm:.1f}"
        row["vlm_score_raw"] = raw.get("text", "")

        # ULIP-2 (point-cloud↔text, scored from the mesh on Modal GPU)
        ulip = _score_ulip(mesh, caption)
        row["ulip_mean"] = "" if math.isnan(ulip) else f"{ulip:.4f}"

        row["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
    row["seconds"] = f"{time.time() - t0:.1f}"
    return row


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None,
                    help="run identifier; outputs go to eval/runs/<run-id>/. "
                         "Reuse the same id to resume. Default: timestamp.")
    ap.add_argument("--out", default=None, type=Path,
                    help="override output CSV (default: "
                         "eval/runs/<run-id>/results_<input>.csv)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max prompts to process (0 = all)")
    ap.add_argument("--no-vlm-score", action="store_true",
                    help="skip the VLM judge scoring call")
    ap.add_argument("--input", choices=["generated", "photo", "retrieved"],
                    default="generated",
                    help="generated: agent makes a concept image from the prompt "
                         "(default). photo: feed a hand-supplied real photo from "
                         "eval/dataset/images/<filename>. retrieved: auto-fetch a "
                         "real web photo per prompt (Tavily) and cache it there.")
    ap.add_argument("--conditions", nargs="+",
                    default=["raw", "all_on", "loo_lighting", "raw_review", "all_on_review"],
                    metavar="COND",
                    help="conditions to run. Default: raw, all_on, loo_lighting "
                         "(full restyle minus the lighting clause) on every object, "
                         "plus raw_review and all_on_review (agent review loop on) "
                         "on the 20 submission objects only.")
    ap.add_argument("--auto-retrieve", action="store_true",
                    help="retrieved mode: auto-fetch a web photo (Tavily) for "
                         "prompts with no curated photo. Off by default.")
    args = ap.parse_args()

    global _SKIP_VLM
    _SKIP_VLM = args.no_vlm_score
    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        args.out = run_dir / f"results_{args.input}.csv"
    _write_run_meta(run_dir, run_id, args)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    captions = load_captions()
    if args.limit:
        captions = captions[:args.limit]

    # retrieved: auto-fetch one real web photo per prompt, cached to images/.
    # retrieved: photos are hand-curated (eval/dataset/curate_photos.py) with a
    # licence manifest. Auto-fetching a web image for a missing one is what
    # once saved an Instagram HTML page as kite_01.png — only on request now.
    if args.input == "retrieved" and args.auto_retrieve:
        ensure_retrieved_photos(captions)

    # photo/retrieved both feed a real photo from images/ to the agent.
    photo_mode = args.input in ("photo", "retrieved")
    if photo_mode:
        missing = [fn for fn, _ in captions
                   if not (IMAGES_DIR / fn).exists()]
        if missing:
            sys.exit(
                f"--input {args.input}: missing {len(missing)} photo(s) in "
                f"{IMAGES_DIR}/:\n  " + "\n  ".join(missing) +
                "\n(retrieval failed for these — rerun, or drop a file manually.)")

    conds = conditions()
    if args.conditions:
        valid = {c for c, _ in conds}
        unknown = [c for c in args.conditions if c not in valid]
        if unknown:
            sys.exit(f"unknown condition(s): {unknown}; valid: {sorted(valid)}")
        conds = [(c, i) for c, i in conds if c in args.conditions]
    done  = load_done(args.out)
    subset = review_subset()

    def runs(filename: str, cond: str) -> bool:
        return not cond.endswith("_review") or filename in subset

    total = sum(runs(f, c) for f, _ in captions for c, _ in conds)
    print(f"{len(captions)} prompts x {len(conds)} conditions = {total} runs "
          f"({len(done)} already done)")

    # Write a header when the file is missing OR empty. A headerless CSV breaks
    # load_done (DictReader reads row 1 as field names), which silently disables
    # resume and reprocesses every cell — so guard against it explicitly.
    need_header = (not args.out.exists()) or args.out.stat().st_size == 0
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if need_header:
            writer.writeheader()
            fh.flush()
        i = 0
        for filename, caption in captions:
            stem = Path(filename).stem  # e.g. bicycle_01
            for cond_name, instruction in conds:
                if not runs(filename, cond_name):
                    continue
                i += 1
                key = (caption, cond_name)
                if key in done:
                    print(f"[{i}/{total}] skip (done) {caption!r} | {cond_name}")
                    continue
                print(f"[{i}/{total}] {caption!r} | {cond_name}")
                # Run- and mode-prefixed tag so media/meshes never collide
                # with or get reused from another run or input mode.
                prefix = {"generated": "gen_", "photo": "photo_",
                          "retrieved": "ret_"}[args.input]
                tag = f"{run_id}_{prefix}{stem}__{cond_name}"
                photo_path = str(IMAGES_DIR / filename) if photo_mode else None
                style = None if base_condition(cond_name) == "raw" else STYLE_PROMPTS[cond_name]
                prov = {"run_id": run_id, "input_mode": args.input, "stem": stem,
                        "agent_model": agent.AGENT_LLM_MODEL,
                        "judge_model": JUDGE_MODEL,
                        "image_model": tools.GEMINI_IMAGE_MODEL,
                        "gemini_seed": os.environ.get("GEMINI_SEED", "")}
                row = run_one(caption, cond_name, instruction,
                              tag=tag, photo_path=photo_path,
                              style_prompt=style,
                              concept_cache=(run_dir / "concepts" / f"{stem}.png"
                                             if not photo_mode else None),
                              prov=prov)
                writer.writerow(row)
                fh.flush()
                print(f"    -> {row['status']}  "
                      f"clip={row['clip_mean']}  "
                      f"vlm={row['vlm_score']}  "
                      f"backbone={row['backbone_tool']}  "
                      f"enforced={row['condition_enforced']}  "
                      f"({row['seconds']}s)")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
