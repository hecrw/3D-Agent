#!/usr/bin/env python
"""Restyle-preprocessing ablation sweep — agent-level evaluation.

Drives the 3D agent (agent.process_chat_stream) with each prompt under each
restyle condition and scores the resulting mesh.

For each (prompt, condition) it:
  1. builds a system instruction that forces the restyle axes for the condition,
  2. sends the prompt to the agent and waits for a mesh to be produced,
  3. renders multi-view PNGs from the mesh,
  4. scores with CLIP, Gen3DEval (VLM-as-judge via Claude Haiku), and ULIP-2,
and appends one row to a CSV. Already-present rows are skipped so the sweep is
resumable after a crash.

Run from the repo root:
    .venv/bin/python eval/run_pilot.py --limit 10 --out eval/results_pilot.csv

Flags:
    --limit N       process only the first N prompts (0 = all)
    --out PATH      output CSV path (default: eval/results_pilot.csv)
    --no-gen3deval  skip the Claude Haiku scoring call (saves API cost)
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
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

DATASET_DIR  = REPO_ROOT / "eval" / "dataset"
IMAGES_DIR   = DATASET_DIR / "images"   # real photos for --input photo mode
CAPTIONS_CSV = DATASET_DIR / "captions.csv"
WORK_DIR     = REPO_ROOT / "eval" / "work"

CSV_FIELDS = [
    "prompt", "condition",
    "clip_mean", "clip_accept", "worst_view", "per_view",
    "gen3deval", "ulip_mean",
    "restyled_path", "mesh_path", "status", "error", "seconds",
]

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
    return conds


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_gen3deval(view_paths: list[str], caption: str) -> float:
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
        resp = agent.llm.invoke([HumanMessage(content=content)])
        raw = resp.content
        text = (raw if isinstance(raw, str)
                else "".join(b.get("text", "") for b in raw
                             if isinstance(b, dict)))
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
    from ulip_client import score_ulip  # noqa: PLC0415
    return score_ulip(mesh_path, caption)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_captions() -> list[tuple[str, str]]:
    """Returns list of (filename_stem, caption) in CSV order."""
    if not CAPTIONS_CSV.exists():
        sys.exit(f"missing {CAPTIONS_CSV}")
    rows = []
    with open(CAPTIONS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["filename"].strip(), row["caption"].strip()))
    return rows


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


def _run_agent(user_message: str) -> dict[str, str | None]:
    """Drive the agent and return {'mesh': path, 'restyled': path}.

    Auto-approves HITL gates so the eval runs unattended. Detects the produced
    mesh (.glb) and restyled input (objaverse_*.png) by newest-file-in-output-
    dir rather than event parsing, since the agent does not surface paths.
    """
    session_id = str(uuid.uuid4())
    # mtime fence: only count files created from here on. Subtract 1s of slack
    # for filesystem timestamp granularity.
    since = time.time() - 1.0

    def _consume(stream) -> None:
        for event in stream:
            if event.get("type") == "interrupt":
                requests = event.get("action_requests", [])
                decisions = [{"type": "approve"} for _ in requests] or [{"type": "approve"}]
                _consume(resume_chat_stream(session_id, decisions))
                return

    _consume(process_chat_stream(user_message, [], session_id=session_id))
    return {
        "mesh": _newest_after("*.glb", since),
        "restyled": _newest_after("objaverse_*.png", since),
    }


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


def run_one(caption: str, cond_name: str, instruction: str | None,
            tag: str = "", photo_path: str | None = None) -> dict:
    global _EVAL_TAG
    _EVAL_TAG = tag  # the patched agent._stamp() reads this to name media
    row = {f: "" for f in CSV_FIELDS}
    row.update(prompt=caption, condition=cond_name)
    t0 = time.time()
    try:
        # Reuse an existing mesh for this cell if a prior (crashed) run already
        # generated one — skips the expensive concept/restyle/backbone steps.
        produced = _existing_artifacts(tag) if tag else {"mesh": None}
        if produced["mesh"]:
            print(f"    reusing existing mesh {Path(produced['mesh']).name}")
        elif tag and _existing_input_image(tag, cond_name, photo_path)[0]:
            # No mesh, but the concept/restyled/retrieved image already exists:
            # skip the Gemini concept + restyle calls and just run the backbone.
            inp, kind = _existing_input_image(tag, cond_name, photo_path)
            print(f"    reusing existing {kind} image {Path(inp).name} "
                  f"-> backbone only")
            out_glb = str(agent.OUT / f"trellis2_{tag}_{int(time.time()*1000)}.glb")
            mesh = tools.trellis2(inp, out_path=out_glb)
            produced = {"mesh": mesh,
                        "restyled": inp if kind == "restyled" else ""}
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
            base = f"{instruction}\n{stop}" if instruction else stop
            message = f"{base}{verbatim}{upload}\n\nGenerate a 3D model of: {caption}"
            produced = _run_agent(message)

        mesh = produced["mesh"]
        if not mesh:
            raise RuntimeError("agent did not produce a .glb mesh")
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

        # Gen3DEval
        gen3d = _score_gen3deval(view_paths, caption)
        row["gen3deval"] = "" if math.isnan(gen3d) else f"{gen3d:.1f}"

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
    ap.add_argument("--out", default="eval/results_pilot.csv", type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="max prompts to process (0 = all)")
    ap.add_argument("--no-gen3deval", action="store_true",
                    help="skip the Gen3DEval scoring call")
    ap.add_argument("--input", choices=["generated", "photo", "retrieved"],
                    default="generated",
                    help="generated: agent makes a concept image from the prompt "
                         "(default). photo: feed a hand-supplied real photo from "
                         "eval/dataset/images/<filename>. retrieved: auto-fetch a "
                         "real web photo per prompt (Tavily) and cache it there.")
    ap.add_argument("--conditions", nargs="+", default=None,
                    metavar="COND",
                    help="subset of conditions to run, e.g. --conditions raw "
                         "all_on (just the headline comparison = 2 gens/prompt). "
                         "Default: all 8 (raw, all_on, 6x loo_*).")
    args = ap.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    captions = load_captions()
    if args.limit:
        captions = captions[:args.limit]

    # retrieved: auto-fetch one real web photo per prompt, cached to images/.
    if args.input == "retrieved":
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
    total = len(captions) * len(conds)
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
                i += 1
                key = (caption, cond_name)
                if key in done:
                    print(f"[{i}/{total}] skip (done) {caption!r} | {cond_name}")
                    continue
                print(f"[{i}/{total}] {caption!r} | {cond_name}")
                if args.no_gen3deval:
                    # monkey-patch scorer to skip API calls
                    import eval.run_pilot as _self  # noqa: PLC0415
                    _self._score_gen3deval = lambda *_: math.nan  # type: ignore[assignment]
                # Mode-prefixed tag so each input mode's media/meshes never
                # collide with or get reused from another mode's run.
                prefix = {"generated": "", "photo": "photo_",
                          "retrieved": "ret_"}[args.input]
                tag = f"{prefix}{stem}__{cond_name}"  # e.g. ret_bicycle_01__loo_view
                photo_path = str(IMAGES_DIR / filename) if photo_mode else None
                row = run_one(caption, cond_name, instruction,
                              tag=tag, photo_path=photo_path)
                writer.writerow(row)
                fh.flush()
                print(f"    -> {row['status']}  "
                      f"clip={row['clip_mean']}  "
                      f"gen3d={row['gen3deval']}  "
                      f"({row['seconds']}s)")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
