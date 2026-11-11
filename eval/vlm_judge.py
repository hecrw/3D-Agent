#!/usr/bin/env python
"""Pairwise VLM judge following the Gen3DEval evaluation protocol.

    .venv/bin/python eval/vlm_judge.py eval/runs/<run-id>

Gen3DEval (Maiti, Agapito, Kokkinos — CVPR 2025) has NOT released its code or
its fine-tuned model, so this is NOT Gen3DEval. It follows the published
protocol with Gemini as the judge, and should be reported as
"VLM judge (Gemini) following the Gen3DEval protocol":

  * pairwise: the judge sees two objects and picks the better one;
  * 4 views per object, evenly spaced around a 360-degree turntable at a fixed
    elevation, RGB and normal maps;
  * three axes, each asked separately:
      appearance     - RGB views,    no text prompt
      surface        - normal maps,  no text prompt
      text_fidelity  - RGB views,    WITH the text prompt (the only axis that
                       gets the prompt, as in Gen3DEval);
  * pairwise outcomes are aggregated with Elo.

Extra controls (not in Gen3DEval): every pair is judged in BOTH orders; a pair
counts as a win only if both orders agree, otherwise a tie (removes position
bias). Pinned judge model, temperature 0, fixed seed.

Pairs judged per (input mode, prompt) — chosen to answer the paper's questions
without judging every pair:
  all_on vs raw (+ leave-one-out conditions, if a run has them) -> restyle effect
  all_on vs each direct baseline                                 -> vs baselines
  raw    vs each direct baseline                                 -> agent w/o restyle
Outputs in the run dir:
  judge_pairs.csv     one row per (pair, order, axis) call — resumable cache
  judge_summary.csv   per mode/axis/comparison: wins, ties, losses, win rate,
                      two-sided sign-test p; plus Elo per system
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from render3d import render_turntable  # noqa: E402

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")
SEED = int(os.environ.get("GEMINI_SEED", "20261111") or 0)
N_VIEWS, ELEVATION = 4, 20.0
AXES = ("appearance", "surface", "text_fidelity")
BASELINES = ("trellis2", "hunyuan3d2", "partcrafter")
LOO = ("background", "framing", "view", "lighting", "isolation", "part_visibility")

QUESTIONS = {
    "appearance": (
        "You are evaluating two 3D objects produced by text-to-3D generators. "
        "Object 1 is shown in the first {n} images and Object 2 in the next {n} "
        "images; each set shows the object from {n} viewpoints around it.\n"
        "Judge APPEARANCE only: the quality, realism, sharpness and consistency "
        "of the colours and textures across views. Ignore what the object is "
        "supposed to be.\n"
        "Which object has better appearance? Reply with exactly one character: 1 or 2."),
    "surface": (
        "You are evaluating two 3D objects produced by text-to-3D generators. "
        "Object 1 is shown in the first {n} images and Object 2 in the next {n} "
        "images. These are SURFACE NORMAL MAPS (colour encodes surface "
        "orientation), from {n} viewpoints around each object.\n"
        "Judge SURFACE QUALITY only: clean, plausible geometry versus noise, "
        "bumps, holes, floaters, broken or melted parts.\n"
        "Which object has better surface quality? Reply with exactly one character: 1 or 2."),
    "text_fidelity": (
        "You are evaluating two 3D objects generated from the text prompt: "
        "\"{prompt}\".\n"
        "Object 1 is shown in the first {n} images and Object 2 in the next {n} "
        "images, each from {n} viewpoints around the object.\n"
        "Judge TEXT FIDELITY only: how completely and accurately the object "
        "matches every part of the prompt (object type, attributes, colours, "
        "materials, details).\n"
        "Which object matches the prompt better? Reply with exactly one character: 1 or 2."),
}
AXIS_VIEWS = {"appearance": "rgb", "surface": "normal", "text_fidelity": "rgb"}

PAIR_FIELDS = ["mode", "prompt", "sys_a", "sys_b", "order", "axis", "choice",
               "winner", "raw", "model", "mesh_a", "mesh_b", "ts"]


# ---------------------------------------------------------------------------
def _gemini():
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(REPO / ".env")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def load_systems(run_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    """{mode: {prompt: {system: mesh_path}}} from every ok row in the run."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for mode in ("generated", "retrieved"):
        f = run_dir / f"results_{mode}.csv"
        if f.exists():
            for r in csv.DictReader(open(f)):
                if r["status"] == "ok" and r["mesh_path"]:
                    out[mode][r["prompt"]][r["condition"]] = r["mesh_path"]
        for bb in BASELINES:
            f = run_dir / f"results_baseline_{bb}_{mode}.csv"
            if f.exists():
                for r in csv.DictReader(open(f)):
                    if r["status"] == "ok" and r["mesh_path"]:
                        out[mode][r["prompt"]][f"baseline_{bb}"] = r["mesh_path"]
    return out


def planned_pairs(systems: dict[str, str]) -> list[tuple[str, str]]:
    # leave-one-out conditions were dropped from the camera-ready; their pairs
    # are only judged if such rows exist in the run
    pairs = [("all_on", c) for c in ["raw"] + [f"loo_{a}" for a in LOO]]
    pairs += [("all_on", f"baseline_{b}") for b in BASELINES]
    pairs += [("raw", f"baseline_{b}") for b in BASELINES]
    return [(a, b) for a, b in pairs if a in systems and b in systems]


def views_for(mesh: str) -> dict[str, list[str]]:
    return render_turntable(mesh, Path(mesh).with_suffix(".judge_views"),
                            n_views=N_VIEWS, elevation_deg=ELEVATION)


def ask(client, axis: str, prompt: str, imgs_1: list[str], imgs_2: list[str]) -> tuple[str, str]:
    from google.genai import types
    parts = [types.Part.from_text(text=QUESTIONS[axis].format(n=N_VIEWS, prompt=prompt))]
    for p in imgs_1 + imgs_2:
        parts.append(types.Part.from_bytes(data=Path(p).read_bytes(), mime_type="image/png"))
    # No tiny max_output_tokens: thinking tokens count against it, so a cap of
    # a few tokens can return an EMPTY answer. Low thinking keeps cost down.
    cfg = types.GenerateContentConfig(
        temperature=0.0, seed=SEED,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW))
    for attempt in range(5):
        try:
            resp = client.models.generate_content(model=JUDGE_MODEL, contents=parts, config=cfg)
            text = (resp.text or "").strip()
            m = re.search(r"[12]", text)
            return (m.group() if m else ""), text[:50]
        except Exception as e:  # noqa: BLE001 — rate limits / transient errors
            msg = str(e)
            if any(s in msg for s in ("402", "credits are depleted", "API_KEY_INVALID", "PERMISSION_DENIED")):
                raise SystemExit(f"Gemini account problem — stopping (not retryable):\n{msg[:400]}")
            if attempt == 4:
                return "", f"error: {type(e).__name__}: {e}"[:120]
            time.sleep(2 ** attempt * 3)
    return "", "unreachable"


# ---------------------------------------------------------------------------
def judge(run_dir: Path, workers: int) -> Path:
    systems = load_systems(run_dir)
    out = run_dir / "judge_pairs.csv"
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out)):
            if r["choice"] in ("1", "2"):
                done.add((r["mode"], r["prompt"], r["sys_a"], r["sys_b"], r["order"], r["axis"]))

    jobs = []
    for mode, by_prompt in systems.items():
        for prompt, sysmap in by_prompt.items():
            for a, b in planned_pairs(sysmap):
                for order in ("ab", "ba"):
                    for axis in AXES:
                        if (mode, prompt, a, b, order, axis) not in done:
                            jobs.append((mode, prompt, a, b, order, axis, sysmap[a], sysmap[b]))
    print(f"{len(jobs)} judge calls to make ({len(done)} cached)")
    if not jobs:
        return out

    # Render every needed mesh once, up front (GL is not thread-safe).
    meshes = sorted({j[6] for j in jobs} | {j[7] for j in jobs})
    views = {}
    for i, m in enumerate(meshes, 1):
        print(f"[render {i}/{len(meshes)}] {Path(m).name}")
        views[m] = views_for(m)

    client = _gemini()
    lock = threading.Lock()
    new_file = not out.exists() or out.stat().st_size == 0
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        if new_file:
            w.writeheader()

        def run(job):
            mode, prompt, a, b, order, axis, ma, mb = job
            first, second = (ma, mb) if order == "ab" else (mb, ma)
            kind = AXIS_VIEWS[axis]
            choice, raw = ask(client, axis, prompt, views[first][kind], views[second][kind])
            winner = ""
            if choice:
                winner = (a if choice == "1" else b) if order == "ab" else (b if choice == "1" else a)
            return dict(mode=mode, prompt=prompt, sys_a=a, sys_b=b, order=order, axis=axis,
                        choice=choice, winner=winner, raw=raw, model=JUDGE_MODEL,
                        mesh_a=ma, mesh_b=mb, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, fut in enumerate(as_completed([ex.submit(run, j) for j in jobs]), 1):
                row = fut.result()
                with lock:
                    w.writerow(row)
                    fh.flush()
                if k % 25 == 0 or k == len(jobs):
                    print(f"  judged {k}/{len(jobs)}")
    return out


# ---------------------------------------------------------------------------
def _sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    from math import comb
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _elo(matches: list[tuple[str, str, float]], k: float = 16, perms: int = 1000) -> dict[str, float]:
    """Mean Elo over random match orderings (removes order dependence)."""
    rng = random.Random(0)
    players = {p for a, b, _ in matches for p in (a, b)}
    total = defaultdict(float)
    for _ in range(perms):
        r = {p: 1000.0 for p in players}
        ms = matches[:]
        rng.shuffle(ms)
        for a, b, s in ms:  # s = score of a (1 win, 0.5 tie, 0 loss)
            ea = 1 / (1 + 10 ** ((r[b] - r[a]) / 400))
            r[a] += k * (s - ea)
            r[b] -= k * (s - ea)
        for p in players:
            total[p] += r[p] / perms
    return dict(total)


def summarize(run_dir: Path) -> Path:
    pairs = run_dir / "judge_pairs.csv"
    rows = [r for r in csv.DictReader(open(pairs)) if r["winner"]]
    both = defaultdict(dict)  # (mode, prompt, a, b, axis) -> {order: winner}
    for r in rows:
        both[(r["mode"], r["prompt"], r["sys_a"], r["sys_b"], r["axis"])][r["order"]] = r["winner"]

    agg = defaultdict(lambda: [0, 0, 0])            # (mode, axis, a, b) -> [w, t, l]
    matches = defaultdict(list)                     # (mode, axis) -> [(a, b, score_a)]
    for (mode, prompt, a, b, axis), o in both.items():
        if len(o) < 2:
            continue                                # need both orders
        if o["ab"] == o["ba"]:
            s = 1.0 if o["ab"] == a else 0.0
        else:
            s = 0.5                                 # orders disagree -> tie
        agg[(mode, axis, a, b)][0 if s == 1 else 1 if s == 0.5 else 2] += 1
        matches[(mode, axis)].append((a, b, s))

    out = run_dir / "judge_summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "axis", "system_a", "system_b", "a_wins", "ties", "a_losses",
                    "n", "a_win_rate", "sign_test_p"])
        for (mode, axis, a, b), (wn, t, l) in sorted(agg.items()):
            n = wn + t + l
            w.writerow([mode, axis, a, b, wn, t, l, n, f"{(wn + 0.5 * t) / n:.3f}",
                        f"{_sign_test_p(wn, l):.4f}"])
        w.writerow([])
        w.writerow(["mode", "axis", "system", "elo"])
        for (mode, axis), ms in sorted(matches.items()):
            for p, e in sorted(_elo(ms).items(), key=lambda x: -x[1]):
                w.writerow([mode, axis, p, f"{e:.1f}"])
    print(f"summary -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--workers", type=int, default=8, help="parallel Gemini calls")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        judge(a.run_dir, a.workers)
    summarize(a.run_dir)


if __name__ == "__main__":
    main()
