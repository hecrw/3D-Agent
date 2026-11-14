#!/usr/bin/env python
"""T3Bench Quality + Alignment scores for every mesh in a run.

    conda run -n t3bench python eval/score_t3bench.py eval/runs/<run-id> --t3bench ~/T3Bench

Our 20 prompts are T3Bench single-object prompts, so we score them with T3Bench's
own metrics (He et al. 2023, github.com/THU-LYJ-Lab/T3Bench, pinned commit
6367462). The protocol is re-implemented from their run_eval_quality.py,
run_caption.py and run_eval_alignment.py (the repo has no licence file, so we
do not vendor its code). The two LLM prompt templates are read at runtime from
the pinned checkout, so the wording is exactly theirs.

Quality (0-100-ish, their scale):
  render 162 icosphere positions x 5 fields of view (render3d.T3Renderer),
  ImageReward-v1.0(prompt, view) -> max over the 5 FOVs per position ->
  3 passes of 1-ring neighbour averaging on the icosphere -> max -> x*20+50.
Alignment (1-5):
  BLIP-2 (flan-t5-xxl) captions 12 icosahedron views -> LLM merges captions
  into one -> LLM scores prompt-vs-caption 1-5 -> round.

Documented deviations (report these in the paper):
  * LLM: Gemini (pinned JUDGE_MODEL, temperature 0) instead of GPT-4 — no
    OpenAI key in this project.
  * BLIP-2 via Hugging Face transformers (Salesforce/blip2-flan-t5-xxl, same
    weights as LAVIS blip2_t5/pretrain_flant5xxl), with LAVIS's generate
    defaults (nucleus sampling, top_p 0.9, 5 beams, max 30 tokens). T3Bench
    sets no seed; we seed it so captions are reproducible.
  * Caption view order is the fixed icosahedron vertex order (T3Bench uses
    os.listdir order, which is arbitrary).
  * glTF meshes are rotated +Y-up -> +Z-up to match threestudio exports.

Adds columns t3_quality, t3_alignment, t3_caption to every results_*.csv in the
run (atomic rewrite). Per-mesh results are cached in <mesh>.t3bench.json, so
the script is resumable.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

T3_COMMIT = "6367462cfd04467af772c168c5cd8570964b9df3"
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.8-flash")
SEED = int(os.environ.get("GEMINI_SEED", "20261111") or 0)
BLIP2 = "Salesforce/blip2-flan-t5-xxl"
NEW_COLS = ["t3_quality", "t3_alignment", "t3_caption"]


# ---------------------------------------------------------------------------
# T3Bench prompt templates, read from the pinned checkout
# ---------------------------------------------------------------------------

def load_templates(t3: Path) -> dict[str, str]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=t3, capture_output=True,
                          text=True).stdout.strip()
    if head != T3_COMMIT:
        sys.exit(f"{t3} is at {head or '?'}, expected {T3_COMMIT}. "
                 f"Run: git -C {t3} checkout {T3_COMMIT}")

    def consts(path: Path, name: str) -> tuple[str | None, list[str]]:
        tree = ast.parse(path.read_text())
        first, appended = None, []
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) \
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
                first = n.value.value
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) \
                    and n.target.id == name and isinstance(n.value, ast.Constant):
                appended.append((n.lineno, n.value.value))
        # ast.walk is breadth-first, NOT source order — sort by line number
        return first, [v for _, v in sorted(appended)]

    merge_head, merge_tail = consts(t3 / "run_caption.py", "prompt_input")
    score_tpl, _ = consts(t3 / "run_eval_alignment.py", "prompt_to_gpt4_ori")
    if not (merge_head and merge_tail and score_tpl):
        sys.exit("could not find the prompt templates in the T3Bench checkout")
    # run_caption.py: head, then per view "view{i}: {caption}\n", then the tail
    return {"merge_head": merge_head, "merge_tail": merge_tail[-1], "score": score_tpl}


def process_text(text: str) -> str:
    """Same clean-up as T3Bench run_eval_alignment.process_text."""
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1]
    if text.endswith("."):
        text = text[:-1]
    return text.strip()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class Models:
    def __init__(self, need_quality: bool, need_alignment: bool):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.reward = self.blip = self.proc = self.gem = None
        if need_quality:
            import ImageReward as RM
            self.reward = RM.load("ImageReward-v1.0", device=self.dev)
        if need_alignment:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
            self.proc = Blip2Processor.from_pretrained(BLIP2)
            self.blip = Blip2ForConditionalGeneration.from_pretrained(
                BLIP2, torch_dtype=torch.float16).to(self.dev).eval()
            # transformers>=4.46 splices the 32 Q-Former outputs into <image>
            # placeholder tokens. With images only (no text) the processor adds
            # none and generate() fails ("shape [131072] ... indexing result of
            # shape [0]"). Captioning with text="" makes the processor prepend
            # the placeholders: the T5 encoder then sees [32 query embeds, </s>],
            # exactly LAVIS blip2_t5 unconditional captioning (prompt="").
            self.proc.num_query_tokens = self.blip.config.num_query_tokens
            from dotenv import load_dotenv
            from google import genai
            load_dotenv(REPO / ".env")
            self.gem = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
            # a call that never returns otherwise blocks the run forever; with a
            # 120 s timeout it raises and the retry loop tries again
            http_options=genai.types.HttpOptions(timeout=120_000))

    def caption(self, img) -> str:
        t = self.torch
        inputs = self.proc(images=img, text="", return_tensors="pt").to(self.dev, t.float16)
        itok = getattr(self.blip.config, "image_token_index", None)
        if itok is not None:
            n = int((inputs["input_ids"] == itok).sum())
            assert n == self.blip.config.num_query_tokens, f"{n} image tokens in BLIP-2 input"
        with t.no_grad():
            out = self.blip.generate(**inputs, do_sample=True, top_p=0.9, temperature=1.0,
                                     num_beams=5, max_new_tokens=30, min_length=1,
                                     repetition_penalty=1.0, length_penalty=1.0)
        return self.proc.batch_decode(out, skip_special_tokens=True)[0].strip()

    def llm(self, text: str) -> str:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            temperature=0.0, seed=SEED,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW))
        for attempt in range(5):
            try:
                return (self.gem.models.generate_content(
                    model=JUDGE_MODEL, contents=text, config=cfg).text or "").strip()
            except Exception as e:  # noqa: BLE001
                if "402" in str(e) or "credits are depleted" in str(e):
                    raise SystemExit(f"Gemini out of credit — stopping:\n{str(e)[:300]}")
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt * 3)
        return ""


# ---------------------------------------------------------------------------
# per-mesh scoring
# ---------------------------------------------------------------------------

def quality(models: Models, mesh: str, prompt: str) -> float:
    from render3d import T3Renderer, t3_icosphere
    ico = t3_icosphere()
    scores = {}
    with T3Renderer(mesh) as r:
        for idx, imgs in r.quality_views():
            scores[idx] = max(float(models.reward.score(prompt, im)) for im in imgs)
    for _ in range(3):  # regional convolution, exactly as T3Bench
        new = {}
        for idx in range(len(ico.vertices)):
            nb = ico.vertex_neighbors[idx]
            new[idx] = (scores[idx] + sum(scores[n] for n in nb)) / (len(nb) + 1)
        scores = new
    return max(scores.values()) * 20 + 50


def alignment(models: Models, tpl: dict, mesh: str, prompt: str) -> dict:
    from render3d import T3Renderer
    models.torch.manual_seed(SEED)
    with T3Renderer(mesh) as r:
        views = r.caption_views()
    caps = [models.caption(im) for im in views]
    merge = tpl["merge_head"]
    for i, c in enumerate(caps):
        merge += f"view{i + 1}: {c}\n"
    merge += tpl["merge_tail"]
    merged = process_text(models.llm(merge))
    ask = tpl["score"] + "Prompt: " + prompt + "\n" + "Prediction: " + merged
    reply = models.llm(ask)
    m = re.search(r"\d+(?:\.\d+)?", reply)
    score = float(round(float(m.group()))) if m else float("nan")
    return {"captions": caps, "merged": merged, "reply": reply, "alignment": score}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--t3bench", type=Path, default=Path.home() / "T3Bench")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-alignment", action="store_true")
    a = ap.parse_args()

    tpl = None if a.skip_alignment else load_templates(a.t3bench)
    csvs = sorted(a.run_dir.glob("results_*.csv"))
    todo = {}  # mesh -> prompt
    for f in csvs:
        for r in csv.DictReader(open(f)):
            if r["status"] == "ok" and r["mesh_path"]:
                todo[r["mesh_path"]] = r["prompt"]

    def cache_path(m):
        return Path(m).with_suffix(".t3bench.json")

    def cached(m):
        p = cache_path(m)
        return json.loads(p.read_text()) if p.exists() else {}

    need_q = not a.skip_quality and any("quality" not in cached(m) for m in todo)
    need_a = not a.skip_alignment and any("alignment" not in cached(m) for m in todo)
    print(f"{len(todo)} meshes; quality needed: {need_q}, alignment needed: {need_a}")
    models = Models(need_q, need_a) if (need_q or need_a) else None

    for i, (mesh, prompt) in enumerate(todo.items(), 1):
        c = cached(mesh)
        t0 = time.time()
        try:
            if need_q and "quality" not in c:
                c["quality"] = quality(models, mesh, prompt)
                cache_path(mesh).write_text(json.dumps(c, indent=1))
            if need_a and "alignment" not in c:
                c.update(alignment(models, tpl, mesh, prompt))
            c.update(t3bench_commit=T3_COMMIT, llm=JUDGE_MODEL, captioner=BLIP2, seed=SEED)
            cache_path(mesh).write_text(json.dumps(c, indent=1))
            print(f"[{i}/{len(todo)}] {Path(mesh).name}: quality={c.get('quality', '—')!s:.6} "
                  f"alignment={c.get('alignment', '—')}  ({time.time() - t0:.0f}s)")
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 — one bad mesh shouldn't stop the run
            print(f"[{i}/{len(todo)}] {Path(mesh).name}: FAILED {type(e).__name__}: {e}")

    for f in csvs:  # write the columns back
        rows = list(csv.DictReader(open(f)))
        fields = list(rows[0].keys()) if rows else []
        fields += [c for c in NEW_COLS if c not in fields]
        for r in rows:
            c = cached(r["mesh_path"]) if r.get("mesh_path") else {}
            r["t3_quality"] = f"{c['quality']:.2f}" if "quality" in c else ""
            r["t3_alignment"] = f"{c['alignment']:.0f}" if c.get("alignment") == c.get("alignment") and "alignment" in c else ""
            r["t3_caption"] = c.get("merged", "")
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(f)
        print(f"wrote t3 columns -> {f}")


if __name__ == "__main__":
    main()
