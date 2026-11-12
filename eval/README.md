# Restyle-preprocessing ablation — eval harness

Measures the **per-axis marginal effect** of Objaverse-restyle preprocessing on
text-to-3D generation quality, across three backbones (TRELLIS.2, Hunyuan3D-2,
PartCrafter). This is the empirical core of the paper.

## Pipeline

The agent is the model under test — the eval drives it end-to-end, exactly as a user would.

```
text prompt + condition instruction
  → agent (LangGraph + Claude)   autonomously calls concept gen, restyle, backbone
  → .glb mesh
  → render_mesh_views()          8 PNGs from different camera angles
  → CLIP / VLM judge / ULIP-2    scores rendered views against the prompt caption
```

The condition instruction is prepended to the prompt to control which restyle
axes the agent uses. The agent's choice of backbone is unrestricted.

## Conditions

| Condition | Objects | Preprocessing | Review loop |
|-----------|---------|---------------|-------------|
| `raw` | all 50 | none (restyle tool is an identity pass-through) | off |
| `all_on` | all 50 | full restyle prompt | off |
| `loo_lighting` | all 50 | full restyle prompt **without** the lighting clause | off |
| `raw_review` | 20 submission objects | none | on |
| `all_on_review` | 20 submission objects | full restyle prompt | on |

Review loop = the agent's own quality control: it inspects the input image
(`tool_inspect_image`), generates, scores the mesh with `tool_score_alignment`
(CLIP, accept >= 0.22) and regenerates when told to — at most 2 regenerations.
Each regeneration uses a new backbone seed (42, 43, 44), so retrying the same
image yields a new sample; a requested new concept image also gets a new
image seed. The final output is the last mesh generated. With the loop off the
agent is told to stop after the first mesh, and a hard cap of one generation
is enforced. Every score/verdict the loop saw is logged (`review_scores`,
`review_verdicts`).

Paired contrasts (`eval/analyze.py`): restyle = `all_on − raw`, lighting =
`all_on − loo_lighting` (n=50); review with/without preprocessing,
preprocessing vs review, and their interaction (n=20).
**The review loop optimises CLIP, so judge its effect on the other metrics
(ULIP-2, T3Bench, pairwise judge), not on `clip_mean`.**

## Metrics

| Metric | Column | Notes |
|--------|--------|-------|
| CLIP | `clip_mean` | CLIP ViT-L/14 image-text cosine similarity, mean over 8 views. Always available. |
| VLM score | `vlm_score` | A pinned Gemini judge (temperature 0) rates 4 rendered views 1–10 on geometry, texture and semantic alignment. Our own prompt — **not** the published Gen3DEval. |
| ULIP-2 | `ulip_mean` | 3D-aware **point-cloud**↔text similarity (the metric Twist & Compute reports). Scored on Modal GPU — see below. |
| T3Bench Quality | `t3_quality` | T3Bench's own quality metric (our prompts are T3Bench prompts): ImageReward over 162×5 icosphere views, regional smoothing, `×20+50`. `eval/score_t3bench.py`, run post hoc. |
| T3Bench Alignment | `t3_alignment` | T3Bench's alignment metric, 1–5: BLIP-2 captions 12 views → LLM merges → LLM scores vs the prompt. Gemini stands in for GPT-4 (documented deviation). |
| Pairwise VLM judge | `judge_summary.csv` | Follows the **Gen3DEval protocol** (pairwise, 4 turntable views, RGB + normal maps, appearance / surface / text-fidelity, Elo) with Gemini as judge, because Gen3DEval's model is not released. Both presentation orders; a win counts only if they agree. `eval/vlm_judge.py`, run post hoc. |

**Naming in the paper:** call the pairwise metric "VLM judge (Gemini) following
the Gen3DEval protocol", never "Gen3DEval" — the Gen3DEval model is not public.

CLIP and VLM judge run automatically. ULIP-2 is skipped (logged as empty) if its
Modal app isn't deployed/reachable — the rest of the sweep is unaffected.

### ULIP-2 setup (Modal GPU)

ULIP-2's colored PointBERT encoder needs CUDA ops that don't build on macOS, so
it runs as a Modal app. The client (`eval/ulip_client.py`) samples a 10k xyz+rgb
point cloud from each mesh locally and POSTs it to the GPU scorer.

```bash
# deploy the scorer (from a workspace you're authed to)
.venv/bin/modal deploy eval/ulip_modal.py
# health check
curl https://<workspace>--ulip2-scorer-web.modal.run/
```

The endpoint URL is derived from `TRELLIS_WORKSPACE`. Once deployed, `ulip_mean`
populates on new sweeps, or backfill existing results from their saved meshes
(no regeneration):

```bash
.venv/bin/python eval/backfill_ulip.py eval/results_retrieved.csv
```

First build compiles `pointnet2_ops` and downloads ViT-bigG-14 (~5 GB) + the
402 MB checkpoint — slow on the first call, fast after.

## Dataset

50 generic object prompts ("a bicycle", "a teapot"; no attributes), built by
`eval/dataset/make_prompts.py`: object categories from the T3Bench
single-object prompts reduced to their head object — the 20 objects of the
original submission plus T3Bench lines in a seeded random order (seed
20261112), skipping duplicates, until there are 50.

Real photos (retrieved arm): for each object, ~8 openly licensed Wikimedia
Commons candidates (CC0 / public domain / CC BY / CC BY-SA, >= 500 px) are
fetched in Commons' search order (`eval/dataset/curate_photos.py fetch`), and
the lowest-numbered candidate meeting five fixed criteria is taken — real
photograph; whole object visible; the named object; object is the main
subject (everyday background allowed); no watermark or overlaid text. Picks
are recorded in `eval/dataset/picks.csv`; source page, author and licence of
each photo in `eval/dataset/images/manifest.csv`. An object with no candidate
meeting the criteria is replaced by the next object in the seeded order
(`NO_PHOTO` in `make_prompts.py`). All selection happened before any 3D
generation.

## Cost estimate

| Scale | Agent runs | Gemini calls (approx) | Claude scoring calls |
|-------|------------|----------------------|----------------------|
| 10 prompts (pilot) | 80 | ~160 (concept + restyle) | 80 |
| 20 prompts (full) | 160 | ~320 | 160 |

Each agent run may call concept gen + restyle + backbone — actual Gemini call
count depends on the agent's decisions.

## Input modes

| Mode | What the agent gets | Notes |
|------|---------------------|-------|
| `--input generated` (default) | Gemini generates a concept image from the **bare caption** (no added style/lighting/background terms) | the "generate with no extra information" baseline |
| `--input retrieved` | a real web photo auto-fetched per prompt (Tavily) and cached | the real-photo → Objaverse domain gap the restyle targets |
| `--input photo` | a hand-supplied real photo from `eval/dataset/images/<filename>` | same as retrieved but you curate the photos |

Each mode runs the full 8-condition restyle ablation, which gives the three
input strategies under comparison:

| Strategy | How to read it |
|----------|----------------|
| 1. Generate (no extra info) | `--input generated`, `raw` condition |
| 2. Retrieve | `--input retrieved`, `raw` condition |
| 3. Retrieve + restyle | `--input retrieved`, `all_on` (+ `loo_*` for the per-axis ablation) |

The generated pilot (`results_pilot.csv`) showed the per-axis effect is ~zero
because the concept images were already studio-clean. The **retrieved** arm is
where the restyle should actually matter. Each mode caches/tags its media
separately, so the arms never collide.

## Run

```bash
# generated-input pilot: first 10 prompts
.venv/bin/python eval/run_pilot.py --limit 10 --out eval/results_pilot.csv

# real-photo arm (needs photos in eval/dataset/images/) — SEPARATE output file
.venv/bin/python eval/run_pilot.py --input photo --out eval/results_photos.csv

# resume: ok rows in the CSV are skipped automatically
.venv/bin/python eval/run_pilot.py --limit 10 --out eval/results_pilot.csv
```

Requires Modal apps deployed and `.env` populated with `GEMINI_API_KEY` and
`TRELLIS_WORKSPACE` (see top-level README). All three metrics that run by
default (CLIP, VLM judge via Gemini) need no additional keys.

## Analyze

```bash
.venv/bin/python eval/analyze.py eval/results_pilot.csv
```

Prints three tables (one per metric): absolute mean by condition, then the
per-axis marginal effect with paired standard deviation and n.

## Sanity-check which axes are live

Before trusting (or spending compute on) the per-axis numbers, eyeball whether
each axis actually changes the restyled input the backbone receives:

```bash
.venv/bin/python eval/contact_sheet.py eval/results_pilot.csv
open eval/contact_sheet.html
```

For each prompt it lays out one row per condition showing the **restyled input**
plus the front mesh view. If `all_on` and a `loo_*` row look identical, that
axis isn't changing the input — no metric can detect an effect there, and the
fix is a stronger restyle clause, not more samples.

## Output columns

| Column | Description |
|--------|-------------|
| `image` | concept image filename (matches captions.csv) |
| `condition` | `raw`, `all_on`, or `loo_<axis>` |
| `backbone` | `trellis2`, `hunyuan3d2`, or `partcrafter` |
| `clip_mean` | mean CLIP score across rendered views |
| `clip_accept` | True if clip_mean ≥ 0.25 (rough quality gate) |
| `worst_view` | path to the lowest-scoring rendered view |
| `per_view` | JSON list of per-view CLIP scores |
| `vlm_score` | VLM judge score 1–10 (blank if API unavailable) |
| `ulip_mean` | mean ULIP-2 score (blank if not installed) |
| `concept_path` | path to the Gemini-generated concept image |
| `restyled_path` | path to the restyled image (= concept_path for raw) |
| `mesh_path` | path to the generated .glb mesh |
| `status` | `ok` or `error` |
| `error` | exception message if status=error |
| `seconds` | wall-clock time for this cell |

## Related work

| Paper | What they measured | Gap this work fills |
|-------|--------------------|---------------------|
| PartCrafter (NeurIPS 2025) | Qualitative pairs only | No metrics, single model |
| SceneTransporter (arXiv:2602.22785) | Qualitative pairs only | No metrics, single model |
| Twist & Compute (EurIPS 2025) | ULIP, one axis (rotation), one model | Multi-axis, multi-backbone, VLM editor |
