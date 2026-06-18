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
  → CLIP / Gen3DEval / ULIP-2    scores rendered views against the prompt caption
```

The condition instruction is prepended to the prompt to control which restyle
axes the agent uses. The agent's choice of backbone is unrestricted.

## Conditions (8)

| Condition | Restyle |
|-----------|---------|
| `raw` | no restyle — concept image fed straight to backbone |
| `all_on` | all 6 axes active (background, framing, view, lighting, isolation, part_visibility) |
| `loo_background` | all axes **except** background |
| `loo_framing` | all axes **except** framing |
| `loo_view` | all axes **except** view |
| `loo_lighting` | all axes **except** lighting |
| `loo_isolation` | all axes **except** isolation |
| `loo_part_visibility` | all axes **except** part_visibility |

Marginal effect of axis X = `score(all_on) − score(loo_X)`. Positive = axis helps.

## Metrics

| Metric | Column | Notes |
|--------|--------|-------|
| CLIP | `clip_mean` | CLIP ViT-L/14 image-text cosine similarity, mean over 8 views. Always available. |
| Gen3DEval | `gen3deval` | The agent's Gemini model (`agent.llm`) rates rendered views 1–10 on geometry, texture, and semantic alignment. Uses the existing `GEMINI_API_KEY` — no extra account needed. |
| ULIP-2 | `ulip_mean` | 3D-aware **point-cloud**↔text similarity (the metric Twist & Compute reports). Scored on Modal GPU — see below. |

CLIP and Gen3DEval run automatically. ULIP-2 is skipped (logged as empty) if its
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

20 text prompts drawn from the T3Bench single-object benchmark
(arXiv:2310.02977), covering vehicles, animals, furniture, instruments, and
everyday objects. Prompts are in `eval/dataset/captions.csv`.

Concept images are generated automatically by `run_pilot.py` and cached in
`eval/dataset/images/`. Delete an image to regenerate it.

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
default (CLIP, Gen3DEval via Gemini) need no additional keys.

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
| `gen3deval` | VLM judge score 1–10 (blank if API unavailable) |
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
