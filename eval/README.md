# Restyle-preprocessing ablation — eval harness

Measures the **per-axis marginal effect** of Objaverse-restyle preprocessing on
text-to-3D generation quality, across three backbones (TRELLIS.2, Hunyuan3D-2,
PartCrafter). This is the empirical core of the paper.

## Pipeline

```
text prompt
  → generate_concept_image()     Gemini generates a concept PNG
  → restyle_to_objaverse()       Gemini restyles to Objaverse look  (skipped for "raw")
  → backbone (Modal)             generates a .glb mesh
  → render_mesh_views()          8 PNGs from different camera angles
  → CLIP / Gen3DEval / ULIP-2    scores rendered views against the prompt caption
```

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
| Gen3DEval | `gen3deval` | Claude Haiku rates rendered views 1–10 on geometry, texture, and semantic alignment. Requires `ANTHROPIC_API_KEY` in `.env`. |
| ULIP-2 | `ulip_mean` | 3D-aware image-text similarity. Optional — see install below. |

CLIP and Gen3DEval run automatically. ULIP-2 is skipped (logged as empty) if the
library is not installed — the rest of the sweep is unaffected.

### Installing ULIP-2 (optional)

```bash
git clone https://github.com/salesforce/ULIP
pip install -e ULIP/
```

Once installed, `ulip_mean` will be populated on the next run for any rows not
yet in the CSV.

## Dataset

20 text prompts drawn from the T3Bench single-object benchmark
(arXiv:2310.02977), covering vehicles, animals, furniture, instruments, and
everyday objects. Prompts are in `eval/dataset/captions.csv`.

Concept images are generated automatically by `run_pilot.py` and cached in
`eval/dataset/images/`. Delete an image to regenerate it.

## Cost estimate

| Scale | Mesh gens | Gemini concept calls | Gemini restyle calls | Claude scoring calls |
|-------|-----------|----------------------|----------------------|----------------------|
| 10 prompts (pilot) | 240 | 10 (once, cached) | 210 | 240 |
| 20 prompts (full) | 480 | 20 (once, cached) | 420 | 480 |

Mesh settings are held cheap and constant (`--cheap` is the default) so mesh
quality does not confound the restyle signal.

## Run

```bash
# pilot: first 10 prompts, all backbones
.venv/bin/python eval/run_pilot.py --limit 10 --out eval/results_pilot.csv

# single backbone while iterating
.venv/bin/python eval/run_pilot.py --limit 10 --backbones trellis2

# resume: rows already in the CSV are skipped automatically
.venv/bin/python eval/run_pilot.py --limit 10 --out eval/results_pilot.csv
```

Requires Modal apps deployed and `.env` populated with `GEMINI_API_KEY`,
`ANTHROPIC_API_KEY`, and `TRELLIS_WORKSPACE` (see top-level README).

## Analyze

```bash
.venv/bin/python eval/analyze.py eval/results_pilot.csv
```

Prints three tables (one per metric): absolute mean by condition × backbone,
then the per-axis marginal effect with paired standard deviation and n.

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
