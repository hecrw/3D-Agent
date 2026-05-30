# Restyle-preprocessing ablation — eval harness

Measures the **per-axis marginal effect** of the Objaverse-restyle preprocessing
step on image-to-3D quality, across three backbones (TRELLIS.2, Hunyuan3D-2,
PartCrafter). This is the empirical core of the paper.

## Conditions (8)

For the 6 restyle axes in `tools.RESTYLE_AXES`
(`background, framing, view, lighting, isolation, part_visibility`):

| Condition | Restyle prompt |
|-----------|----------------|
| `raw` | no restyle — feed the original image straight to the backbone |
| `all_on` | all 6 axes enabled (the current production prompt) |
| `loo_<axis>` (×6) | all axes **except** `<axis>` — isolates that axis's marginal contribution |

The marginal effect of an axis = `score(all_on) − score(loo_<axis>)`.

## Dataset

Real, in-the-wild photographs (the whole premise is the domain gap between
Objaverse renders and real photos — do **not** use generated concept images).

- Put images in `eval/dataset/images/`.
- Provide `eval/dataset/captions.csv` with columns `filename,caption`.
  The caption is the ground-truth text used by the CLIP/ULIP metric.

A stratified set of ~30 spanning object categories is the target; start with
~10 for the pilot.

## Cost

generations = images × conditions × backbones (raw needs no restyle call).
Pilot: 10 × 8 × 3 = **240** mesh generations.
Run with cheap, fixed mesh settings (`--cheap`, the default) so mesh quality is
held constant and doesn't confound the restyle effect.

## Run

```bash
# pilot: first 10 images, all backbones, cheap settings
.venv/bin/python eval/run_pilot.py --limit 10 --out eval/results_pilot.csv

# single backbone while iterating
.venv/bin/python eval/run_pilot.py --limit 10 --backbones trellis2

# resume: rows already in the CSV are skipped
.venv/bin/python eval/run_pilot.py --limit 10 --out eval/results_pilot.csv
```

Requires the Modal apps deployed and `.env` populated (see top-level README).

## Output

`results_pilot.csv` — one row per (image, condition, backbone) with the mean
CLIP alignment score and per-view scores. Aggregate with `analyze.py`.
```bash
.venv/bin/python eval/analyze.py eval/results_pilot.csv
```
