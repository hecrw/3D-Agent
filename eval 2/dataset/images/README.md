# Real photos for `--input photo` mode

Drop one **real photograph** here for each row in `../captions.csv`, named
**exactly** as the `filename` column (e.g. `bicycle_01.png`, `cat_01.png`).

The whole point of the photo arm is the **domain gap**: these should be
in-the-wild photos (phone snaps, product shots, messy backgrounds, real
lighting) — NOT clean studio renders and NOT AI-generated images. That gap is
what the restyle is supposed to bridge, so it's where a per-axis effect should
actually appear.

The `caption` for each file is the text CLIP/Gen3DEval score against, so the
photo must depict what the caption says. If you use different objects than the
T3Bench prompts, edit the captions in `../captions.csv` to match your photos.

Sources: your own phone photos, Google Scanned Objects (CC-BY), or OmniObject3D
real captures.

Run the photo arm into a SEPARATE results file so it doesn't mix with the
generated-input run:

```bash
.venv/bin/python eval/run_pilot.py --input photo --out eval/results_photos.csv
.venv/bin/python eval/analyze.py eval/results_photos.csv
```
