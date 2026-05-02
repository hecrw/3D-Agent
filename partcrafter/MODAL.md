# PartCrafter on Modal

Async endpoints for [PartCrafter](https://github.com/wgsxm/PartCrafter) —
single image → N 3D parts + composite mesh.

| Route                    | Class            | What it does                |
| ------------------------ | ---------------- | --------------------------- |
| `POST /generate`         | ObjectGenerator  | object with 1–16 parts      |
| `POST /generate-scene`   | SceneGenerator   | scene with 1–16 parts       |
| `GET  /jobs/{id}`        | (both)           | poll for ZIP result         |

Each job returns a single composite GLB — all parts baked into one mesh
with per-part vertex colors (same response shape as the TRELLIS deployment).

## Deploy

```sh
pip install modal
modal setup
modal deploy partcrafter_modal.py
```

First build is quick (5–10 min) — pure pip, no CUDA source compiles.
Two URLs get printed:

```
https://<workspace>--partcrafter-objectgenerator-web.modal.run
https://<workspace>--partcrafter-scenegenerator-web.modal.run
```

## Object generation

```sh
# 1. submit
RESP=$(curl -s -X POST \
  -F image=@my_photo.png \
  -F num_parts=3 \
  -F rmbg=true \
  https://<workspace>--partcrafter-objectgenerator-web.modal.run/generate)
JOB=$(echo "$RESP" | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')

# 2. poll
URL=https://<workspace>--partcrafter-objectgenerator-web.modal.run/jobs/$JOB
while true; do
  CODE=$(curl -s -o generated.glb -w "%{http_code}" $URL)
  [ "$CODE" = "200" ] && { echo "done: generated.glb"; break; }
  [ "$CODE" = "202" ] || { echo "error $CODE"; cat generated.glb; break; }
  echo "pending..."; sleep 10
done
```

Form fields:
- `num_parts` (required) — 1 to 16
- `seed` (default 0)
- `num_tokens` (default 1024)
- `num_inference_steps` (default 50)
- `guidance_scale` (default 7.0)
- `rmbg` (default true) — remove background before generation

## Scene generation

Same shape, different endpoint/class:

```sh
curl -s -X POST \
  -F image=@room_photo.jpg \
  -F num_parts=6 \
  https://<workspace>--partcrafter-scenegenerator-web.modal.run/generate-scene
```

## Tuning

- Default GPU is **L4** (~$1/hr). Bump to `A100` or `H100` in
  `partcrafter_modal.py` if you need faster inference.
- `scaledown_window=300` — container sleeps after 5 min idle.
- HF weights live on volume `partcrafter-hf-cache`; jobs on `partcrafter-jobs`.
  Wipe with `modal volume rm <name> --yes` if needed.
- Pin a specific upstream commit by setting `REPO_COMMIT` at the top of
  `partcrafter_modal.py`.
