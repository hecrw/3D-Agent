# TRELLIS.2 on Modal

Async job pattern: submit returns a `job_id` immediately, then poll. This
sidesteps Modal's ~5 min web-proxy timeout so jobs of any length work.

| Service   | Submit           | Poll               | Result |
| --------- | ---------------- | ------------------ | ------ |
| Generator | `POST /generate` | `GET /jobs/{id}`   | `.glb` |
| Texturer  | `POST /texture`  | `GET /jobs/{id}`   | `.glb` |

`GET /jobs/{id}` returns:
- **202** — still running
- **200** + GLB body — done
- **500** + traceback text — job crashed

## First-time setup

```sh
pip install modal
modal setup

# Gated DINOv3 checkpoint — create a HF token with
# "Read access to contents of all public gated repos you can access".
modal secret create huggingface-secret HF_TOKEN=hf_xxx
```

## Deploy

```sh
modal deploy modal_app.py
```

First build ~20 min (CUDA extensions). Deploy prints two URLs:

```
https://<workspace>--trellis2-generator-web.modal.run
https://<workspace>--trellis2-texturer-web.modal.run
```

## Image-to-3D

```sh
# 1. submit
JOB=$(curl -s -X POST \
  -F image=@assets/example_image/T.png \
  -F pipeline_type=1024_cascade \
  https://<workspace>--trellis2-generator-web.modal.run/generate \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')

# 2. poll until done
URL=https://<workspace>--trellis2-generator-web.modal.run/jobs/$JOB
while true; do
  CODE=$(curl -s -o generated.glb -w "%{http_code}" $URL)
  [ "$CODE" = "200" ] && { echo "done: generated.glb"; break; }
  [ "$CODE" = "202" ] || { echo "error ($CODE): $(cat generated.glb)"; break; }
  echo "pending..."; sleep 10
done
```

Form fields (all optional except `image`):
- `pipeline_type`: `512` | `1024` | `1024_cascade` | `1536_cascade` (default `1024_cascade`)
- `seed`: int (default 42)
- `decimation_target`: int (default 1_000_000)
- `texture_size`: int (default 4096)
- `remesh`: bool (default true) — expensive; set `false` for ~10× faster postprocess

## PBR texturing

```sh
JOB=$(curl -s -X POST \
  -F image=@assets/example_texturing/image.webp \
  -F mesh=@assets/example_texturing/the_forgotten_knight.ply \
  https://<workspace>--trellis2-texturer-web.modal.run/texture \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')

# same poll loop against .../jobs/$JOB
```

## Notes

- Jobs persist in volume `trellis2-jobs`. Wipe with
  `modal volume rm trellis2-jobs --yes` if you want to clear them.
- The submit endpoint spawns work on the same class's GPU pool, so the
  pipeline stays hot across jobs; the ASGI handler itself doesn't do GPU work.
- Bumping `max_containers` in `modal_app.py` raises concurrency.
- Expensive postprocess (`remesh=True`, `decimation_target=1M`) typically
  adds 5–15 min on top of diffusion. Tune per your use.
