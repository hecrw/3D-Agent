# Paint3D on Modal

Async endpoint for [Paint3D](https://github.com/OpenTexture/Paint3D) —
text-to-texture for an untextured mesh.

| Route               | What it does                                | Returns        |
| ------------------- | ------------------------------------------- | -------------- |
| `POST /paint`       | mesh + prompt -> textured GLB               | `{job_id}`     |
| `GET  /jobs/{id}`   | poll                                        | 202 / 200 GLB / 5xx |

## What it does (and doesn't)

- **Does:** paint a high-res lighting-less albedo onto a mesh. Geometry is
  preserved; only the texture changes.
- **Doesn't:** edit geometry, add parts, or work well on already-textured
  meshes (the "lighting-less" assumption breaks when there's baked shading).

If you want to **edit** geometry and texture together, that's
threestudio_refine or MVEdit — not Paint3D.

## First-time setup

```sh
pip install modal
modal setup
modal secret create huggingface-secret HF_TOKEN=hf_xxx
modal deploy modal_app.py
```

First build is ~20–30 min (PyTorch 1.12.1 + cu116, kaolin precompiled wheel,
and prefetching SD 1.5 + ControlNet weights). Subsequent deploys are fast.

## Submitting

```sh
URL=https://<workspace>--paint3d-painter-web.modal.run

JOB=$(curl -sS -X POST \
  -F mesh=@untextured.obj \
  -F prompt='a wooden sword with intricate carvings' \
  "$URL/paint" | jq -r .job_id)

while true; do
  CODE=$(curl -sS -o textured.glb -w "%{http_code}" "$URL/jobs/$JOB")
  [ "$CODE" = "200" ] && { echo "done -> textured.glb"; break; }
  [ "$CODE" = "202" ] || { echo "err $CODE"; cat textured.glb; break; }
  echo "pending..."; sleep 15
done
```

`mesh` accepts `.obj`, `.glb`, or `.ply` — non-OBJ uploads are auto-converted
via trimesh before the pipeline runs.

## Form fields

| Field           | Default                                                | Notes |
| --------------- | ------------------------------------------------------ | ----- |
| `mesh`          | —                                                      | required |
| `prompt`        | —                                                      | required |
| `ip_image`      | —                                                      | optional IP-adapter style image |
| `seed`          | `0`                                                    |  |
| `sd_config`     | `controlnet/config/depth_based_inpaint_template.yaml`  | switch to a UV-only template for `pipeline_UV_only.py` |
| `render_config` | `paint3d/config/train_config_paint3d.py`               |  |

## Tuning

- GPU is `A10G` (24 GB) — Paint3D fits comfortably. `A100` for ~1.5–2× speedup.
- HF weights persist on `paint3d-hf-cache`; jobs on `paint3d-jobs`.
- `REPO_COMMIT` at the top of `modal_app.py` — pin to a sha for reproducibility.

## Caveats

- The exact CLI flags Paint3D's two pipeline scripts accept may have shifted
  since this app was written. If a flag rejects, check `pipeline_paint3d_stage1.py
  --help` inside a Modal shell:
  ```sh
  modal shell modal_app.py::Painter
  python pipeline_paint3d_stage1.py --help
  ```
  and adjust `_do_paint`'s subprocess args.
- The output GLB-bundling step uses a simple `trimesh.visual.TextureVisuals`
  wrap. If your mesh has no UVs, Paint3D's xatlas step inside the pipeline
  generates them — that's fine — but make sure the textured OBJ Paint3D
  actually exports is what we re-bundle. If colors look wrong, swap the
  re-bundle step for `tm = trimesh.load(<paint3d's exported textured OBJ>)`
  directly.
