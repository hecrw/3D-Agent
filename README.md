# 3D-Agent

A Django chat application that wraps an LLM agent capable of turning text prompts
and reference images into textured 3D models. The agent reasons over a set of
tools — image generation, web/image search, GPU-hosted 3D pipelines, and two
quality "judges" — and streams its progress back to a chat UI.

## How it works

```
Browser (chat UI, SSE stream)
        │
        ▼
Django (core/ + chat_interface/)  ──  SQLite (sessions, messages, gallery)
        │
        ▼
LangChain agent (agent.py)  ──  Gemini (gemini-flash-latest)
        │
        ├─ Gemini image model      → concept images, Objaverse restyle
        ├─ Tavily                  → web search, image search
        ├─ Modal GPU endpoints     → TRELLIS.2, PartCrafter, Hunyuan3D-2
        ├─ pyrender / trimesh      → render mesh views
        └─ CLIP                    → alignment scoring (judge)
```

The agent (`agent.py`) is built with `langchain.agents.create_agent` over the
tools in `tools.py`. It can:

- Generate a concept image from text (Gemini) or download one from image search.
- Restyle an image into the clean "Objaverse" single-object look.
- Submit an image to a 3D pipeline (TRELLIS.2, PartCrafter, or Hunyuan3D-2) hosted
  on [Modal](https://modal.com), then poll and download the resulting `.glb`.
- Render multi-view PNGs of a generated mesh.
- Judge quality two ways: a **VLM judge** (`tool_inspect_image`, Gemini vision)
  and a **CLIP alignment score** (`tool_score_alignment`) that flags the worst-
  matching camera view.

## Project layout

| Path | Purpose |
|------|---------|
| `agent.py` | LangChain agent: tool wrappers, system prompt, streaming + status translation |
| `tools.py` | Core implementations: Gemini, Tavily, Modal job runner, mesh rendering, CLIP |
| `core/` | Django project (settings, urls, wsgi/asgi) |
| `chat_interface/` | Django app: models, views, SSE endpoints, gallery |
| `TRELLIS.2/`, `PartCrafter/`, `hunyuan3d-2/` | Modal apps (`modal_app.py`) for each GPU pipeline |
| `media/3d_outputs/` | Generated images and `.glb` files (served at `/media/`) |

## Prerequisites

- Python (see `.python-version`)
- [uv](https://github.com/astral-sh/uv) for dependency management
- A [Modal](https://modal.com) account with the CLI installed and authenticated
  (`modal token new`) — the 3D pipelines run there, and result download falls
  back to `modal volume get`.
- API keys for Google Gemini and Tavily.

## Setup

1. **Install dependencies** (uses `pyproject.toml` / `uv.lock`):

   ```bash
   uv sync
   ```

   (Or `pip install -r requirements.txt` into a virtualenv.)

2. **Create a `.env`** in the project root:

   ```dotenv
   GEMINI_API_KEY=your-gemini-key
   TAVILY_API_KEY=your-tavily-key
   TRELLIS_WORKSPACE=your-modal-workspace
   ```

   `TRELLIS_WORKSPACE` is your Modal workspace slug. `tools.py` builds the
   pipeline URLs from it, e.g.
   `https://<workspace>--trellis2-generator-web.modal.run`. `.env` is
   gitignored — keep your real keys out of version control.

3. **Deploy the Modal apps** (one per pipeline). See each folder's `MODAL.md`
   for details, then:

   ```bash
   modal deploy TRELLIS.2/modal_app.py
   modal deploy PartCrafter/modal_app.py
   modal deploy hunyuan3d-2/modal_app.py
   ```

   These expose the `*-web.modal.run` endpoints the agent calls. The expected
   deployed function names are:

   - `trellis2-generator-web`, `trellis2-texturer-web`
   - `partcrafter-objectgenerator-web`, `partcrafter-scenegenerator-web`
   - `hunyuan3d-2-generator-web`

4. **Run Django migrations:**

   ```bash
   python manage.py migrate
   ```

## Running

```bash
python manage.py runserver
```

Then open <http://127.0.0.1:8000/>:

- `/` — landing page with the central prompt box
- `/chat/<id>/` — a chat session
- `/gallery/` — all generated assets

Type a request (e.g. "make a 3D model of a cyberpunk drone") or upload a
reference image. The agent streams friendly status updates over Server-Sent
Events while it runs the pipeline, and the resulting `.glb` shows up in the chat
and the gallery.

## Notes

- **First CLIP call is slow.** `tool_score_alignment` loads
  `openai/clip-vit-large-patch14` into the Django process on first use; on a
  machine without CUDA it runs on CPU.
- **Modal cold starts.** The job runner in `tools.py` has generous retry,
  backoff, and polling budgets to tolerate GPU container cold starts; a first
  generation can take a while.
- **Stopping a job.** The UI's stop button cancels the in-flight Modal call via
  its `fc-...` call ID.

## License

See [LICENSE](LICENSE).
