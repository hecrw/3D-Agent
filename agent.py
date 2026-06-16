import time
import re
from pathlib import Path
from langchain_core.tools import tool

from tools import (
    generate_concept_image,
    restyle_to_objaverse,
    edit_image,
    trellis2,
    trellis2_texture,
    partcrafter,
    hunyuan3d2,
    web_search,
    image_search,
    download_image,
    render_mesh_views,
    compose_scene,
    check_alignment,
)
 
import os
from django.conf import settings
from pathlib import Path

OUT = Path(settings.MEDIA_ROOT) / "3d_outputs"
OUT.mkdir(parents=True, exist_ok=True)

def _stamp(prefix: str, ext: str) -> str:
    return str(OUT / f"{prefix}_{int(time.time())}.{ext}")
 
 
import sys
import contextlib
import io


class _TeeStdout(io.TextIOBase):
    """Write-through stdout proxy: forwards to the real terminal AND a buffer."""
    def __init__(self, buf, original):
        self._buf = buf
        self._original = original
    def write(self, s):
        try:
            self._original.write(s)
            self._original.flush()
        except Exception:
            pass
        return self._buf.write(s)
    def flush(self):
        try:
            self._original.flush()
        except Exception:
            pass
        self._buf.flush()

def tool_wrapper(func):
    def wrapper(*args, **kwargs):
        # This is a bit tricky inside LangChain tools, 
        # but we can try to intercept prints.
        return func(*args, **kwargs)
    return wrapper

@tool
def tool_web_search(query: str) -> str:
    """Search the web for facts, references, or context."""
    return web_search(query)

@tool
def tool_image_search(query: str) -> str:
    """Search the web for IMAGES of a subject."""
    urls = image_search(query)
    if not urls:
        return "No images found."
    return "Image URLs:\n" + "\n".join(f"- {u}" for u in urls)

@tool
def tool_download_image(url: str) -> str:
    """Download an image from a URL and save it locally."""
    out = _stamp("downloaded", "jpg")
    try:
        path = download_image(url=url, out_path=out)
    except Exception as e:
        return (
            f"Could not download {url} ({type(e).__name__}: {e}). "
            "The host likely blocks hotlinking. Try a different image URL "
            "from the search results."
        )
    return f"Image saved: {path}"

@tool
def tool_generate_concept_image(prompt: str) -> str:
    """Generate a concept PNG image from a text description."""
    out = _stamp("concept", "png")
    path = generate_concept_image(prompt=prompt, out_path=out)
    return f"Concept image saved: {path}"

@tool
def tool_restyle_to_objaverse(image_path: str) -> str:
    """Restyle any photo or image into the clean Objaverse dataset style. 
    this is done usually after generation or retrieving an image online"""
    out = _stamp("objaverse", "png")
    path = restyle_to_objaverse(image_path=image_path, out_path=out)
    return f"Restyled image saved: {path}"

@tool
def tool_edit_image(image_path: str, instruction: str) -> str:
    """Apply a targeted, natural-language edit to an existing image and return the
    NEW image path. Examples of instructions: 'make the background white', 'remove
    the text', 'make it look more realistic', 'show the handle'. Use this to act on
    human feedback at the approval gate (a rejection) instead of regenerating from
    scratch — then send the edited image to the 3D tool for re-approval."""
    out = _stamp("edited", "png")
    path = edit_image(image_path=image_path, instruction=instruction, out_path=out)
    return f"Edited image saved: {path}"

@tool
def tool_trellis2(image_path: str) -> str:
    """Run the TRELLIS pipeline: convert a clean image into a 3D GLB file that is
    ALREADY FULLY TEXTURED. The returned GLB is the finished asset — do NOT call
    tool_trellis2_texture on it."""
    out = _stamp("trellis2", "glb")
    path = trellis2(image_path=image_path, out_path=out)
    return f"3D model saved: {path}"

@tool
def tool_trellis2_texture(image_path: str, mesh_path: str) -> str:
    """Re-texture an EXISTING, UNTEXTURED mesh (e.g. a bare .glb the user supplied
    or a geometry-only output). Do NOT use this on a mesh produced by tool_trellis2
    or tool_hunyuan3d2 — those are already textured and re-texturing wastes a GPU
    job and can fail."""
    out = _stamp("trellisTex", "glb")
    path = trellis2_texture(image_path=image_path, mesh_path=mesh_path, out_path=out)
    return f"Textured model saved: {path}"

@tool
def tool_partcrafter(image_path: str, num_parts: int = 3, scene: bool = False) -> str:
    """Run PartCrafter: Generate part based objects"""
    out = _stamp("parts", "glb")
    path = partcrafter(image_path=image_path, out_path=out, num_parts=num_parts, scene=scene)
    return f"Part-decomposed model saved: {path}"

@tool
def tool_hunyuan3d2(image_path: str) -> str:
    """Run Hunyuan3D-2: convert a clean image into a fully textured 3D GLB file."""
    out = _stamp("hunyuan", "glb")
    path = hunyuan3d2(image_path=image_path, out_path=out)
    return f"3D model saved: {path}"

@tool
def tool_render_mesh_views(mesh_path: str) -> str:
    """Render 6 axis-aligned views (front/back/left/right/top/bottom) of a 3D mesh
    as PNGs. Use to inspect what a generated GLB looks like, or to produce a
    reference image you can feed back into another pipeline or a VLM.
    Accepts .glb/.obj/.ply/.stl. Returns the directory containing the PNGs
    plus the per-view paths."""
    out_dir = OUT / f"views_{int(time.time())}"
    paths = render_mesh_views(mesh_path=mesh_path, out_dir=out_dir)
    lines = [f"Rendered {len(paths)} views to {out_dir}:"]
    lines += [f"- {name}: {p}" for name, p in paths.items()]
    return "\n".join(lines)

@tool
def tool_compose_scene(placements: str) -> str:
    """Combine existing 3D meshes (.glb) into ONE scene in a shared coordinate
    space and return the combined .glb. Use this to arrange already-generated
    assets together — e.g. "place the cat next to the dog", "put these in a row",
    "set the lamp at x=1".

    placements: a JSON list, one entry per mesh:
      [{"mesh_path": "<absolute path to a .glb>",   // required
        "x": <float>, "y": <float>, "z": <float>,   // optional world coords of the
                                                     //   object's center-bottom (m)
        "scale": <float>,        // optional, default 1.0
        "rot_z_deg": <float>}]   // optional yaw about the vertical axis
    Omit x/y/z on ALL entries to auto-arrange them side by side in a row on the
    ground. Generated meshes are ~unit-sized, so use scale to match relative sizes
    and offsets of ~0.5-2.0 to separate objects. Pass the [Previously generated
    asset: ...] paths from earlier turns as mesh_path."""
    out = _stamp("scene", "glb")
    path = compose_scene(placements=placements, out_path=out)
    return f"Composed scene saved: {path}"

@tool
def tool_inspect_image(image_path: str, question: str) -> str:
    """View a local image with vision and answer a question about it.

    Use this to:
    - Judge quality of generated concept images or rendered 3D-model views
    - Verify a retrieved/downloaded image actually depicts the requested subject

    image_path: absolute path to a local image (.png/.jpg/.jpeg/.webp/.gif).
    question: what to assess, e.g. "Does this depict a single cyberpunk drone,
              centered, on a clean background? List any defects."
    """
    import base64, mimetypes
    from langchain_core.messages import HumanMessage

    if not os.path.isfile(image_path):
        return f"inspect_image: file not found at {image_path}"

    mime, _ = mimetypes.guess_type(image_path)
    if not mime or not mime.startswith("image/"):
        mime = "image/png"

    try:
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
    except OSError as e:
        return f"inspect_image: could not read {image_path} ({e})"

    msg = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
    ])
    try:
        resp = llm.invoke([msg])
    except Exception as e:
        return f"inspect_image: model call failed ({type(e).__name__}: {e})"

    raw = resp.content
    if isinstance(raw, list):
        return "".join(
            b.get("text", "") for b in raw
            if isinstance(b, dict) and b.get("type") == "text"
        ) or str(raw)
    return raw if isinstance(raw, str) else str(raw)


@tool
def tool_score_alignment(mesh_path: str, prompt: str) -> str:
    """Score how well a generated mesh matches the text prompt by rendering
    multi-view images and comparing each view to the prompt with CLIP.
    Use this after topology passes, to decide whether to keep or regenerate
    the candidate. Flags the worst view by name so you can target a regeneration
    or texture pass at the failing angle.
    Returns a one-line verdict including a recommended next action:
    'proceed' or 'regenerate'."""
    out_dir = _stamp("views", "")
    paths = render_mesh_views(mesh_path, out_dir, views="default")
    r = check_alignment(paths, prompt)
    worst = f" | worst_view: {r.worst_view}" if not r.accept and r.worst_view else ""
    return (
        f"Alignment score: {r.score:.2f} | accept: {r.accept} | "
        f"next_action: {r.next_action}{worst} | {r.summary}"
    )

 
ALL_TOOLS = [
    tool_web_search,
    tool_image_search,
    tool_download_image,
    tool_generate_concept_image,
    tool_restyle_to_objaverse,
    tool_edit_image,
    tool_trellis2,
    tool_trellis2_texture,
    tool_partcrafter,
    tool_hunyuan3d2,
    tool_render_mesh_views,
    tool_compose_scene,
    tool_inspect_image,
    tool_score_alignment,
]


# --- Friendly status translation ---

TOOL_LABELS = {
    "tool_web_search": "Searching the web",
    "tool_image_search": "Searching for reference images",
    "tool_download_image": "Downloading image",
    "tool_generate_concept_image": "Generating concept image",
    "tool_restyle_to_objaverse": "Restyling to Objaverse",
    "tool_edit_image": "Editing image",
    "tool_trellis2": "Generating 3D model",
    "tool_trellis2_texture": "Re-texturing model",
    "tool_partcrafter": "Decomposing into parts",
    "tool_hunyuan3d2": "Generating 3D model",
    "tool_render_mesh_views": "Rendering views",
    "tool_compose_scene": "Composing scene",
    "tool_inspect_image": "Inspecting image",
    "tool_score_alignment": "Scoring alignment",
}

PIPELINE_LABELS = {
    "trellis2": "TRELLIS",
    "trellis2_texture": "TRELLIS texture",
    "partcrafter": "PartCrafter",
    "hunyuan3d2": "Hunyuan3D",
}


def _friendly_tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, name.replace("tool_", "").replace("_", " ").capitalize())


_TAG_LINE_RE = re.compile(r'^\[([^\]]+)\]\s*(.*)$')


def _friendly_status(line: str):
    """Map a raw stdout line from tools.py into a user-facing status, or None to drop."""
    line = line.strip()
    if not line:
        return None
    m = _TAG_LINE_RE.match(line)
    if not m:
        return None  # drop anything we don't recognize

    tag, rest = m.group(1), m.group(2)

    if tag == "gemini":
        if rest.startswith("concept"):
            return "Generating concept image..."
        if rest.startswith("restyle"):
            return "Restyling to Objaverse..."
        return None  # "saved ..." etc.
    if tag == "tavily":
        if rest.startswith("image search"):
            return "Searching for reference images..."
        if rest.startswith("search"):
            return "Searching the web..."
        return None
    if tag == "download":
        if rest.startswith("saved"):
            return None
        return "Downloading image..."
    if tag == "views":
        return None  # per-view path spam

    label = PIPELINE_LABELS.get(tag)
    if label is None:
        return None  # unknown tag, drop

    if rest.startswith("submit ->") or rest.startswith("submit "):
        if "failed" in rest:
            return f"{label}: reconnecting..."
        return f"Sending image to {label}..."
    if rest.startswith("job="):
        return None  # already surfaced as call_id
    if rest.startswith("pending"):
        return f"{label} is working on it..."
    if rest.startswith("downloading from volume"):
        return f"Receiving result from {label}..."
    if rest.startswith("volume download failed"):
        return None
    if rest.startswith("done"):
        return f"{label} finished."
    if "retrying" in rest:
        return f"{label} retrying..."
    if rest.startswith("WARN"):
        return None
    return None



from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0.2
)

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.sqlite import SqliteSaver

# --- Human-in-the-loop gate ---------------------------------------------------
# Pause for human approval right before any expensive 3D-generation tool runs.
# Because the pipeline order is (image produced) -> (3D generation tool), gating
# the generation tools surfaces the "does this image work?" decision: the agent
# pauses showing the image it is about to send to the pipeline, and the human
# can approve it, or reject to make the agent regenerate the image.
# Each gated tool allows two decisions: approve (run as-is) and reject (the human
# types feedback which the agent sees as an error and uses to regenerate). We
# omit "edit" (fiddling with raw tool args is not a useful human action here) and
# "respond" (a human cannot fabricate a GLB on the tool's behalf).
_GATE = {"allowed_decisions": ["approve", "reject"]}
GATED_TOOLS = {
    "tool_trellis2": _GATE,
    "tool_trellis2_texture": _GATE,
    "tool_partcrafter": _GATE,
    "tool_hunyuan3d2": _GATE,
}

# Durable checkpointer so a paused run survives a server restart. The thread_id
# (the chat session id) is supplied per-invocation in process_chat_stream.
# `.setup()` creates the checkpoint tables if missing; check_same_thread=False
# because Django may resume on a different worker thread than the one that paused.
import sqlite3
from django.conf import settings as _dj_settings

_CKPT_PATH = str(Path(_dj_settings.BASE_DIR) / "agent_checkpoints.sqlite3")
_ckpt_conn = sqlite3.connect(_CKPT_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_ckpt_conn)
checkpointer.setup()

hitl = HumanInTheLoopMiddleware(
    interrupt_on=GATED_TOOLS,
    description_prefix="Approve before generating the 3D model",
)

agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    middleware=[hitl],
    checkpointer=checkpointer,
    system_prompt="""You are a helpful assistant.
    If the user provides an image, you will see an [Uploaded Image Local Path: ...] in the message.
    CRITICAL: Never use 'input_file_0.png' or any other generated path.
    Use the EXACT absolute path provided in the [Uploaded Image Local Path: ...] tag for any tool that requires an 'image_path'.
    If you see an image but no local path is provided in the text, ask the user for clarification.
    Make conversation with the user, but if his request can be
    done using the tools you have, use the tools to fulfill his request.

    PIPELINE ORDER (always follow this): (1) obtain an image — generate one with
    tool_generate_concept_image, or search+download a real photo, or use the
    user's uploaded image; (2) ALWAYS pass that image through
    tool_restyle_to_objaverse to normalize it into the clean Objaverse asset
    style; (3) send the RESTYLED image to a 3D-generation tool (tool_trellis2 /
    tool_hunyuan3d2 / tool_partcrafter). The restyle in step 2 is MANDATORY for
    EVERY image — generated, retrieved, user-uploaded, or edited — because the 3D
    backbones are trained on Objaverse-style renders and fail on raw photos. Never
    send a raw photo, a raw generated image, or a freshly edited image straight to
    a 3D tool without restyling it first. The only image you ever hand to a 3D tool
    is the output path of tool_restyle_to_objaverse.

    HANDLING A REJECTED IMAGE: every 3D-generation tool pauses for human approval
    of the image first. If the human REJECTS, you will receive that tool call back
    as an error whose text is the human's feedback about the IMAGE (e.g. "make him
    more realistic", "white background", "remove the text"). Do NOT re-call the
    same 3D tool with the same image. Instead FIRST produce a NEW image that
    addresses the feedback — call tool_edit_image(image_path, instruction) to edit
    the current image, or tool_generate_concept_image to remake it — THEN run it
    through tool_restyle_to_objaverse (step 2 of the pipeline is still mandatory),
    and only THEN call the 3D tool again with the restyled image path. The gate
    will re-open showing the new image. Never send the rejected image back
    unchanged, and never skip the restyle.

    Earlier assistant turns in the history may carry a
    [Previously generated asset: <absolute path>] tag — that is the local file of
    a 3D mesh (.glb) or image you produced before. When the user refers to a past
    result ("the last one", "that mesh", "make it bigger", "now texture it",
    "render the previous object"), reuse the most recent such path as the
    mesh_path / image_path argument instead of generating from scratch. If several
    are present and it is ambiguous, prefer the most recent one. To put TWO OR MORE
    existing assets together in one scene ("place the cat next to the dog", "add the
    cat onto the couch", "line these up"), call tool_compose_scene with their
    [Previously generated asset: ...] paths — do NOT regenerate them. These bracketed
    tags are internal context for you only — NEVER repeat them or any file path in
    your reply to the user. Refer to assets in plain language ("your previous
    model", "the cat you generated").

    IMPORTANT: tool_trellis2 and tool_hunyuan3d2 already return a FULLY TEXTURED
    GLB. Once one of them produces a mesh, that mesh is the finished asset: return
    it to the user and STOP. Never call tool_trellis2_texture on a TRELLIS or
    Hunyuan output — it is already textured. Only use tool_trellis2_texture on a
    mesh that is genuinely untextured (e.g. a bare geometry file the user gave you).

    You can also act as a vision model. After any tool that produces or
    fetches an image (concept generation, Objaverse restyle, image search +
    download, mesh-view renders), call tool_inspect_image(image_path, question)
    to actually look at it and judge quality, subject match, framing, and
    obvious defects. Use that judgment to decide whether to proceed to the
    next pipeline step, regenerate with a tweaked prompt, or pick a different
    reference URL. Be specific in the question (subject, expected style,
    things that would disqualify it).""",
)


def generate_chat_title(user_prompt):
    """Summarizes a user prompt into a 2-4 word snappy title."""
    try:
        response = llm.invoke(f"Summarize this 3D generation request into a snappy 2-4 word title. Respond ONLY with the title. Prompt: {user_prompt}")
        return response.content.strip().replace('"', '')
    except:
        return user_prompt[:30]

def _thread_config(session_id):
    """LangGraph config that ties a run to a chat session so the checkpointer
    can pause and later resume the *same* conversation thread."""
    return {"configurable": {"thread_id": f"chat-{session_id}"}}


def _normalize_text(message):
    """Collapse an AIMessage's content (str or content-block list) to plain text."""
    if message is None:
        return ""
    raw = message.content
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(
            block.get("text", "")
            for block in raw
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(raw)


def _pending_interrupt(config):
    """If the agent paused for human approval, return a UI-friendly dict
    describing the first gated tool call; otherwise None.

    Shape (verified against HumanInTheLoopMiddleware runtime payload):
      state.interrupts[0].value = {
        "action_requests": [{"name", "args", "description"}, ...],
        "review_configs":  [{"action_name", "allowed_decisions"}, ...],
      }
    """
    state = agent.get_state(config)
    if not state.interrupts:
        return None
    value = state.interrupts[0].value
    reqs = value.get("action_requests") or []
    cfgs = value.get("review_configs") or []
    if not reqs:
        return None
    req = reqs[0]
    args = req.get("args") or {}
    decisions = cfgs[0].get("allowed_decisions") if cfgs else ["approve", "reject"]
    return {
        "type": "interrupt",
        "tool": req.get("name", ""),
        "label": _friendly_tool_label(req.get("name", "")),
        "image_path": args.get("image_path", ""),
        "args": args,
        "allowed_decisions": decisions,
    }


def _run_stream(stream_input, config):
    """Shared driver for both a fresh run and a resume. Yields the same event
    dicts as process_chat_stream and ends with EITHER an `interrupt` event (the
    agent paused for approval) or a `text` event (the run finished)."""
    f = io.StringIO()
    final_message = None
    last_status = None

    def drain_stdout():
        nonlocal last_status
        output = f.getvalue()
        if not output:
            return
        for line in output.strip().split('\n'):
            if not line:
                continue
            # Pick out Modal call IDs even if we drop the line for the user
            call_id_match = re.search(r'job=(fc-[a-zA-Z0-9]+)', line)
            if call_id_match:
                yield {"type": "call_id", "content": call_id_match.group(1)}
            friendly = _friendly_status(line)
            if friendly and friendly != last_status:
                last_status = friendly
                yield {"type": "status", "content": friendly}
        f.truncate(0)
        f.seek(0)

    with contextlib.redirect_stdout(_TeeStdout(f, sys.__stdout__)):
        for chunk in agent.stream(stream_input, config, stream_mode="updates"):
            # 1. Drain any stdout prints into friendly status events
            for evt in drain_stdout():
                yield evt

            # 2. Inspect the chunk for tool calls + track the latest message
            for node_name, node_data in chunk.items():
                if not isinstance(node_data, dict):
                    continue
                if "messages" not in node_data or not node_data["messages"]:
                    continue
                last_msg = node_data["messages"][-1]
                final_message = last_msg  # keep updating; loop ends with the real final

                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        label = _friendly_tool_label(tc["name"])
                        status = f"{label}..."
                        if status != last_status:
                            last_status = status
                            yield {"type": "status", "content": status}

    # 3. Flush any trailing stdout
    for evt in drain_stdout():
        yield evt

    # 4. Did the agent pause for human approval? If so, surface the gate
    #    instead of a final answer — the run is suspended in the checkpointer.
    interrupt = _pending_interrupt(config)
    if interrupt is not None:
        yield interrupt
        return

    # 5. Otherwise the run finished: emit the normalized final text.
    yield {"type": "text", "content": _normalize_text(final_message)}


def process_chat_stream(user_input, chat_history_list, user_image_url=None,
                        session_id=None):
    """
    Generator that yields dictionaries:
      {"type": "call_id",   "content": "fc-..."}
      {"type": "status",    "content": "..."}
      {"type": "interrupt", "tool": ..., "image_path": ..., "allowed_decisions": [...]}
      {"type": "text",      "content": "final response string"}

    session_id ties the run to a checkpointer thread so an approval gate can be
    resumed later via resume_chat_stream(session_id, decisions).
    """
    messages = []
    for msg in chat_history_list:
        role = "user" if msg["role"] == "user" else "assistant"
        content = msg["content"]
        # History messages carry their image two ways: the local-path tag is
        # already baked into the text (see views.py) so tools can re-open the
        # file, and we re-attach the image_url here so the model can see it.
        if msg.get("image"):
            content = [
                {"type": "text", "text": content},
                {"type": "image_url", "image_url": msg["image"]}
            ]
        messages.append({"role": role, "content": content})

    # Current message
    if user_image_url:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": user_image_url}
            ]
        })
    else:
        messages.append({"role": "user", "content": user_input})

    try:
        yield from _run_stream({"messages": messages}, _thread_config(session_id))
    except Exception as e:
        yield {"type": "status", "content": f"Error: {str(e)}"}
        yield {"type": "text", "content": f"Agent error: {str(e)}"}


def resume_chat_stream(session_id, decisions):
    """Resume a run paused at an approval gate.

    decisions: list of decision dicts, one per interrupted tool call, e.g.
      [{"type": "approve"}]
      [{"type": "edit", "edited_action": {"name": tool, "args": {...}}}]
      [{"type": "reject", "message": "make the background white"}]
    Yields the same event types as process_chat_stream (and may pause again).
    """
    from langgraph.types import Command
    try:
        yield from _run_stream(
            Command(resume={"decisions": decisions}),
            _thread_config(session_id),
        )
    except Exception as e:
        yield {"type": "status", "content": f"Error: {str(e)}"}
        yield {"type": "text", "content": f"Agent error: {str(e)}"}


def peek_interrupt(session_id):
    """Return the approval-gate dict if this session's run is currently paused at
    the human-approval gate, else None.

    Unlike process/resume, this does NOT advance the run — it just reads the
    persisted checkpoint. Used to re-render the gate after a browser refresh.
    """
    return _pending_interrupt(_thread_config(session_id))