import time
import re
from pathlib import Path
from langchain_core.tools import tool

from tools import (
    generate_concept_image,
    restyle_to_objaverse,
    trellis2,
    trellis2_texture,
    partcrafter,
    hunyuan3d2,
    web_search,
    image_search,
    download_image,
    render_mesh_views,
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
    path = download_image(url=url, out_path=out)
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
def tool_trellis2(image_path: str) -> str:
    """Run the TRELLIS pipeline: convert a clean image into a textured 3D GLB file."""
    out = _stamp("trellis2", "glb")
    path = trellis2(image_path=image_path, out_path=out)
    return f"3D model saved: {path}"

@tool
def tool_trellis2_texture(image_path: str, mesh_path: str) -> str:
    """Re-texture an existing mesh using TRELLIS."""
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
    tool_trellis2,
    tool_trellis2_texture,
    tool_partcrafter,
    tool_hunyuan3d2,
    tool_score_alignment,
]


# --- Friendly status translation ---

TOOL_LABELS = {
    "tool_web_search": "Searching the web",
    "tool_image_search": "Searching for reference images",
    "tool_download_image": "Downloading image",
    "tool_generate_concept_image": "Generating concept image",
    "tool_restyle_to_objaverse": "Restyling to Objaverse",
    "tool_trellis2": "Generating 3D model",
    "tool_trellis2_texture": "Re-texturing model",
    "tool_partcrafter": "Decomposing into parts",
    "tool_hunyuan3d2": "Generating 3D model",
    "tool_render_mesh_views": "Rendering views",
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
agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    system_prompt="""You are a helpful assistant.
    If the user provides an image, you will see an [Uploaded Image Local Path: ...] in the message.
    CRITICAL: Never use 'input_file_0.png' or any other generated path. 
    Use the EXACT absolute path provided in the [Uploaded Image Local Path: ...] tag for any tool that requires an 'image_path'.
    If you see an image but no local path is provided in the text, ask the user for clarification.
    Make conversation with the user, but if his request can be 
    done using the tools you have, use the tools to fulfill his request.""",
)


def generate_chat_title(user_prompt):
    """Summarizes a user prompt into a 2-4 word snappy title."""
    try:
        response = llm.invoke(f"Summarize this 3D generation request into a snappy 2-4 word title. Respond ONLY with the title. Prompt: {user_prompt}")
        return response.content.strip().replace('"', '')
    except:
        return user_prompt[:30]

def process_chat_stream(user_input, chat_history_list, user_image_url=None):
    """
    Generator that yields dictionaries:
      {"type": "call_id", "content": "fc-..."}
      {"type": "status",  "content": "..."}
      {"type": "text",    "content": "final response string"}
    """
    messages = []
    for msg in chat_history_list:
        role = "user" if msg["role"] == "user" else "assistant"
        content = msg["content"]
        # If there's an image in history, we should ideally handle it too
        # but for now let's focus on the current message.
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
            for chunk in agent.stream({"messages": messages}, stream_mode="updates"):
                # 1. Drain any stdout prints into friendly status events
                for evt in drain_stdout():
                    yield evt

                # 2. Inspect the chunk for tool calls + track the latest message
                for node_name, node_data in chunk.items():
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
        
        # 4. Normalize the final message content to a plain string
        if final_message is None:
            final_text = ""
        else:
            raw = final_message.content
            if isinstance(raw, str):
                final_text = raw
            elif isinstance(raw, list):
                final_text = "".join(
                    block.get("text", "")
                    for block in raw
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                final_text = str(raw)
        
        yield {"type": "text", "content": final_text}
    
    except Exception as e:
        yield {"type": "status", "content": f"Error: {str(e)}"}
        yield {"type": "text", "content": f"Agent error: {str(e)}"}