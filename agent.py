"""
Free local LangChain agent — compatible with langchain >= 1.0
Uses LangGraph's prebuilt ReAct agent + Ollama (no API key needed).

Setup:
    1. Install Ollama:      https://ollama.com/download
    2. Pull a model:        ollama pull llama3.2
    3. Install deps:
       pip install -U langchain langchain-ollama langgraph

Usage:
    python free_langchain_agent.py
"""
import time
from pathlib import Path
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------

from tools import (
    generate_concept_image,
    restyle_to_objaverse,
    trellis2,
    trellis2_texture,
    partcrafter,
    hunyuan3d2,
    hunyuan3d2_texture,
    threestudio_refine,
)
 
import os
from django.conf import settings
from pathlib import Path

# This tells the agent to save everything inside your Django media folder
# instead of a random folder in your root directory.
OUT = Path(settings.MEDIA_ROOT) / "3d_outputs"
OUT.mkdir(parents=True, exist_ok=True)

def _stamp(prefix: str, ext: str) -> str:
    # This stays the same, it just uses the new OUT path from above
    return str(OUT / f"{prefix}_{int(time.time())}.{ext}")
 
 
# ── Stage 1: image preparation ───────────────────────────────────────────────
 
@tool
def tool_generate_concept_image(prompt: str) -> str:
    """Generate a concept PNG image from a text description using Gemini.
    Use this as the first step when the user only has a text prompt and no image.
    Returns the path to the saved PNG file."""
    out = _stamp("concept", "png")
    path = generate_concept_image(prompt=prompt, out_path=out)
    return f"Concept image saved: {path}"
 
 
@tool
def tool_restyle_to_objaverse(image_path: str) -> str:
    """Restyle any photo or image into the clean Objaverse dataset style
    (centered object, neutral background, studio lighting).
    Always run this before feeding a real-world photo into a 3D pipeline.
    Returns the path to the restyled PNG."""
    out = _stamp("objaverse", "png")
    path = restyle_to_objaverse(image_path=image_path, out_path=out)
    return f"Restyled image saved: {path}"
 
 
# ── Stage 2: 3D generation ───────────────────────────────────────────────────
 
@tool
def tool_trellis2(image_path: str) -> str:
    """Run the TRELLIS pipeline: convert a clean image into a textured 3D GLB file.
    Input should be an Objaverse-style image (use tool_restyle_to_objaverse first if needed).
    Returns the path to the output .glb file."""
    out = _stamp("model", "glb")
    path = trellis2(image_path=image_path, out_path=out)
    return f"3D model saved: {path}"


@tool
def tool_trellis2_texture(image_path: str, mesh_path: str) -> str:
    """Re-texture an existing mesh using TRELLIS, guided by a reference image.
    Use this when you already have a .glb/.obj mesh and want better textures.
    Returns the path to the textured .glb file."""
    out = _stamp("textured", "glb")
    path = trellis2_texture(image_path=image_path, mesh_path=mesh_path, out_path=out)
    return f"Textured model saved: {path}"
 
 
@tool
def tool_partcrafter(image_path: str, num_parts: int = 3, scene: bool = False) -> str:
    """Run PartCrafter: decompose an object image into N articulated parts.
    Set scene=True to generate a full multi-object scene instead of a single asset.
    Returns the path to the output .glb file."""
    out = _stamp("parts", "glb")
    path = partcrafter(
        image_path=image_path,
        out_path=out,
        num_parts=num_parts,
        scene=scene,
    )
    return f"Part-decomposed model saved: {path}"
 
 
@tool
def tool_hunyuan3d2(image_path: str) -> str:
    """Run Hunyuan3D-2: convert a clean image into a fully textured 3D GLB file.
    Faster alternative to TRELLIS for image-to-3D with PBR textures.
    Returns the path to the output .glb file."""
    out = _stamp("hunyuan", "glb")
    path = hunyuan3d2(image_path=image_path, out_path=out)
    return f"3D model saved: {path}"


@tool
def tool_hunyuan3d2_texture(image_path: str, mesh_path: str) -> str:
    """Re-texture an existing mesh using Hunyuan3D-2 Paint, guided by a reference image.
    Accepts .glb/.obj/.ply/.stl. Use when you already have a mesh and want PBR textures.
    Returns the path to the textured .glb file."""
    out = _stamp("hunyuan_textured", "glb")
    path = hunyuan3d2_texture(image_path=image_path, mesh_path=mesh_path, out_path=out)
    return f"Textured model saved: {path}"


# ── Stage 3: refinement ───────────────────────────────────────────────────────
 
@tool
def tool_threestudio_refine(mesh_path: str, prompt: str) -> str:
    """Deeply refine a 3D mesh using threestudio SDS with a text prompt.
    This is slow (30+ minutes) but produces the highest quality results.
    Use as a final quality pass after generating the initial mesh.
    Returns the path to the refined .glb file."""
    out = _stamp("refined", "glb")
    path = threestudio_refine(mesh_path=mesh_path, prompt=prompt, out_path=out)
    return f"Refined model saved: {path}"
 
 
 
 
# ── exported list ─────────────────────────────────────────────────────────────
 
ALL_TOOLS = [
    tool_generate_concept_image,
    tool_restyle_to_objaverse,
    tool_trellis2,
    tool_trellis2_texture,
    tool_partcrafter,
    tool_hunyuan3d2,
    tool_hunyuan3d2_texture,
    tool_threestudio_refine,
]


# ---------------------------------------------------------------------------
# 2. LLM — change model= to any model you've pulled via `ollama pull <name>`
# ---------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0.2
)

# ---------------------------------------------------------------------------
# 3. Agent (LangGraph ReAct — works with langchain 1.x)
# ---------------------------------------------------------------------------
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(
    model=llm,
    tools=ALL_TOOLS,
    prompt="""You are a helpful assistant.
    make conversation with the user, ut if his request can be 
    done using the tools you have use the tools to fullfill his request""",
)

# ---------------------------------------------------------------------------
# 4. Exposed Function
# ---------------------------------------------------------------------------

def process_chat(user_input, chat_history_list):
    # chat_history_list should be a list of dicts: [{"role": "user", "content": "..."}, ...]
    messages = chat_history_list + [{"role": "user", "content": user_input}]
    
    try:
        result = agent.invoke({"messages": messages})
        reply = result["messages"][-1].content
        return reply
    except Exception as e:
        return f"Agent error: {str(e)}"