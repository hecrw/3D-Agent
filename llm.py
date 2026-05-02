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
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------------

from tools import (
    generate_concept_image,
    restyle_to_objaverse,
    image_to_3d,
    texture_mesh,
    partcrafter,
    threestudio_refine,
    paint3d_texture,
)
 
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
 
 
def _stamp(prefix: str, ext: str) -> str:
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
def tool_image_to_3d(image_path: str) -> str:
    """Run the TRELLIS pipeline: convert a clean image into a textured 3D GLB file.
    Input should be an Objaverse-style image (use tool_restyle_to_objaverse first if needed).
    Returns the path to the output .glb file."""
    out = _stamp("model", "glb")
    path = image_to_3d(image_path=image_path, out_path=out)
    return f"3D model saved: {path}"
 
 
@tool
def tool_texture_mesh(image_path: str, mesh_path: str) -> str:
    """Re-texture an existing mesh using TRELLIS, guided by a reference image.
    Use this when you already have a .glb/.obj mesh and want better textures.
    Returns the path to the textured .glb file."""
    out = _stamp("textured", "glb")
    path = texture_mesh(image_path=image_path, mesh_path=mesh_path, out_path=out)
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
 
 
@tool
def tool_paint3d_texture(mesh_path: str, prompt: str, ip_image_path: str = None) -> str:
    """Paint a high-res, lighting-free texture onto an untextured mesh using Paint3D.
    Faster than threestudio for texture-only work. Accepts .obj, .glb, or .ply.
    Optionally pass ip_image_path for image-guided texturing.
    Returns the path to the painted .glb file."""
    out = _stamp("painted", "glb")
    path = paint3d_texture(
        mesh_path=mesh_path,
        prompt=prompt,
        out_path=out,
        ip_image_path=ip_image_path or None,
    )
    return f"Painted model saved: {path}"
 
 
# ── exported list ─────────────────────────────────────────────────────────────
 
ALL_TOOLS = [
    tool_generate_concept_image,
    tool_restyle_to_objaverse,
    tool_image_to_3d,
    tool_texture_mesh,
    tool_partcrafter,
    tool_threestudio_refine,
    tool_paint3d_texture,
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

agent = create_react_agent(
    model=llm,
    tools=ALL_TOOLS,
    prompt="""You are a helpful assistant.
    make conversation with the user, ut if his request can be 
    done using the tools you have use the tools to fullfill his request""",
)

# ---------------------------------------------------------------------------
# 4. Chat loop
# ---------------------------------------------------------------------------

def main():
    print(" Cloud Agent (Gemini API + LangGraph)")
    print("   Model : gemini-flash-latest")
    print("   Type 'exit' to quit\n")

    chat_history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        chat_history.append({"role": "user", "content": user_input})

        try:
            result = agent.invoke({"messages": chat_history})
            reply = result["messages"][-1].content
        except Exception as e:
            reply = f"(Agent error: {e})"

        print(f"\nAgent: {reply}\n")
        chat_history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()