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
)
 
import os
from django.conf import settings
from pathlib import Path

OUT = Path(settings.MEDIA_ROOT) / "3d_outputs"
OUT.mkdir(parents=True, exist_ok=True)

def _stamp(prefix: str, ext: str) -> str:
    return str(OUT / f"{prefix}_{int(time.time())}.{ext}")
 
 

@tool
def tool_web_search(query: str) -> str:
    """Search the web for facts, references, or context. Use whenever the user
    asks about something you don't know, or you need real-world details to
    inform a 3D asset (e.g. 'what does a Roman gladius look like?').
    Returns up to 5 results with titles, URLs, and snippets."""
    return web_search(query)


@tool
def tool_image_search(query: str) -> str:
    """Search the web for IMAGES of a subject. Use this when you need a
    reference image to feed into the 3D pipeline (e.g. 'medieval longsword',
    'art-deco lamp'). Returns image URLs — pick one and call tool_download_image
    to pull it locally before feeding it to tool_restyle_to_objaverse / tool_trellis2."""
    urls = image_search(query)
    if not urls:
        return "No images found."
    return "Image URLs (pick one and download with tool_download_image):\n" + \
           "\n".join(f"- {u}" for u in urls)


@tool
def tool_download_image(url: str) -> str:
    """Download an image from a URL and save it locally. Use after tool_image_search
    to pull a chosen image into your workflow, or to ingest any image URL the user
    provides. Returns the path to the saved image — pass it directly to
    tool_restyle_to_objaverse, tool_trellis2, tool_hunyuan3d2, etc."""
    out = _stamp("downloaded", "jpg")
    path = download_image(url=url, out_path=out)
    return f"Image saved: {path}"



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
]



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
    make conversation with the user, ut if his request can be 
    done using the tools you have use the tools to fullfill his request""",
)


def process_chat(user_input, chat_history_list):
    messages = chat_history_list + [{"role": "user", "content": user_input}]
    
    try:
        result = agent.invoke({"messages": messages})
        reply = result["messages"][-1].content
        return reply
    except Exception as e:
        return f"Agent error: {str(e)}"
