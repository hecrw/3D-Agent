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
    render_mesh_views,
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

class VerboseCapture(io.IOBase):
    def __init__(self, callback):
        self.callback = callback
    def write(self, b):
        line = b.strip()
        if line:
            self.callback(line)
        return len(b)

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
    """Restyle any photo or image into the clean Objaverse dataset style."""
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
    """Run PartCrafter: decompose an object image into N articulated parts."""
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
    """Render 6 axis-aligned views of a 3D mesh as PNGs."""
    out_dir = OUT / f"views_{int(time.time())}"
    paths = render_mesh_views(mesh_path=mesh_path, out_dir=out_dir)
    lines = [f"Rendered views to {out_dir}:"]
    lines += [f"- {name}: {p}" for name, p in paths.items()]
    return "\n".join(lines)


 
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
    tool_render_mesh_views,
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


def process_chat_stream(user_input, chat_history_list):
    """
    Generator that yields dictionaries:
    {"type": "status", "content": "..."}
    {"type": "text", "content": "..."}
    """
    messages = []
    for msg in chat_history_list:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_input})
    
    try:
        # We'll use a custom stream to capture prints from tools
        def on_print(line):
            # Filtering out some noise
            if any(x in line for x in ["[gemini]", "[trellis", "[hunyuan", "[tavily", "[download", "pending...", "job="]):
                yield {"type": "status", "content": line}

        # Using stream() to capture tool calls and progress
        with contextlib.redirect_stdout(VerboseCapture(lambda x: None)): # Just to initialize
            # This is complex because we want to yield while the stream is running.
            # We'll use a simpler approach: wrap the agent call and use a custom tool executor if possible,
            # but for now, let's just capture the stdout of the whole process.
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                for chunk in agent.stream({"messages": messages}, stream_mode="values"):
                    # Check if new lines were printed
                    output = f.getvalue()
                    if output:
                        lines = output.strip().split('\n')
                        for line in lines:
                            yield {"type": "status", "content": line}
                        f.truncate(0)
                        f.seek(0)

                    if "messages" in chunk:
                        last_msg = chunk["messages"][-1]
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                tool_name = tc["name"].replace("tool_", "").replace("_", " ")
                                yield {"type": "status", "content": f"Initializing {tool_name}..."}
        
        # Finally, get the full result
        result = agent.invoke({"messages": messages})
        final_text = result["messages"][-1].content
        yield {"type": "text", "content": final_text}

    except Exception as e:
        yield {"type": "text", "content": f"Agent error: {str(e)}"}

def generate_chat_title(user_prompt):
    """Summarizes a user prompt into a 2-4 word snappy title."""
    try:
        response = llm.invoke(f"Summarize this 3D generation request into a snappy 2-4 word title. Respond ONLY with the title. Prompt: {user_prompt}")
        return response.content.strip().replace('"', '')
    except:
        return user_prompt[:30]

# Keep the old one for compatibility if needed, but point it to the stream
def process_chat(user_input, chat_history_list):
    final_text = ""
    for chunk in process_chat_stream(user_input, chat_history_list):
        if chunk["type"] == "text":
            final_text = chunk["content"]
    return final_text
