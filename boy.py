import os
import sys
import time
import requests
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.tools import tool
from duckduckgo_search import DDGS
from PIL import Image

load_dotenv()

llm = ChatOllama(model="qwen3.5:4b")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")

# Ensure output directories exist
os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Add Hunyuan3D-2 to path so we can import hy3dgen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Hunyuan3D-2"))


@tool
def langsearch_websearch_tool(query: str, count: int = 10) -> str:
    """
    Perform web search using LangSearch Web Search API.

    Parameters:
    - query: Search keywords
    - count: Number of search results to return

    Returns:
    - Detailed information of search results, including web page title, web page URL, web page content, web page publication time, etc.
    """

    url = "https://api.langsearch.com/v1/web-search"
    headers = {
        "Authorization": f"Bearer {LANGSEARCH_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": count,
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        json_response = response.json()
        try:
            if json_response["code"] != 200 or not json_response["data"]:
                return f"Search API request failed, reason: {response.msg or 'Unknown error'}"

            webpages = json_response["data"]["webPages"]["value"]
            if not webpages:
                return "No relevant results found."
            formatted_results = ""
            for idx, page in enumerate(webpages, start=1):
                formatted_results += (
                    f"Citation: {idx}\n"
                    f"Title: {page['name']}\n"
                    f"URL: {page['url']}\n"
                    f"Content: {page['summary']}\n"
                )
            return formatted_results.strip()
        except Exception as e:
            return f"Search API request failed, reason: Failed to parse search results {str(e)}"
    else:
        return f"Search API request failed, status code: {response.status_code}, error message: {response.text}"


@tool
def find_and_download_image(query: str, filename: str) -> str:
    """
    Search the internet for an image matching the query, then automatically download
    the best one. Tries multiple results if downloads fail.
    This is the FIRST tool you should use — give it a search query describing the object
    you want to turn into a 3D model.

    Parameters:
    - query: Descriptive search query (e.g. 'red sports car product photo white background')
    - filename: Name for the saved file without extension (e.g. 'red_car')

    Returns:
    - SUCCESS: The local file path of the downloaded image ready for 3D generation.
    - FAILURE: An error message explaining what went wrong.
    """
    # Step 1: Search for images
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=10))
    except Exception as e:
        return f"FAILED: Image search error: {str(e)}. Try calling this tool again with a simpler query."

    if not results:
        return f"FAILED: No images found for '{query}'. Try calling this tool again with different search terms."

    # Step 2: Try downloading images, starting from the first result
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    errors = []

    for idx, result in enumerate(results):
        image_url = result.get("image", "")
        if not image_url:
            continue

        try:
            response = requests.get(
                image_url, headers=headers, timeout=15, stream=True
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type and not image_url.lower().endswith(
                (".png", ".jpg", ".jpeg", ".webp")
            ):
                errors.append(f"Result {idx+1}: not an image ({content_type})")
                continue

            ext = ".png"
            if "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            elif "webp" in content_type:
                ext = ".webp"

            filepath = os.path.join("images", f"{filename}{ext}")
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify it's a valid image and has reasonable size
            img = Image.open(filepath)
            img.verify()
            img = Image.open(filepath)  # reopen after verify
            w, h = img.size
            if w < 128 or h < 128:
                errors.append(f"Result {idx+1}: image too small ({w}x{h})")
                os.remove(filepath)
                continue

            title = result.get("title", "unknown")
            return (
                f"SUCCESS: Image downloaded to '{filepath}' "
                f"(from result {idx+1}: '{title}', size {w}x{h}). "
                f"Now use generate_3d_mesh with image_path='{filepath}' to create the 3D model."
            )

        except Exception as e:
            errors.append(f"Result {idx+1}: {str(e)}")
            continue

    error_summary = "; ".join(errors[:5])
    return (
        f"FAILED: Could not download any of the {len(results)} image results. "
        f"Errors: {error_summary}. "
        f"Try calling this tool again with a different query."
    )


@tool
def generate_3d_mesh(image_path: str, output_name: str) -> str:
    """
    Generate a textured 3D mesh from an image using Hunyuan3D-2.
    Produces a .glb file. Use this AFTER find_and_download_image has saved an image.

    Parameters:
    - image_path: Path to the input image file (e.g. 'images/red_car.png')
    - output_name: Name for the output 3D file without extension (e.g. 'red_car_3d')

    Returns:
    - SUCCESS: The path to the generated .glb file.
    - FAILURE: An error message explaining what went wrong.
    """
    # Validate input file exists
    if not os.path.isfile(image_path):
        return (
            f"FAILED: File '{image_path}' does not exist. "
            f"Use find_and_download_image first to get an image."
        )

    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from hy3dgen.rembg import BackgroundRemover

        image = Image.open(image_path).convert("RGBA")

        # Remove background for JPEGs (they don't have transparency)
        if image_path.lower().endswith((".jpg", ".jpeg")):
            rembg = BackgroundRemover()
            image = rembg(image)

        model_path = "tencent/Hunyuan3D-2"

        # Stage 1: Shape generation
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path
        )
        mesh = pipeline_shapegen(image=image)[0]

        # Stage 2: Texture generation
        pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
        mesh = pipeline_texgen(mesh, image=image)

        # Export
        output_path = os.path.join("outputs", f"{output_name}.glb")
        mesh.export(output_path)

        return f"SUCCESS: 3D mesh saved to '{output_path}'. The user can open this .glb file in any 3D viewer."
    except Exception as e:
        return (
            f"FAILED: 3D generation error: {str(e)}. "
            f"This may be a GPU/memory issue. Try again or use a simpler image."
        )


tools = [langsearch_websearch_tool, find_and_download_image, generate_3d_mesh]
llm_with_tools = llm.bind_tools(tools)
