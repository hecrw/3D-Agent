import os
import sys
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
def search_images(query: str, max_results: int = 5) -> str:
    """
    Search for images on the internet matching a query.
    Returns a numbered list of image results with title and image URL.
    Use this to find reference images for 3D model generation.

    Parameters:
    - query: Descriptive search query for the image you want to find
    - max_results: Number of image results to return (default 5)

    Returns:
    - A numbered list of image results with their titles and direct image URLs
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))

        if not results:
            return "No images found for the given query."

        formatted = ""
        for idx, result in enumerate(results, start=1):
            formatted += (
                f"Result {idx}:\n"
                f"  Title: {result.get('title', 'No title')}\n"
                f"  Image URL: {result.get('image', 'No URL')}\n"
                f"  Source: {result.get('source', 'Unknown')}\n"
                f"  Size: {result.get('width', '?')}x{result.get('height', '?')}\n\n"
            )
        return formatted.strip()
    except Exception as e:
        return f"Image search failed: {str(e)}"


@tool
def download_image(image_url: str, filename: str) -> str:
    """
    Download an image from a URL and save it locally.
    Use this after search_images to download the chosen image.

    Parameters:
    - image_url: The direct URL of the image to download
    - filename: Name for the saved file (without extension, e.g. 'red_car')

    Returns:
    - The local file path of the downloaded image, or an error message
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(image_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type and not image_url.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        ):
            return f"URL does not point to an image. Content-Type: {content_type}"

        ext = ".png"
        if "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "webp" in content_type:
            ext = ".webp"

        filepath = os.path.join("images", f"{filename}{ext}")
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify it's a valid image
        img = Image.open(filepath)
        img.verify()

        return f"Image downloaded successfully to: {filepath}"
    except Exception as e:
        return f"Failed to download image: {str(e)}"


@tool
def generate_3d_mesh(image_path: str, output_name: str) -> str:
    """
    Generate a 3D mesh from an image using Hunyuan3D-2.
    This takes an image file and produces a textured 3D model saved as a .glb file.
    Use this after downloading an image with download_image.

    Parameters:
    - image_path: Path to the input image file (e.g. 'images/red_car.png')
    - output_name: Name for the output 3D file (without extension, e.g. 'red_car_3d')

    Returns:
    - The path to the generated .glb 3D mesh file, or an error message
    """
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from hy3dgen.rembg import BackgroundRemover

        image = Image.open(image_path).convert("RGBA")

        # Remove background if the image doesn't have transparency
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

        return f"3D mesh generated successfully and saved to: {output_path}"
    except Exception as e:
        return f"3D mesh generation failed: {str(e)}"


tools = [langsearch_websearch_tool, search_images, download_image, generate_3d_mesh]
llm_with_tools = llm.bind_tools(tools)
