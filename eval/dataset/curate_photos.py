#!/usr/bin/env python
"""Pick one openly licensed REAL photo per prompt for the retrieved arm.

    .venv/bin/python eval/dataset/curate_photos.py fetch     # candidates + picker page
    open eval/dataset/candidates/picker.html                 # choose, export picks.csv
    .venv/bin/python eval/dataset/curate_photos.py apply eval/dataset/picks.csv

Candidates come from Wikimedia Commons (every file there is under a free
licence; we keep only CC0 / public domain / CC BY / CC BY-SA) and the source,
author and licence of every chosen photo go into images/manifest.csv, so the
photos can be shown in the paper with proper attribution.

Selection criteria (fixed BEFORE any results; shown on the picker page):
  1. it is a real photograph (not a drawing, painting, render or scan);
  2. the WHOLE object is visible (not cropped, not an extreme close-up);
  3. it matches the prompt (object type; attributes where possible);
  4. the object is the main subject; a real, everyday background is fine —
     that is exactly what the restyle is meant to handle;
  5. no watermark or overlaid text.
If no candidate meets all five, mark "none" and rerun fetch with --more for
that prompt (it pages further into the search results).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTIONS = HERE / "captions.csv"
CAND = HERE / "candidates"
IMAGES = HERE / "images"
API = "https://commons.wikimedia.org/w/api.php"
UA = "3D-Agent-eval-photo-curation/1.0 (academic research; one-off script)"
PER_PROMPT = 8
OK_LICENSE = re.compile(r"^(cc0|cc[- ]by(-sa)?([- ]\d(\.\d)?)?|public domain|pd\b|pd-)", re.I)

# Short Commons search queries, keyed by T3Bench line number.
QUERIES = {
    4: "golden retriever bow tie", 6: "polka dot sneakers", 8: "red fire hydrant",
    9: "saxophone", 10: "leather bound book gilt", 17: "diamond ring box",
    18: "antique porcelain doll", 19: "chameleon on branch", 20: "brass pocket watch",
    21: "floral porcelain teapot", 25: "green metallic beetle", 27: "velvet cushion gold embroidery",
    28: "porcelain rabbit figurine", 29: "black electric guitar", 33: "antique map",
    36: "old book yellowed pages", 38: "straw sun hat", 40: "antique perfume bottle",
    41: "mahogany grand piano", 42: "brass trumpet", 43: "white wedding dress",
    47: "velvet armchair", 51: "paper airplane", 53: "wooden chess set",
    54: "red kite flying", 55: "opal stone", 57: "lace doily",
    58: "wooden walking cane", 59: "crystal chandelier", 60: "red flannel shirt",
    62: "old typewriter", 65: "electric concept car", 70: "red apple",
    71: "leather briefcase", 73: "smartphone", 74: "beach ball",
    75: "rusty abandoned bicycle", 76: "orange cat", 79: "wooden baseball bat",
    81: "old straw hat", 82: "worn soccer ball", 83: "blue jeans",
    84: "old teddy bear", 85: "white coffee mug", 91: "purple orchid pot",
    93: "diamond tiara", 96: "quill pen", 98: "ornate hand mirror",
    99: "green watering can", 100: "vintage radio",
}


def _get(url: str, **params) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return b""


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def search(query: str, offset: int) -> list[dict]:
    data = json.loads(_get(API, action="query", format="json", generator="search",
                           gsrnamespace=6, gsrsearch=f"{query} filetype:bitmap",
                           gsrlimit=40, gsroffset=offset, prop="imageinfo",
                           iiprop="url|size|mime|extmetadata", iiurlwidth=1024))
    out = []
    for page in sorted(data.get("query", {}).get("pages", {}).values(),
                       key=lambda p: p.get("index", 0)):
        ii = (page.get("imageinfo") or [{}])[0]
        meta = ii.get("extmetadata", {})
        lic = _strip_html(meta.get("LicenseShortName", {}).get("value", ""))
        if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
            continue
        if min(ii.get("width", 0), ii.get("height", 0)) < 500 or not OK_LICENSE.match(lic):
            continue
        out.append({
            "title": page["title"], "file_page": ii.get("descriptionurl", ""),
            "thumb": ii.get("thumburl") or ii.get("url"), "original": ii.get("url"),
            "width": ii.get("width"), "height": ii.get("height"), "license": lic,
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            "author": _strip_html(meta.get("Artist", {}).get("value", ""))[:120],
        })
    return out


def load_prompts() -> list[dict]:
    return list(csv.DictReader(open(CAPTIONS)))


def fetch(more: list[str]) -> None:
    prompts = load_prompts()
    for p in prompts:
        stem = Path(p["filename"]).stem
        d = CAND / stem
        meta_f = d / "candidates.json"
        if meta_f.exists() and stem not in more:
            continue
        line = int(p["t3bench_line"])
        q = QUERIES.get(line) or p["caption"]
        offset = 0
        if meta_f.exists():  # --more: page past what we already showed
            offset = json.loads(meta_f.read_text()).get("next_offset", 40)
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        print(f"[{stem}] {p['caption']!r}  (query: {q!r}, offset {offset})")
        try:
            cands = search(q, offset)[:PER_PROMPT]
        except Exception as e:  # noqa: BLE001
            print(f"   search failed: {e}")
            cands = []
        kept = []
        for i, c in enumerate(cands, 1):
            try:
                (d / f"c{i:02d}.jpg").write_bytes(_get(c["thumb"]))
                c["file"] = f"c{i:02d}.jpg"
                kept.append(c)
            except Exception as e:  # noqa: BLE001
                print(f"   download failed: {c['title']}: {e}")
        meta_f.write_text(json.dumps({"query": q, "next_offset": offset + 40,
                                      "candidates": kept}, indent=1))
        print(f"   {len(kept)} candidates")
    build_picker(prompts)


def build_picker(prompts: list[dict]) -> None:
    rows = []
    for p in prompts:
        stem = Path(p["filename"]).stem
        meta_f = CAND / stem / "candidates.json"
        cands = json.loads(meta_f.read_text())["candidates"] if meta_f.exists() else []
        cells = [f'<label class="c none"><input type="radio" name="{stem}" value="none">'
                 f'<span>none fit</span></label>']
        for c in cands:
            n = c["file"][1:3]
            cells.append(
                f'<label class="c"><input type="radio" name="{stem}" value="{n}">'
                f'<img loading="lazy" src="{stem}/{c["file"]}"><span>#{n} · '
                f'{html.escape(c["license"])} · <a href="{html.escape(c["file_page"])}" '
                f'target="_blank">source</a></span></label>')
        rows.append(f'<section><h2>{stem} — {html.escape(p["caption"])}</h2>'
                    f'<div class="row">{"".join(cells)}</div></section>')
    page = f"""<!doctype html><meta charset="utf-8"><title>Photo picker</title>
<style>
:root{{--bg:#fff;--fg:#111;--mut:#666;--hi:#2563eb;--card:#f4f4f5}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#999;--card:#1f1f23}}}}
body{{font:14px system-ui;background:var(--bg);color:var(--fg);margin:0 16px 80px}}
h1{{font-size:20px}} h2{{font-size:15px;margin:22px 0 8px}}
ol{{color:var(--mut)}} .row{{display:flex;gap:10px;overflow-x:auto;padding-bottom:6px}}
.c{{flex:0 0 220px;background:var(--card);border-radius:8px;padding:6px;cursor:pointer;
   border:3px solid transparent;display:flex;flex-direction:column;gap:4px}}
.c.none{{flex-basis:90px;justify-content:center;align-items:center}}
.c img{{width:100%;height:180px;object-fit:contain;background:#0001}}
.c span{{font-size:11px;color:var(--mut)}} .c input{{display:none}}
.c:has(input:checked){{border-color:var(--hi)}}
#bar{{position:fixed;bottom:0;left:0;right:0;padding:10px 16px;background:var(--card);
     display:flex;gap:12px;align-items:center}}
button{{font:inherit;padding:6px 14px}}
</style>
<h1>Pick one photo per prompt</h1>
<ol><li>real photograph (not a drawing, painting, render or scan)</li>
<li>the WHOLE object is visible</li><li>matches the prompt</li>
<li>object is the main subject — an everyday background is fine</li>
<li>no watermark or overlaid text</li></ol>
{"".join(rows)}
<div id="bar"><button id="dl">Download picks.csv</button><span id="n"></span></div>
<script>
const stems={json.dumps([Path(p['filename']).stem for p in prompts])};
function count(){{const k=stems.filter(s=>document.querySelector(`input[name="${{s}}"]:checked`)).length;
 document.getElementById('n').textContent=`${{k}} / ${{stems.length}} decided`;}}
document.addEventListener('change',count);count();
document.getElementById('dl').onclick=()=>{{
 const lines=['stem,choice'].concat(stems.map(s=>{{const e=document.querySelector(`input[name="${{s}}"]:checked`);
  return `${{s}},${{e?e.value:''}}`;}}));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')+'\\n'],{{type:'text/csv'}}));
 a.download='picks.csv';a.click();}};
</script>"""
    (CAND / "picker.html").write_text(page)
    print(f"picker -> {CAND / 'picker.html'}")


def apply(picks_path: Path) -> None:
    prompts = {Path(p["filename"]).stem: p for p in load_prompts()}
    picks = {r["stem"]: r["choice"].strip() for r in csv.DictReader(open(picks_path))}
    IMAGES.mkdir(exist_ok=True)
    manifest, missing = [], []
    for stem, p in prompts.items():
        choice = picks.get(stem, "")
        if not choice or choice == "none":
            missing.append(f"{stem} ({choice or 'no pick'})")
            continue
        meta = json.loads((CAND / stem / "candidates.json").read_text())
        c = next(c for c in meta["candidates"] if c["file"] == f"c{int(choice):02d}.jpg")
        # full-resolution original (thumbnail was only for picking)
        data = _get(c["original"])
        from io import BytesIO
        from PIL import Image
        Image.open(BytesIO(data)).convert("RGB").save(IMAGES / p["filename"], quality=95)
        manifest.append({"filename": p["filename"], "caption": p["caption"],
                         "file_page": c["file_page"], "author": c["author"],
                         "license": c["license"], "license_url": c["license_url"],
                         "original_url": c["original"]})
        print(f"{stem}: #{choice} {c['license']}")
    with open(IMAGES / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]) if manifest else ["filename"])
        w.writeheader()
        w.writerows(manifest)
    print(f"\n{len(manifest)} photos -> {IMAGES}")
    if missing:
        print("still needed (rerun: curate_photos.py fetch --more STEM ...):\n  " + "\n  ".join(missing))
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--more", nargs="*", default=[], help="stems to re-search further down the results")
    a = sub.add_parser("apply")
    a.add_argument("picks", type=Path)
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.more)
    else:
        apply(args.picks)


if __name__ == "__main__":
    main()
