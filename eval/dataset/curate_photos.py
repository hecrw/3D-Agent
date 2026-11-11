#!/usr/bin/env python
"""Automatically select one openly licensed REAL photo per object (no human picking).

    .venv/bin/python eval/dataset/curate_photos.py

Fixed, reproducible rule:
  1. search Wikimedia Commons for the object name (bitmap files only);
  2. walk the results in Commons' search order, keeping only files under
     CC0 / public domain / CC BY / CC BY-SA, at least 500 px on the short side;
  3. accept the FIRST candidate that CLIP (ViT-L/14, the eval's own CLIP)
     classifies as a real photo of the whole object: among the texts below,
     "a photo of a {object}" must be the most likely, with probability >= 0.5.
         a photo of a {object}            <- accept
         a drawing of a {object}
         a painting of a {object}
         a 3D render of a {object}
         a close-up photo of part of a {object}
         a photo of text or a document
         a photo of something else
The chosen 960 px Commons thumbnail is saved as images/<file>; its source,
author, licence, search rank and CLIP probability go to images/manifest.csv,
and images/contact_sheet.jpg shows all 50 for inspection. Resumable: objects
already in the manifest are skipped.
"""
from __future__ import annotations

import csv
import html
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

CAPTIONS = HERE / "captions.csv"
IMAGES = HERE / "images"
MANIFEST = IMAGES / "manifest.csv"
API = "https://commons.wikimedia.org/w/api.php"
UA = "3D-Agent-eval-photo-curation/1.0 (academic research; one-off script)"
OK_LICENSE = re.compile(r"^(cc0|cc[- ]by(-sa)?([- ]\d(\.\d)?)?|public domain|pd\b|pd-)", re.I)
THRESHOLD, PAGE, MAX_PAGES = 0.5, 40, 3
TEXTS = ["a photo of a {o}", "a drawing of a {o}", "a painting of a {o}", "a 3D render of a {o}",
         "a close-up photo of part of a {o}", "a photo of text or a document",
         "a photo of something else"]
FIELDS = ["filename", "caption", "object", "query", "search_rank", "clip_p_photo",
          "file_page", "author", "license", "license_url", "original_url", "thumb_url"]


def _get(url: str, **params) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            # 429 = Wikimedia rate limit: back off properly
            time.sleep(30 * (attempt + 1) if "429" in str(e) else 2 * (attempt + 1))
    return b""


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def search(query: str, offset: int) -> list[dict]:
    data = json.loads(_get(API, action="query", format="json", generator="search",
                           gsrnamespace=6, gsrsearch=f"{query} filetype:bitmap",
                           gsrlimit=PAGE, gsroffset=offset, prop="imageinfo",
                           iiprop="url|size|mime|extmetadata",
                           iiurlwidth=960))  # a Wikimedia standard thumbnail size
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
            "file_page": ii.get("descriptionurl", ""), "thumb_url": ii.get("thumburl") or ii.get("url"),
            "original_url": ii.get("url"), "license": lic,
            "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            "author": _strip_html(meta.get("Artist", {}).get("value", ""))[:120],
        })
    return out


class Clip:
    def __init__(self):
        import torch
        import tools
        self.torch = torch
        self.model, self.proc = tools._load()
        self.dev = tools._DEVICE

    def p_photo(self, img, obj: str) -> tuple[float, bool]:
        texts = [t.format(o=obj) for t in TEXTS]
        inp = self.proc(text=texts, images=img, return_tensors="pt", padding=True).to(self.dev)
        with self.torch.no_grad():
            probs = self.model(**inp).logits_per_image.softmax(-1)[0].tolist()
        return probs[0], probs[0] == max(probs) and probs[0] >= THRESHOLD


def main() -> None:
    from PIL import Image
    IMAGES.mkdir(exist_ok=True)
    prompts = list(csv.DictReader(open(CAPTIONS)))
    done = {r["filename"]: r for r in csv.DictReader(open(MANIFEST))} if MANIFEST.exists() else {}
    clip = Clip()
    failed = []
    for p in prompts:
        if p["filename"] in done and (IMAGES / p["filename"]).exists():
            continue
        obj = p["object"]
        query = obj.removeprefix("pair of ")
        print(f"[{p['filename']}] {query!r}", flush=True)
        pick, rank = None, 0
        for page in range(MAX_PAGES):
            try:
                cands = search(query, page * PAGE)
            except Exception as e:  # noqa: BLE001
                print(f"   search failed: {e}")
                break
            for c in cands:
                rank += 1
                time.sleep(1.0)  # stay polite: Wikimedia rate-limits bursts
                try:
                    img = Image.open(io.BytesIO(_get(c["thumb_url"]))).convert("RGB")
                except Exception:  # noqa: BLE001
                    continue
                prob, ok = clip.p_photo(img, obj)
                if ok:
                    pick = (c, img, prob)
                    break
            if pick or not cands:
                break
        if not pick:
            print("   no candidate passed")
            failed.append(p["filename"])
            continue
        c, img, prob = pick
        img.save(IMAGES / p["filename"], quality=95)
        done[p["filename"]] = {"filename": p["filename"], "caption": p["caption"], "object": obj,
                               "query": query, "search_rank": rank, "clip_p_photo": f"{prob:.3f}",
                               **{k: c[k] for k in ("file_page", "author", "license", "license_url",
                                                   "original_url", "thumb_url")}}
        print(f"   picked rank {rank} (p_photo={prob:.2f}, {c['license']})", flush=True)
        with open(MANIFEST, "w", newline="") as fh:  # save after every pick (resumable)
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(done[q["filename"]] for q in prompts if q["filename"] in done)

    # contact sheet for inspection (NOT for choosing)
    W = 200
    sheet = Image.new("RGB", (W * 10, W * 5), "white")
    for k, p in enumerate(prompts):
        f = IMAGES / p["filename"]
        if f.exists():
            im = Image.open(f).convert("RGB")
            im.thumbnail((W - 6, W - 6))
            sheet.paste(im, ((k % 10) * W + (W - im.width) // 2, (k // 10) * W + (W - im.height) // 2))
    sheet.save(IMAGES / "contact_sheet.jpg", quality=85)
    print(f"\n{len(done)}/{len(prompts)} photos -> {IMAGES}  (contact_sheet.jpg)")
    if failed:
        print("no photo passed the rule for: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
