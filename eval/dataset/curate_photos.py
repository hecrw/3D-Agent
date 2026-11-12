#!/usr/bin/env python
"""Select one openly licensed REAL photo per object, by fixed criteria.

    .venv/bin/python eval/dataset/curate_photos.py fetch          # candidates + review sheets
    .venv/bin/python eval/dataset/curate_photos.py apply picks.csv  # copy picks + manifest

Candidates come from Wikimedia Commons (only CC0 / public domain / CC BY /
CC BY-SA files, >= 500 px on the short side), ~8 per object in Commons'
search order. One photo per object is then chosen by a curator against fixed
criteria, before any 3D generation:
  1. a real photograph (not a drawing, painting, render, scan or map);
  2. the whole object is visible (not cut off, not an extreme close-up);
  3. it is the named object;
  4. the object is the main subject (a normal everyday background is fine —
     that is what the restyle is meant to handle);
  5. no watermark or overlaid text.
The chosen 960 px Commons thumbnail is saved as images/<file>; source page,
author, licence and candidate number go to images/manifest.csv.
"""
from __future__ import annotations

import csv
import html
import io
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
OK_LICENSE = re.compile(r"^(cc0|cc[- ]by(-sa)?([- ]\d(\.\d)?)?|public domain|pd\b|pd-)", re.I)
PER_OBJECT = 8
# Search terms where the bare object name is ambiguous on Commons.
QUERY = {
    "ring": "gold ring jewelry", "flag": "flag waving pole", "kite": "kite toy",
    "hat": "fedora hat", "map": "paper road map", "doll": "doll toy",
    "car": "car parked", "teapot": "teapot", "wreath": "christmas wreath door",
    "carnival mask": "venetian mask", "flamingo lawn ornament": "plastic flamingo",
    "candle holder": "candlestick brass", "blanket": "folded blanket",
    "scarf": "silk scarf", "pair of shoes": "pair of leather shoes",
    "pair of swimming goggles": "swimming goggles", "wooden chest": "wooden chest antique",
    "chess set": "chess set board pieces", "rocking horse": "wooden rocking horse",
    "soda can": "soda can", "mug": "coffee mug", "letter opener": "letter opener",
    "pendant necklace": "pendant necklace", "brooch": "brooch jewelry",
    "electric guitar": "electric guitar", "grand piano": "grand piano",
    "potted orchid": "orchid in pot", "bicycle": "bicycle parked",
}
FIELDS = ["filename", "caption", "object", "query", "candidate", "file_page", "author",
          "license", "license_url", "thumb_url", "original_url"]


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
            time.sleep(30 * (attempt + 1) if "429" in str(e) else 2 * (attempt + 1))
    return b""


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def search(query: str) -> list[dict]:
    data = json.loads(_get(API, action="query", format="json", generator="search",
                           gsrnamespace=6, gsrsearch=f"{query} filetype:bitmap", gsrlimit=50,
                           prop="imageinfo", iiprop="url|size|mime|extmetadata",
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
        out.append({"file_page": ii.get("descriptionurl", ""),
                    "thumb_url": ii.get("thumburl") or ii.get("url"),
                    "original_url": ii.get("url"), "license": lic,
                    "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                    "author": _strip_html(meta.get("Artist", {}).get("value", ""))[:120]})
    return out


def prompts() -> list[dict]:
    return list(csv.DictReader(open(CAPTIONS)))


def fetch() -> None:
    from PIL import Image
    for p in prompts():
        stem = Path(p["filename"]).stem
        d = CAND / stem
        if (d / "candidates.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        q = QUERY.get(p["object"], p["object"])
        print(f"[{stem}] {q!r}", flush=True)
        kept = []
        for c in search(q):
            if len(kept) == PER_OBJECT:
                break
            time.sleep(1.0)  # Wikimedia rate-limits bursts
            try:
                img = Image.open(io.BytesIO(_get(c["thumb_url"]))).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            c["file"] = f"c{len(kept) + 1}.jpg"
            img.save(d / c["file"], quality=92)
            kept.append(c)
        (d / "candidates.json").write_text(json.dumps({"query": q, "candidates": kept}, indent=1))
        print(f"   {len(kept)} candidates", flush=True)
    sheets()


def sheets(per_sheet: int = 5) -> None:
    """Review sheets: one row per object, candidates numbered c1..c8."""
    from PIL import Image, ImageDraw
    ps = prompts()
    T, LBL = 230, 210
    for s in range(0, len(ps), per_sheet):
        chunk = ps[s:s + per_sheet]
        sheet = Image.new("RGB", (LBL + T * PER_OBJECT, T * len(chunk)), "white")
        d = ImageDraw.Draw(sheet)
        for r, p in enumerate(chunk):
            stem = Path(p["filename"]).stem
            d.text((6, r * T + T // 2 - 8), stem, fill="black")
            meta = CAND / stem / "candidates.json"
            cands = json.loads(meta.read_text())["candidates"] if meta.exists() else []
            for c in cands:
                k = int(c["file"][1:-4])
                im = Image.open(CAND / stem / c["file"])
                im.thumbnail((T - 8, T - 26))
                x, y = LBL + (k - 1) * T, r * T
                sheet.paste(im, (x + (T - im.width) // 2, y + 22))
                d.text((x + 4, y + 4), f"c{k}", fill="red")
        sheet.save(CAND / f"sheet_{s // per_sheet + 1:02d}.jpg", quality=85)
    print(f"sheets -> {CAND}/sheet_*.jpg")


def apply(picks_path: Path) -> None:
    picks = {r["stem"]: r["candidate"].strip() for r in csv.DictReader(open(picks_path))}
    IMAGES.mkdir(exist_ok=True)
    rows, missing = [], []
    for p in prompts():
        stem = Path(p["filename"]).stem
        k = picks.get(stem, "")
        if not k:
            missing.append(stem)
            continue
        meta = json.loads((CAND / stem / "candidates.json").read_text())
        c = next(c for c in meta["candidates"] if c["file"] == f"c{k}.jpg")
        shutil.copyfile(CAND / stem / c["file"], IMAGES / p["filename"])
        rows.append({"filename": p["filename"], "caption": p["caption"], "object": p["object"],
                     "query": meta["query"], "candidate": f"c{k}",
                     **{f: c[f] for f in ("file_page", "author", "license", "license_url",
                                          "thumb_url", "original_url")}})
    with open(IMAGES / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} photos -> {IMAGES}")
    if missing:
        sys.exit("no pick for: " + ", ".join(missing))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "fetch":
        fetch()
    elif len(sys.argv) == 3 and sys.argv[1] == "apply":
        apply(Path(sys.argv[2]))
    elif len(sys.argv) >= 2 and sys.argv[1] == "sheets":
        sheets()
    else:
        sys.exit(__doc__)
