#!/usr/bin/env python
"""Build an HTML contact sheet to eyeball which restyle axes are actually live.

    .venv/bin/python eval/contact_sheet.py eval/results_pilot.csv

For each prompt, lays out one row per condition (raw, all_on, each loo_*) showing
the restyled INPUT image the backbone actually received, plus the front mesh view
and the metric scores. If two conditions look identical, that axis isn't doing
anything — no metric will detect a difference there.

Writes eval/contact_sheet.html. Open it in a browser.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_HTML = REPO_ROOT / "eval" / "contact_sheet.html"

COND_ORDER = ["raw", "all_on"]  # loo_* appended sorted


def _front_view(mesh_path: str) -> str:
    """Front render path for a mesh, if it exists (rendered as <stem>.views/front.png)."""
    if not mesh_path:
        return ""
    front = Path(mesh_path).with_suffix(".views") / "front.png"
    return str(front) if front.exists() else ""


def _rel(path: str) -> str:
    """Path relative to the HTML file's directory, for the <img src>."""
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(OUT_HTML.parent.resolve()))
    except ValueError:
        # outside eval/ — use a file:// absolute URI so the browser still loads it
        return Path(path).resolve().as_uri()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: contact_sheet.py <results.csv>")
    rows = []
    with open(sys.argv[1], newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") == "ok"]
    if not rows:
        sys.exit("no successful rows to render")

    by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
    conds = set()
    for r in rows:
        by_prompt[r["prompt"]][r["condition"]] = r
        conds.add(r["condition"])

    cond_order = (COND_ORDER
                  + sorted(c for c in conds if c.startswith("loo_")))

    parts = ["""<!doctype html><meta charset="utf-8">
<title>Restyle ablation contact sheet</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 24px;
         background:#111; color:#eee; }
  h2 { margin-top:40px; border-bottom:1px solid #444; padding-bottom:6px; }
  .row { display:flex; align-items:center; gap:16px; padding:8px 0;
         border-bottom:1px solid #222; }
  .cond { width:160px; font-weight:600; font-family:monospace; }
  .raw  { color:#f88; }
  .all_on { color:#8f8; }
  img { width:140px; height:140px; object-fit:contain; background:#000;
        border:1px solid #333; }
  .scores { font-family:monospace; font-size:13px; color:#bbb; min-width:220px; }
  .lbl { font-size:11px; color:#888; text-align:center; width:140px; }
  .pair { display:flex; flex-direction:column; gap:3px; }
  .hint { color:#888; font-size:13px; margin-bottom:20px; }
</style>
<h1>Restyle ablation — which axes are live?</h1>
<p class="hint">For each prompt: compare the <b>restyled input</b> across rows.
If <code>all_on</code> and a <code>loo_*</code> row look identical, that axis
isn't changing the input — no metric can detect an effect there.</p>
"""]

    for prompt in sorted(by_prompt):
        parts.append(f"<h2>{prompt}</h2>")
        for cond in cond_order:
            r = by_prompt[prompt].get(cond)
            if not r:
                continue
            restyled = _rel(r.get("restyled_path", ""))
            front = _rel(_front_view(r.get("mesh_path", "")))
            cls = cond if cond in ("raw", "all_on") else ""
            restyled_html = (
                f'<div class="pair"><img src="{restyled}"><div class="lbl">restyled input</div></div>'
                if restyled else
                '<div class="pair"><div class="lbl">(no restyle — raw)</div></div>'
            )
            front_html = (
                f'<div class="pair"><img src="{front}"><div class="lbl">mesh front</div></div>'
                if front else ""
            )
            scores = (f"CLIP {r.get('clip_mean','—')}  "
                      f"Gen3D {r.get('gen3deval','—')}  "
                      f"ULIP {r.get('ulip_mean','—')}")
            parts.append(
                f'<div class="row">'
                f'<div class="cond {cls}">{cond}</div>'
                f'{restyled_html}{front_html}'
                f'<div class="scores">{scores}</div>'
                f'</div>'
            )

    OUT_HTML.write_text("\n".join(parts))
    print(f"wrote {OUT_HTML}")
    print(f"open it with:  open {OUT_HTML}")


if __name__ == "__main__":
    main()
