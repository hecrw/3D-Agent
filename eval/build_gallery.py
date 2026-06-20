#!/usr/bin/env python
"""Build a static browser for every generated GLB in media/3d_outputs.

Scans the folder, parses each filename into {backbone, arm, object, condition},
and writes manifest.json + gallery.html next to the GLBs. Open it through a
local web server (model-viewer fetches GLBs over HTTP, not file://):

    python eval/build_gallery.py
    cd media/3d_outputs && python -m http.server 8000
    # then open http://localhost:8000/gallery.html
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MEDIA = Path(__file__).resolve().parent.parent / "media" / "3d_outputs"
BACKBONES = {
    "trellis2": "TRELLIS.2", "hunyuan": "Hunyuan3D-2",
    "parts": "PartCrafter", "model": "model", "objaverse": "Objaverse",
}
CONDS = ["raw", "all_on", "loo_background", "loo_framing", "loo_view",
         "loo_lighting", "loo_isolation", "loo_part_visibility"]


def parse(name: str) -> dict:
    """Pull structured metadata out of a GLB filename."""
    stem = name[:-4]  # drop .glb
    backbone = next((BACKBONES[p] for p in BACKBONES if stem.startswith(p + "_") or stem == p), "other")
    body = stem
    for p in BACKBONES:
        if body.startswith(p + "_"):
            body = body[len(p) + 1:]
            break
    arm = "retrieved" if body.startswith("ret_") else "generated"
    if body.startswith("ret_"):
        body = body[4:]
    obj, cond = "(ad-hoc)", ""
    if "__" in body:
        left, right = body.split("__", 1)
        obj = left or "(ad-hoc)"
        m = re.match(r"(.+?)_\d+$", right)          # condition_<timestamp>
        cond = m.group(1) if m else right
    elif re.match(r"^[a-z]+_\d+$", body) or re.match(r"^\d+$", body):
        obj = "(ad-hoc)"                             # e.g. trellis2_1781896597
    else:
        obj = re.sub(r"_\d+$", "", body) or "(ad-hoc)"
    return {"backbone": backbone, "arm": arm, "object": obj, "condition": cond}


# External baseline GLB folders (one mesh per object, no restyle). Paths in the
# manifest are written relative to gallery.html (which lives in media/3d_outputs),
# so the page must be served from the REPO ROOT for these to resolve.
REPO = MEDIA.parent.parent
BASELINE_DIRS = {
    "TRELLIS.2": REPO / "eval" / "baseline_glbs_trellis",
    "Hunyuan3D-2": REPO / "eval" / "baseline_glbs_hunyuan",
    "PartCrafter": REPO / "eval" / "baseline_glbs_partcrafter",
}


def build_items() -> list[dict]:
    """Scan media/3d_outputs + the baseline folders and return manifest rows.

    Pure scan, no file writes — reused by serve_gallery.py to recompute the
    manifest live so newly generated meshes appear without a rebuild.
    """
    items = []
    # 1) everything our pipeline produced, in media/3d_outputs
    for g in sorted(MEDIA.glob("*.glb")):
        meta = parse(g.name)
        # "ours" = our restyle pipeline's 3D output (TRELLIS.2 backbone, a real
        # object, a known condition) — not the Hunyuan/PartCrafter baselines or
        # the ad-hoc chat tests. all_on is the canonical/headline "ours".
        ours = (meta["backbone"] == "TRELLIS.2"
                and meta["object"] != "(ad-hoc)"
                and meta["condition"] in CONDS)
        meta["ours"] = ours
        meta["final"] = ours and meta["condition"] == "all_on"
        meta["baseline"] = False
        meta.update(file=g.name, size_mb=round(g.stat().st_size / 1e6, 1))
        items.append(meta)
    # 2) external baselines from eval/baseline_glbs_* (filename = object stem)
    for backbone, d in BASELINE_DIRS.items():
        if not d.exists():
            continue
        for g in sorted(d.glob("*.glb")):
            items.append({
                "backbone": backbone, "arm": "baseline", "object": g.stem,
                "condition": "", "ours": False, "final": False, "baseline": True,
                "file": f"../../{d.relative_to(REPO).as_posix()}/{g.name}",
                "size_mb": round(g.stat().st_size / 1e6, 1),
            })
    # objects first (real stems), ad-hoc last; within object: condition order
    cond_rank = {c: i for i, c in enumerate(CONDS)}
    items.sort(key=lambda d: (d["object"] == "(ad-hoc)", d["object"], d["arm"],
                              cond_rank.get(d["condition"], 99), d["file"]))
    return items


def write_pages():
    """Write the static gallery.html + view.html (templates only; the manifest
    is data and is written/served separately)."""
    (MEDIA / "gallery.html").write_text(HTML)
    (MEDIA / "view.html").write_text(VIEW_HTML)


def main():
    items = build_items()
    write_pages()
    (MEDIA / "manifest.json").write_text(json.dumps(items, indent=0))
    objs = sorted({i["object"] for i in items if i["object"] != "(ad-hoc)"})
    nb = sum(i["baseline"] for i in items)
    print(f"-> {MEDIA/'gallery.html'}")
    print(f"   {len(items)} GLBs across {len(objs)} objects ({nb} baseline meshes)")
    print("   static manifest written. For a LIVE gallery that auto-includes new")
    print("   app generations, run instead:  python eval/serve_gallery.py")


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D-Agent — Generated Object Browser</title>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
<style>
  :root { --bg:#0f1115; --card:#1a1d24; --line:#2a2f3a; --txt:#e8eaed; --mut:#9aa0ab; --acc:#e63946; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header { position:sticky; top:0; z-index:10; background:rgba(15,17,21,.96);
           backdrop-filter:blur(8px); border-bottom:1px solid var(--line); padding:14px 20px; }
  h1 { margin:0 0 10px; font-size:18px; font-weight:700; }
  h1 span { color:var(--mut); font-weight:400; font-size:13px; }
  .bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  select,input { background:var(--card); color:var(--txt); border:1px solid var(--line);
                 border-radius:8px; padding:7px 10px; font-size:13px; }
  input[type=search] { min-width:200px; flex:1; }
  .count { color:var(--mut); margin-left:auto; white-space:nowrap; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
          gap:16px; padding:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          overflow:hidden; display:flex; flex-direction:column; position:relative; }
  .card.ours { border-color:#d8a64a; box-shadow:0 0 0 1px #d8a64a55; }
  .card.final { border-color:var(--acc); box-shadow:0 0 0 1px var(--acc); }
  .badge { position:absolute; top:8px; left:8px; z-index:2; font-size:10px; font-weight:700;
           letter-spacing:.5px; padding:3px 8px; border-radius:99px; color:#1a1d24; }
  .badge.ours  { background:#d8a64a; }
  .badge.final { background:var(--acc); color:#fff; }
  .badge.base  { background:#5a6270; color:#fff; }
  .card.base { border-color:#3a4150; }
  .viewport { width:100%; height:240px; background:
     radial-gradient(circle at 50% 35%, #262b35 0%, #14161b 80%);
     display:flex; align-items:center; justify-content:center; position:relative; }
  .viewport model-viewer { width:100%; height:100%; --poster-color:transparent; }
  .ph { color:var(--mut); font-size:34px; font-weight:700; opacity:.35;
        text-transform:uppercase; letter-spacing:1px; user-select:none; }
  .meta { padding:10px 12px; border-top:1px solid var(--line); }
  .obj { font-weight:600; }
  .tags { margin-top:6px; display:flex; flex-wrap:wrap; gap:5px; }
  .tag { font-size:11px; padding:2px 7px; border-radius:99px; background:#22272f; color:var(--mut); }
  .tag.cond { background:#3a2024; color:#ff8a95; }
  .tag.arm { background:#1e2b22; color:#7fd99a; }
  .tag.bb  { background:#202636; color:#8fb4ff; }
  .empty { padding:60px; text-align:center; color:var(--mut); }
  a.dl { color:var(--mut); font-size:11px; text-decoration:none; }
  a.dl:hover { color:var(--acc); }
  .viewport { cursor:zoom-in; }
  .open-hint { position:absolute; bottom:8px; right:10px; font-size:11px; color:var(--mut);
               background:#0008; padding:2px 7px; border-radius:6px; pointer-events:none; }
  /* fullscreen lightbox */
  #modal { position:fixed; inset:0; z-index:50; background:rgba(8,9,12,.92);
           display:none; flex-direction:column; }
  #modal.open { display:flex; }
  #modal model-viewer { flex:1; width:100%; height:100%; }
  .mbar { display:flex; align-items:center; gap:14px; padding:12px 18px;
          border-bottom:1px solid var(--line); }
  .mbar .name { font-weight:600; }
  .mbar a { color:#8fb4ff; font-size:13px; text-decoration:none; }
  .mbar a:hover { text-decoration:underline; }
  .mbtn { background:var(--acc); color:#fff; border:none; border-radius:8px;
          padding:7px 12px; font-size:13px; font-weight:600; cursor:pointer; }
  .mbtn:hover { filter:brightness(1.1); }
  .mbar .close { margin-left:auto; cursor:pointer; font-size:22px; line-height:1;
                 color:var(--mut); background:none; border:none; }
  .mbar .close:hover { color:var(--txt); }
</style>
</head>
<body>
<header>
  <h1>3D-Agent <span>— generated object browser</span></h1>
  <div class="bar">
    <input type="search" id="q" placeholder="search object / file…">
    <select id="scope">
      <option value="">everything</option>
      <option value="ours">ours only</option>
      <option value="final">ours — final (all_on)</option>
      <option value="baseline">baselines only</option>
    </select>
    <select id="obj"></select>
    <select id="cond"></select>
    <select id="arm"></select>
    <select id="bb"></select>
    <span class="count" id="count"></span>
  </div>
</header>
<div class="grid" id="grid"></div>
<div class="empty" id="empty" style="display:none">No models match these filters.</div>
<div id="modal">
  <div class="mbar">
    <span class="name" id="mname"></span>
    <button class="mbtn" id="msnap" title="save a PNG of the current camera angle">📷 snapshot view</button>
    <a id="mopen" target="_blank" rel="noopener">open in new tab ↗</a>
    <a id="mdl" download>download .glb ↓</a>
    <button class="close" id="mclose" title="close (Esc)">✕</button>
  </div>
  <div id="mbody"></div>
</div>
<script>
let DATA = [];
const $ = s => document.querySelector(s);
const grid = $("#grid");

function opts(sel, vals, label) {
  sel.innerHTML = `<option value="">${label} (all)</option>` +
    vals.map(v => `<option>${v}</option>`).join("");
}
function uniq(key) { return [...new Set(DATA.map(d => d[key]).filter(Boolean))].sort(); }

// Mount a model-viewer only while a card is near the viewport; tear it down
// when it scrolls away so we never hold more than a handful of WebGL contexts.
function mount(vp) {
  if (vp.dataset.mounted) return;
  vp.dataset.mounted = "1";
  vp.innerHTML =
    `<model-viewer src="${vp.dataset.src}" camera-controls touch-action="pan-y"
       reveal="auto" shadow-intensity="1" exposure="1.1"
       camera-orbit="35deg 70deg auto" interaction-prompt="none"></model-viewer>`;
}
function unmount(vp) {
  if (!vp.dataset.mounted) return;
  const mv = vp.querySelector("model-viewer");
  if (mv) { mv.src = ""; mv.remove(); }            // releases the GL context
  vp.innerHTML = `<div class="ph">${vp.dataset.label}</div>`;
  delete vp.dataset.mounted;
}
const io = new IntersectionObserver((entries) => {
  for (const e of entries) e.isIntersecting ? mount(e.target) : unmount(e.target);
}, { root:null, rootMargin:"300px 0px", threshold:0.01 });

function render() {
  io.disconnect();
  const q = $("#q").value.toLowerCase();
  const scope = $("#scope").value;
  const f = { object:$("#obj").value, condition:$("#cond").value,
              arm:$("#arm").value, backbone:$("#bb").value };
  const rows = DATA.filter(d =>
    (scope!=="ours" || d.ours) && (scope!=="final" || d.final) &&
    (scope!=="baseline" || d.baseline) &&
    (!f.object || d.object===f.object) && (!f.condition || d.condition===f.condition) &&
    (!f.arm || d.arm===f.arm) && (!f.backbone || d.backbone===f.backbone) &&
    (!q || d.file.toLowerCase().includes(q) || d.object.toLowerCase().includes(q)));
  $("#count").textContent = `${rows.length} / ${DATA.length} models`;
  $("#empty").style.display = rows.length ? "none" : "block";
  grid.innerHTML = rows.map(d => {
    const init = (d.object[0] || "?");
    const cls = d.final ? "card final" : (d.ours ? "card ours" : (d.baseline ? "card base" : "card"));
    const badge = d.final ? `<span class="badge final">OURS ★</span>`
                : (d.ours ? `<span class="badge ours">OURS</span>`
                : (d.baseline ? `<span class="badge base">BASELINE</span>` : ""));
    return `<div class="${cls}">
      ${badge}
      <div class="viewport" data-src="${d.file}" data-label="${init}"><div class="ph">${init}</div><span class="open-hint">click to open</span></div>
      <div class="meta">
        <div class="obj">${d.object}</div>
        <div class="tags">
          ${d.condition?`<span class="tag cond">${d.condition}</span>`:""}
          <span class="tag arm">${d.arm}</span>
          <span class="tag bb">${d.backbone}</span>
          <span class="tag">${d.size_mb} MB</span>
        </div>
        <div style="margin-top:6px"><a class="dl" href="${d.file}" download>↓ ${d.file}</a></div>
      </div>
    </div>`; }).join("");
  grid.querySelectorAll(".viewport").forEach(vp => {
    io.observe(vp);
    vp.addEventListener("click", () => openModal(vp.dataset.src));
  });
}

// Fullscreen lightbox: one big model-viewer, created on open and destroyed on
// close so it never leaks a WebGL context.
const modal = $("#modal"), mbody = $("#mbody");
function openModal(src) {
  const name = src.split("/").pop();
  $("#mname").textContent = name;
  $("#mopen").href = "view.html?m=" + encodeURIComponent(src);   // standalone viewer page
  $("#mdl").href = src;
  mbody.innerHTML =
    `<model-viewer id="bigmv" src="${src}" camera-controls touch-action="pan-y" reveal="auto"
       shadow-intensity="1" exposure="1.1" camera-orbit="35deg 70deg auto"></model-viewer>`;
  modal.classList.add("open");
}
function closeModal() { modal.classList.remove("open"); mbody.innerHTML = ""; }
$("#mclose").addEventListener("click", closeModal);
modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });

// Render a PNG of exactly the angle currently on screen.
$("#msnap").addEventListener("click", async () => {
  const mv = $("#bigmv");
  if (!mv) return;
  try {
    const blob = await mv.toBlob({ mimeType: "image/png", idealAspect: false });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = $("#mname").textContent.replace(/\.glb$/i, "") + "_view.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  } catch (err) { alert("Snapshot failed: " + err); }
});

fetch("manifest.json").then(r=>r.json()).then(d=>{
  DATA = d;
  opts($("#obj"), uniq("object"), "object");
  opts($("#cond"), uniq("condition"), "condition");
  opts($("#arm"), uniq("arm"), "arm");
  opts($("#bb"), uniq("backbone"), "backbone");
  ["#q","#scope","#obj","#cond","#arm","#bb"].forEach(s=>$(s).addEventListener("input",render));
  render();
}).catch(e=>{ $("#empty").style.display="block";
  $("#empty").textContent="Could not load manifest.json — run build_gallery.py and serve over http."; });
</script>
</body>
</html>
"""

VIEW_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLB viewer</title>
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
<style>
  html,body{margin:0;height:100%;background:#0f1115;color:#e8eaed;
    font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
  .bar{display:flex;gap:14px;align-items:center;padding:10px 16px;border-bottom:1px solid #2a2f3a;}
  .name{font-weight:600;} a{color:#8fb4ff;text-decoration:none;} a:hover{text-decoration:underline;}
  button{background:#e63946;color:#fff;border:none;border-radius:8px;padding:7px 12px;
    font-weight:600;cursor:pointer;} button:hover{filter:brightness(1.1);}
  model-viewer{width:100%;height:calc(100vh - 46px);
    background:radial-gradient(circle at 50% 35%,#262b35 0%,#14161b 80%);}
  .err{padding:40px;color:#9aa0ab;}
</style></head><body>
<div class="bar">
  <span class="name" id="name"></span>
  <button id="snap" title="save PNG of current angle">📷 snapshot view</button>
  <a id="dl" download>download .glb ↓</a>
  <a href="gallery.html">← back to gallery</a>
</div>
<div id="host"></div>
<script>
  const src = new URLSearchParams(location.search).get("m");
  const host = document.getElementById("host");
  if (!src) { host.innerHTML = '<div class="err">No model given. Open this page via the gallery.</div>'; }
  else {
    document.getElementById("name").textContent = src.split("/").pop();
    document.getElementById("dl").href = src;
    host.innerHTML = `<model-viewer id="mv" src="${src}" camera-controls touch-action="pan-y"
      reveal="auto" shadow-intensity="1" exposure="1.1" camera-orbit="35deg 70deg auto"></model-viewer>`;
    document.getElementById("snap").addEventListener("click", async () => {
      const mv = document.getElementById("mv");
      const blob = await mv.toBlob({mimeType:"image/png", idealAspect:false});
      const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
      a.download = src.split("/").pop().replace(/\.glb$/i,"") + "_view.png"; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    });
  }
</script></body></html>
"""

if __name__ == "__main__":
    main()
