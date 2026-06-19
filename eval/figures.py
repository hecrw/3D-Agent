#!/usr/bin/env python
"""Generate the §4.5 data figures from the results CSVs (publication-styled PNGs).

    python eval/figures.py

Reads results_pilot.csv (generated), results_retrieved.csv, and the three
results_baseline_*.csv, then writes to eval/figures/:
  fig_baseline_gen3d.png   - pipeline vs 3 baselines (Gen3DEval)
  fig_domain_gap.png       - raw vs all_on, generated vs retrieved (the thesis)
  fig_axis_ablation.png    - per-axis Gen3DEval marginal drop (LOO)
  fig_per_object.png       - per-object Gen3DEval, sorted (difficulty spread)
  fig_per_view_clip.png    - CLIP by rendered view (top/bottom weakness)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL = Path(__file__).resolve().parent
OUT = EVAL / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})
INK = "#1d3557"
ACCENT = "#e63946"
MUTED = "#a8a8a8"


def load(p):
    return [r for r in csv.DictReader(open(EVAL / p)) if r.get("status") == "ok"]


def col(rows, cond, c):
    v = [float(r[c]) for r in rows
         if r["condition"] == cond and r.get(c, "").strip() not in ("", "nan")]
    return mean(v) if v else float("nan")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT / name}")


def fig_baseline():
    pilot = load("results_pilot.csv")
    bases = {b: load(f"results_baseline_{b}.csv") for b in ("trellis", "hunyuan", "partcrafter")}
    names = ["Ours\n(all_on)", "Ours\n(raw)", "TRELLIS", "Hunyuan3D-2", "PartCrafter"]
    vals = [col(pilot, "all_on", "gen3deval"), col(pilot, "raw", "gen3deval"),
            col(bases["trellis"], "baseline", "gen3deval"),
            col(bases["hunyuan"], "baseline", "gen3deval"),
            col(bases["partcrafter"], "baseline", "gen3deval")]
    colors = [ACCENT, ACCENT, MUTED, MUTED, MUTED]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Gen3DEval (1–10)")
    ax.set_title("Geometric quality: our pipeline vs. baselines")
    ax.set_ylim(0, 10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=10)
    save(fig, "fig_baseline_gen3d.png")


def fig_domain_gap():
    gen = load("results_pilot.csv")
    ret = load("results_retrieved.csv")
    arms = ["Generated\n(synthetic input)", "Retrieved\n(real photo)"]
    raw = [col(gen, "raw", "gen3deval"), col(ret, "raw", "gen3deval")]
    allon = [col(gen, "all_on", "gen3deval"), col(ret, "all_on", "gen3deval")]
    x = range(len(arms))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    b1 = ax.bar([i - w / 2 for i in x], raw, w, label="raw (no restyle)", color=MUTED)
    b2 = ax.bar([i + w / 2 for i in x], allon, w, label="all_on (restyle)", color=ACCENT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms)
    ax.set_ylabel("Gen3DEval (1–10)")
    ax.set_ylim(0, 10)
    ax.set_title("Restyle is inert on clean inputs, large on real photos")
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    f"{b.get_height():.2f}", ha="center", fontsize=9)
    save(fig, "fig_domain_gap.png")


def fig_axis_ablation():
    ret = load("results_retrieved.csv")
    allon = col(ret, "all_on", "gen3deval")
    axes = ["background", "framing", "view", "lighting", "isolation", "part_visibility"]
    drops = [(a, allon - col(ret, f"loo_{a}", "gen3deval")) for a in axes]
    drops.sort(key=lambda t: t[1])  # ascending so biggest effect on top
    labels = [a.replace("_", " ") for a, _ in drops]
    vals = [d for _, d in drops]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh(labels, vals, color=INK)
    ax.set_xlabel("Gen3DEval drop when axis removed (all_on − loo_X)")
    ax.set_title("Per-axis marginal contribution (retrieved arm)")
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    save(fig, "fig_axis_ablation.png")


def fig_per_object():
    gen = load("results_pilot.csv")
    objs = {}
    for r in gen:
        if r["condition"] == "all_on" and r.get("gen3deval", "").strip() not in ("", "nan"):
            objs[r["prompt"]] = float(r["gen3deval"])
    items = sorted(objs.items(), key=lambda t: t[1])
    labels = [p[:34] for p, _ in items]
    vals = [v for _, v in items]
    colors = [ACCENT if v <= 3 else INK for v in vals]
    fig, ax = plt.subplots(figsize=(6.6, max(4, 0.32 * len(items))))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Gen3DEval (1–10)")
    ax.set_title("Per-object geometric quality (all_on, generated)")
    ax.set_xlim(0, 10)
    save(fig, "fig_per_object.png")


def fig_per_view():
    # Mean per-view CLIP from the per_view JSON column across all arms/conditions.
    views = ["front", "back", "left", "right", "top", "bottom"]
    acc = {v: [] for v in views}
    for csvname in ("results_pilot.csv", "results_retrieved.csv"):
        for r in load(csvname):
            try:
                pv = json.loads(r.get("per_view") or "{}")
            except Exception:
                continue
            for v in views:
                if v in pv:
                    acc[v].append(float(pv[v]))
    vals = [mean(acc[v]) if acc[v] else float("nan") for v in views]
    colors = [ACCENT if v in ("top", "bottom") else INK for v in views]
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    bars = ax.bar(views, vals, color=colors)
    ax.set_ylabel("Mean CLIP")
    ax.set_title("Directional weakness: top/bottom views underperform")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    save(fig, "fig_per_view_clip.png")


if __name__ == "__main__":
    print("writing figures to", OUT)
    fig_baseline()
    fig_domain_gap()
    fig_axis_ablation()
    fig_per_object()
    fig_per_view()
    print("done.")
