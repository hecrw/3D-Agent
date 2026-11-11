#!/usr/bin/env python
"""Aggregate a results CSV into per-axis marginal-effect tables.

    .venv/bin/python eval/analyze.py eval/results_pilot.csv

For each metric (CLIP, VLM score, ULIP-2, T3Bench Quality/Alignment), prints:
  1. Mean score by condition
  2. Paired contrasts: restyle, lighting, review, and the review x
     preprocessing interaction — mean difference, bootstrap 95% CI,
     Wilcoxon signed-rank p, Holm-adjusted p

A positive marginal effect means removing that axis hurts quality,
i.e. the axis contributes positively to the restyle.
"""
from __future__ import annotations

import csv
import sys
from statistics import mean, pstdev

METRICS = [
    ("clip_mean", "CLIP"),
    ("vlm_score",  "VLM judge"),
    ("ulip_mean",  "ULIP-2"),
    ("t3_quality",   "T3Bench Quality"),
    ("t3_alignment", "T3Bench Alignment"),
]


def load(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("status") == "ok"]


def _mean_table(rows: list[dict], metric: str, cond_order: list[str]) -> None:
    score: dict[tuple[str, str], float] = {}
    for r in rows:
        try:
            score[(r["condition"], r["prompt"])] = float(r[metric])
        except (ValueError, KeyError):
            continue

    print(f"{'condition':<22}  {'mean':>6}  {'n':>4}")
    print("-" * 36)
    for c in cond_order:
        vals = [v for (cc, _), v in score.items() if cc == c]
        if vals:
            print(f"{c:<22}  {mean(vals):>6.3f}  {len(vals):>4}")
        else:
            print(f"{c:<22}  {'—':>6}")


def _marginal_table(rows: list[dict], metric: str, axes: list[str]) -> None:
    score: dict[tuple[str, str], float] = {}
    for r in rows:
        try:
            score[(r["condition"], r["prompt"])] = float(r[metric])
        except (ValueError, KeyError):
            continue

    print(f"{'axis':<20}  {'Δ (all_on − loo)':>18}  {'±sd':>8}  {'n':>4}")
    print("-" * 56)
    for ax in axes:
        prompts = {p for (cc, p) in score
                   if cc in ("all_on", f"loo_{ax}")}
        deltas = []
        for p in prompts:
            a = score.get(("all_on",    p))
            l = score.get((f"loo_{ax}", p))
            if a is not None and l is not None:
                deltas.append(a - l)
        if deltas:
            sd = pstdev(deltas) if len(deltas) > 1 else 0.0
            print(f"{ax:<20}  {mean(deltas):>+18.4f}  {sd:>8.4f}  {len(deltas):>4}")
        else:
            print(f"{ax:<20}  {'no paired data':>18}")


# Paired contrasts reported in the paper: (label, plus-conditions, minus-conditions).
# Each is computed per object over the objects that have every condition.
CONTRASTS = [
    ("restyle (all_on - raw)",                  ["all_on"], ["raw"]),
    ("lighting (all_on - loo_lighting)",        ["all_on"], ["loo_lighting"]),
    ("review, no preproc (raw_review - raw)",   ["raw_review"], ["raw"]),
    ("review, with preproc (all_on_review - all_on)", ["all_on_review"], ["all_on"]),
    ("preproc vs review (all_on - raw_review)", ["all_on"], ["raw_review"]),
    ("both vs neither (all_on_review - raw)",   ["all_on_review"], ["raw"]),
    ("interaction ((AR-A) - (RR-R))",           ["all_on_review", "raw"], ["all_on", "raw_review"]),
]


def _contrast_table(rows: list[dict], metric: str) -> None:
    """Mean paired difference, bootstrap 95% CI, Wilcoxon signed-rank p, and
    Holm-adjusted p across the contrasts of this metric."""
    import random
    try:
        from scipy.stats import wilcoxon
    except ImportError:  # pragma: no cover
        wilcoxon = None
    score: dict[tuple[str, str], float] = {}
    for r in rows:
        try:
            score[(r["condition"], r["prompt"])] = float(r[metric])
        except (ValueError, KeyError):
            continue
    prompts = sorted({p for _, p in score})
    results = []
    for label, plus, minus in CONTRASTS:
        need = plus + minus
        diffs = [sum(score[(c, p)] for c in plus) - sum(score[(c, p)] for c in minus)
                 for p in prompts if all((c, p) in score for c in need)]
        if len(diffs) < 3:
            continue
        rng = random.Random(0)
        boots = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(5000))
        ci = (boots[124], boots[4874])
        pval = float("nan")
        if wilcoxon and any(d != 0 for d in diffs):
            pval = float(wilcoxon(diffs).pvalue)
        results.append([label, len(diffs), mean(diffs), ci, pval])
    # Holm correction across this metric's contrasts
    order = sorted(range(len(results)), key=lambda i: (results[i][4] != results[i][4], results[i][4]))
    m, running = len(results), 0.0
    for rank, i in enumerate(order):
        p = results[i][4]
        adj = min(1.0, max(running, (m - rank) * p)) if p == p else float("nan")
        running = adj if adj == adj else running
        results[i].append(adj)
    print(f"  {'contrast':<48} {'n':>3} {'mean diff':>10} {'95% CI':>20} {'p':>8} {'p_holm':>8}")
    for label, n, md, (lo, hi), p, ph in results:
        print(f"  {label:<48} {n:>3} {md:>+10.4f}  [{lo:+.4f}, {hi:+.4f}] {p:>8.4f} {ph:>8.4f}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: analyze.py <results.csv>")
    rows = load(sys.argv[1])
    if not rows:
        sys.exit("no successful rows to analyze")

    conds      = {r["condition"] for r in rows}
    known      = (["raw", "all_on"] + sorted(c for c in conds if c.startswith("loo_"))
                  + ["raw_review", "all_on_review"])
    # append any other conditions present (e.g. "baseline" from score_baseline.py)
    extra      = sorted(c for c in conds if c not in known)
    cond_order = [c for c in known if c in conds] + extra
    axes       = sorted(c[len("loo_"):] for c in conds if c.startswith("loo_"))

    for col, label in METRICS:
        has_data = any(r.get(col, "") not in ("", "nan") for r in rows)
        if not has_data:
            print(f"\n[{label}] no data — skipping\n")
            continue

        print(f"\n{'='*56}")
        print(f"  {label} — mean by condition")
        print(f"{'='*56}")
        _mean_table(rows, col, cond_order)

        print(f"\n  {label} — paired contrasts (positive = first term better)\n")
        _contrast_table(rows, col)


if __name__ == "__main__":
    main()
