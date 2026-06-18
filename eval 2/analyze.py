#!/usr/bin/env python
"""Aggregate a results CSV into per-axis marginal-effect tables.

    .venv/bin/python eval/analyze.py eval/results_pilot.csv

For each metric (CLIP, Gen3DEval, ULIP-2), prints:
  1. Mean score by condition
  2. Marginal effect per axis: all_on − loo_<axis>, paired per prompt

A positive marginal effect means removing that axis hurts quality,
i.e. the axis contributes positively to the restyle.
"""
from __future__ import annotations

import csv
import sys
from statistics import mean, pstdev

METRICS = [
    ("clip_mean", "CLIP"),
    ("gen3deval",  "Gen3DEval"),
    ("ulip_mean",  "ULIP-2"),
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


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: analyze.py <results.csv>")
    rows = load(sys.argv[1])
    if not rows:
        sys.exit("no successful rows to analyze")

    conds      = {r["condition"] for r in rows}
    known      = ["raw", "all_on"] + sorted(c for c in conds if c.startswith("loo_"))
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

        print(f"\n  {label} — marginal effect per axis  (positive = axis helps)\n")
        _marginal_table(rows, col, axes)


if __name__ == "__main__":
    main()
