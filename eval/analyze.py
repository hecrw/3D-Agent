#!/usr/bin/env python
"""Aggregate a results CSV into per-axis marginal-effect tables.

    .venv/bin/python eval/analyze.py eval/results_pilot.csv

For each metric (CLIP, Gen3DEval, ULIP-2), prints:
  1. Mean score by condition × backbone
  2. Marginal effect per axis: all_on − loo_<axis>, paired per image

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


def _mean_table(rows: list[dict], metric: str,
                backbones: list[str], cond_order: list[str]) -> None:
    score: dict[tuple[str, str, str], float] = {}
    for r in rows:
        try:
            v = float(r[metric])
        except (ValueError, KeyError):
            continue
        score[(r["backbone"], r["condition"], r["image"])] = v

    header = "condition".ljust(22) + "".join(b.ljust(18) for b in backbones)
    print(header)
    for c in cond_order:
        line = c.ljust(22)
        for b in backbones:
            vals = [v for (bb, cc, _), v in score.items() if bb == b and cc == c]
            line += (f"{mean(vals):.3f} (n={len(vals)})".ljust(18)
                     if vals else "-".ljust(18))
        print(line)


def _marginal_table(rows: list[dict], metric: str,
                    backbones: list[str], axes: list[str]) -> None:
    score: dict[tuple[str, str, str], float] = {}
    for r in rows:
        try:
            v = float(r[metric])
        except (ValueError, KeyError):
            continue
        score[(r["backbone"], r["condition"], r["image"])] = v

    for b in backbones:
        print(f"-- {b} --")
        for ax in axes:
            imgs = {img for (bb, cc, img) in score
                    if bb == b and cc in ("all_on", f"loo_{ax}")}
            deltas = []
            for img in imgs:
                a = score.get((b, "all_on",    img))
                l = score.get((b, f"loo_{ax}", img))
                if a is not None and l is not None:
                    deltas.append(a - l)
            if deltas:
                sd = pstdev(deltas) if len(deltas) > 1 else 0.0
                print(f"  {ax.ljust(18)} Δ={mean(deltas):+.4f}  "
                      f"(±{sd:.4f}, n={len(deltas)})")
            else:
                print(f"  {ax.ljust(18)} no paired data")
        print()


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: analyze.py <results.csv>")
    rows = load(sys.argv[1])
    if not rows:
        sys.exit("no successful rows to analyze")

    backbones  = sorted({r["backbone"]  for r in rows})
    conds      = {r["condition"] for r in rows}
    cond_order = (["raw", "all_on"]
                  + sorted(c for c in conds if c.startswith("loo_")))
    axes       = sorted(c[len("loo_"):] for c in conds if c.startswith("loo_"))

    for col, label in METRICS:
        has_data = any(r.get(col, "") not in ("", "nan") for r in rows)
        if not has_data:
            print(f"\n[{label}] no data — skipping\n")
            continue

        print(f"\n{'='*60}")
        print(f"  {label} — mean by condition x backbone")
        print(f"{'='*60}")
        _mean_table(rows, col, backbones, cond_order)

        print(f"\n  {label} — marginal effect per axis  [all_on - loo_axis]  (paired)")
        print("  positive = axis helps quality\n")
        _marginal_table(rows, col, backbones, axes)


if __name__ == "__main__":
    main()
