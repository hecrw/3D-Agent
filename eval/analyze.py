#!/usr/bin/env python
"""Aggregate a results CSV into the per-axis marginal-effect table.

    .venv/bin/python eval/analyze.py eval/results_pilot.csv

Marginal effect of an axis = mean(all_on) - mean(loo_<axis>), computed per
backbone over images where BOTH conditions succeeded (paired). A positive
number means the axis helps. Pure stdlib — no pandas needed.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from statistics import mean, pstdev


def load(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r.get("status") == "ok"]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: analyze.py <results.csv>")
    rows = load(sys.argv[1])
    if not rows:
        sys.exit("no successful rows to analyze")

    # score[(backbone, condition, image)] = clip_mean
    score: dict[tuple[str, str, str], float] = {}
    backbones, conds = set(), set()
    for r in rows:
        try:
            score[(r["backbone"], r["condition"], r["image"])] = float(r["clip_mean"])
        except ValueError:
            continue
        backbones.add(r["backbone"])
        conds.add(r["condition"])

    # --- absolute means per condition x backbone ---
    print("\n=== Mean CLIP score by condition x backbone ===")
    cond_order = ["raw", "all_on"] + sorted(c for c in conds if c.startswith("loo_"))
    header = "condition".ljust(20) + "".join(b.ljust(14) for b in sorted(backbones))
    print(header)
    for c in cond_order:
        if c not in conds:
            continue
        line = c.ljust(20)
        for b in sorted(backbones):
            vals = [v for (bb, cc, _), v in score.items() if bb == b and cc == c]
            line += (f"{mean(vals):.3f} (n={len(vals)})".ljust(14)
                     if vals else "-".ljust(14))
        print(line)

    # --- per-axis marginal effect: all_on - loo_<axis>, paired per image ---
    print("\n=== Marginal effect per axis  [all_on - loo_axis]  (paired) ===")
    print("positive = axis helps quality\n")
    axes = sorted(c[len("loo_"):] for c in conds if c.startswith("loo_"))
    for b in sorted(backbones):
        print(f"-- {b} --")
        for ax in axes:
            deltas = []
            imgs = {img for (bb, cc, img) in score
                    if bb == b and cc in ("all_on", f"loo_{ax}")}
            for img in imgs:
                a = score.get((b, "all_on", img))
                l = score.get((b, f"loo_{ax}", img))
                if a is not None and l is not None:
                    deltas.append(a - l)
            if deltas:
                sd = pstdev(deltas) if len(deltas) > 1 else 0.0
                print(f"  {ax.ljust(16)} Δ={mean(deltas):+.4f}  "
                      f"(±{sd:.4f}, n={len(deltas)})")
            else:
                print(f"  {ax.ljust(16)} no paired data")
        print()


if __name__ == "__main__":
    main()
