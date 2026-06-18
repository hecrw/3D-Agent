#!/usr/bin/env python
"""Merge ULIP scores from Colab back into a results CSV.

    .venv/bin/python eval/merge_ulip.py eval/results_retrieved.csv eval/ulip_scores.csv

ulip_scores.csv (from the Colab notebook) has columns key,ulip. We match each
results row by its mesh-filename stem and fill ulip_mean. Rewrites in place (atomic).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: merge_ulip.py <results.csv> <ulip_scores.csv>")
    results_path, scores_path = Path(sys.argv[1]), Path(sys.argv[2])

    scores = {}
    with open(scores_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("ulip"):
                scores[r["key"]] = r["ulip"]

    with open(results_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "ulip_mean" not in fields:
        sys.exit(f"{results_path}: missing ulip_mean column")

    filled = 0
    for r in rows:
        mp = r.get("mesh_path")
        if not mp:
            continue
        stem = Path(mp).stem
        if stem in scores:
            r["ulip_mean"] = scores[stem]
            filled += 1

    tmp = results_path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(results_path)
    print(f"filled ulip_mean on {filled} rows -> {results_path}")


if __name__ == "__main__":
    main()
