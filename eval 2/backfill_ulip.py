#!/usr/bin/env python
"""Backfill the ulip_mean column for ok rows that are missing it.

    .venv/bin/python eval/backfill_ulip.py eval/results_retrieved.csv

Needs only each row's saved mesh (mesh_path). Samples a point cloud locally and
scores it on the ULIP-2 Modal app (must be deployed; TRELLIS_WORKSPACE in .env
selects the workspace). Rewrites the CSV in place (atomic). Safe to re-run.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(EVAL_DIR.parent / ".env")

from ulip_client import score_ulip, _endpoint  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: backfill_ulip.py <results.csv>")
    if not _endpoint():
        sys.exit("TRELLIS_WORKSPACE not set — can't reach the ULIP Modal app.")

    path = Path(sys.argv[1])
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "ulip_mean" not in fields:
        sys.exit(f"{path}: missing ulip_mean column")

    todo = [r for r in rows
            if r.get("status") == "ok"
            and not r.get("ulip_mean")
            and r.get("mesh_path") and Path(r["mesh_path"]).exists()]
    print(f"{len(todo)} ok rows need a ULIP score")

    for i, r in enumerate(todo, 1):
        mesh = r["mesh_path"]
        print(f"[{i}/{len(todo)}] {r['condition']:22} {Path(mesh).name}")
        score = score_ulip(mesh, r["prompt"])
        r["ulip_mean"] = "" if math.isnan(score) else f"{score:.4f}"
        print(f"    -> ulip={r['ulip_mean'] or 'FAILED'}")

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
