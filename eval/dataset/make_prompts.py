#!/usr/bin/env python
"""Build eval/dataset/captions.csv: the 50 T3Bench prompts used in the paper.

    .venv/bin/python eval/dataset/make_prompts.py

Selection rule (fixed before any results, reproducible):
  * the 20 prompts of the original submission (kept for continuity), plus
  * 30 more drawn uniformly at random, seed 20261112, from the remaining 80
    T3Bench single-object prompts.
Prompts are T3Bench's exact wording (the submission had lower-cased them and
dropped commas). Source: data/prompt_single.txt of github.com/THU-LYJ-Lab/T3Bench
at the pinned commit. Each prompt's id is its 1-based line number there
(stem t3s_NNN), so any prompt can be traced back.
"""
from __future__ import annotations

import csv
import random
import urllib.request
from pathlib import Path

COMMIT = "6367462cfd04467af772c168c5cd8570964b9df3"
URL = f"https://raw.githubusercontent.com/THU-LYJ-Lab/T3Bench/{COMMIT}/data/prompt_single.txt"
SEED, N_EXTRA = 20261112, 30
# The 20 prompts of the original submission, by T3Bench line number.
ORIGINAL = [75, 76, 47, 65, 41, 29, 9, 21, 59, 8, 51, 70, 85, 91, 53, 73, 54, 18, 42, 17]
OUT = Path(__file__).resolve().parent / "captions.csv"


def main() -> None:
    lines = urllib.request.urlopen(URL, timeout=30).read().decode("utf-8").splitlines()
    prompts = {i + 1: p.strip() for i, p in enumerate(lines) if p.strip()}
    assert len(prompts) == 100, len(prompts)
    rest = sorted(set(prompts) - set(ORIGINAL))
    extra = sorted(random.Random(SEED).sample(rest, N_EXTRA))
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "caption", "t3bench_line", "source"])
        for n in ORIGINAL:
            w.writerow([f"t3s_{n:03d}.jpg", prompts[n], n, "original"])
        for n in extra:
            w.writerow([f"t3s_{n:03d}.jpg", prompts[n], n, "random"])
    print(f"wrote {len(ORIGINAL) + len(extra)} prompts -> {OUT}")


if __name__ == "__main__":
    main()
