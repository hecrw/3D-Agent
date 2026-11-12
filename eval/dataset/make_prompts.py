#!/usr/bin/env python
"""Build eval/dataset/captions.csv: the 50 generic object prompts.

    .venv/bin/python eval/dataset/make_prompts.py

Each prompt is a plain object category ("a bicycle", "a teapot") with no
attributes. Categories are taken from the T3Bench single-object prompts
(github.com/THU-LYJ-Lab/T3Bench, data/prompt_single.txt), each reduced to its
head object (GENERIC below, keyed by T3Bench line number). Selection rule,
fixed before any results:
  * the 20 objects of the original submission, then
  * the other 80 T3Bench lines in a random order (seed 20261112), taking each
    one whose generic object is not already in the set, until there are 50;
  * an object for which no openly licensed real photo met the photo criteria
    (see curate_photos.py) is skipped, and the next object in the same order
    takes its place.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

SEED, N = 20261112, 50
# Objects with no Wikimedia Commons candidate meeting the photo criteria.
NO_PHOTO = {"map", "carnival mask"}
ORIGINAL = [75, 76, 47, 65, 41, 29, 9, 21, 59, 8, 51, 70, 85, 91, 53, 73, 54, 18, 42, 17]
OUT = Path(__file__).resolve().parent / "captions.csv"

# T3Bench single-object line -> head object (attributes removed)
GENERIC = {
    1: "cactus", 2: "umbrella", 3: "rocking horse", 4: "dog", 5: "candle holder",
    6: "pair of sneakers", 7: "mug", 8: "fire hydrant", 9: "saxophone", 10: "book",
    11: "sunflower", 12: "sandcastle", 13: "skateboard", 14: "flag", 15: "teddy bear",
    16: "watermelon", 17: "ring", 18: "doll", 19: "chameleon", 20: "pocket watch",
    21: "teapot", 22: "brooch", 23: "feather quill", 24: "typewriter", 25: "beetle",
    26: "paperweight", 27: "cushion", 28: "rabbit figurine", 29: "electric guitar",
    30: "pair of swimming goggles", 31: "tortoise shell", 32: "scarf", 33: "map",
    34: "frying pan", 35: "soda can", 36: "book", 37: "pendant necklace", 38: "hat",
    39: "letter opener", 40: "perfume bottle", 41: "grand piano", 42: "trumpet",
    43: "wedding dress", 44: "teacup", 45: "candle holder", 46: "quilt", 47: "armchair",
    48: "top hat", 49: "easel", 50: "crowbar", 51: "paper airplane", 52: "tire swing",
    53: "chess set", 54: "kite", 55: "gemstone", 56: "key", 57: "doily",
    58: "walking cane", 59: "chandelier", 60: "shirt", 61: "doll", 62: "typewriter",
    63: "grand piano", 64: "hat", 65: "car", 66: "lipstick", 67: "wooden chest",
    68: "carnival mask", 69: "map", 70: "apple", 71: "briefcase", 72: "pocket watch",
    73: "smartphone", 74: "beach ball", 75: "bicycle", 76: "cat", 77: "origami",
    78: "electric guitar", 79: "baseball bat", 80: "champagne flute", 81: "hat",
    82: "soccer ball", 83: "pair of jeans", 84: "teddy bear", 85: "mug",
    86: "rubber duck", 87: "teapot", 88: "window", 89: "vase",
    90: "flamingo lawn ornament", 91: "potted orchid", 92: "ship's wheel", 93: "tiara",
    94: "blanket", 95: "pair of shoes", 96: "quill pen", 97: "wreath", 98: "mirror",
    99: "watering can", 100: "radio",
}


def article(obj: str) -> str:
    return ("an " if obj[0] in "aeiou" else "a ") + obj


def main() -> None:
    assert len(GENERIC) == 100
    chosen, seen = [], set()
    for n in ORIGINAL:
        chosen.append((n, "submission")); seen.add(GENERIC[n])
    rest = [n for n in GENERIC if n not in ORIGINAL]
    random.Random(SEED).shuffle(rest)
    for n in rest:
        if len(chosen) == N:
            break
        if GENERIC[n] not in seen and GENERIC[n] not in NO_PHOTO:
            chosen.append((n, "random")); seen.add(GENERIC[n])
    assert len(chosen) == N
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "caption", "object", "t3bench_line", "source"])
        for n, src in chosen:
            obj = GENERIC[n]
            slug = obj.replace(" ", "_").replace("'", "")
            w.writerow([f"{slug}.jpg", article(obj), obj, n, src])
    print(f"wrote {N} prompts -> {OUT}")


if __name__ == "__main__":
    main()
