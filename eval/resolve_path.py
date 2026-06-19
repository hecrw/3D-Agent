#!/usr/bin/env python
"""Resolve an artifact path that may have been written on another machine.

CSVs record absolute mesh/image paths. When artifacts are generated across two
boxes (Mac + the Linux GPU box) and then consolidated into one
media/3d_outputs, the stored absolute path no longer exists locally even though
the file (by basename) is present. resolve() falls back to a basename lookup in
media/3d_outputs so every figure script finds every artifact regardless of
which machine produced it.
"""
from __future__ import annotations

from pathlib import Path

_MEDIA = Path(__file__).resolve().parent.parent / "media" / "3d_outputs"
_NAMES: dict[str, Path] | None = None


def resolve(p) -> str | None:
    """Return a local path for `p`, or None if neither the stored path nor a
    same-basename file in media/3d_outputs exists."""
    if not p:
        return None
    pp = Path(p)
    if pp.exists():
        return str(pp)
    global _NAMES
    if _NAMES is None:
        _NAMES = {f.name: f for f in _MEDIA.glob("*")} if _MEDIA.exists() else {}
    hit = _NAMES.get(pp.name)
    return str(hit) if hit else None
