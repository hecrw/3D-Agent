#!/usr/bin/env python
"""Standalone render helper — called as a subprocess by run_pilot.py.

Usage: python _render_helper.py <mesh_path> <out_dir>

Prints one PNG path per line to stdout. Runs on its own main thread so
pyrender's pyglet backend doesn't crash macOS AppKit.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tools

if __name__ == "__main__":
    mesh_path, out_dir = sys.argv[1], sys.argv[2]
    paths = tools.render_mesh_views(mesh_path, out_dir, views="default")
    for p in paths.values():
        print(p)
