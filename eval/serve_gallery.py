#!/usr/bin/env python
"""Live server for the 3D gallery.

Same static pages as build_gallery.py, but the manifest is recomputed on every
request — so meshes the app writes into media/3d_outputs show up the moment you
refresh the page, with no rebuild step. Serves from the repo root so the
external baseline GLBs (../../eval/baseline_glbs_*) resolve.

    python eval/serve_gallery.py            # http://localhost:8000/media/3d_outputs/gallery.html
    python eval/serve_gallery.py --port 8001
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import build_gallery as bg

MANIFEST_PATH = "/media/3d_outputs/manifest.json"


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        # Intercept the manifest and recompute it live from disk.
        if self.path.split("?")[0] == MANIFEST_PATH:
            payload = json.dumps(bg.build_items()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def log_message(self, *a):  # quieter console
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    bg.write_pages()  # ensure gallery.html + view.html exist / are current
    handler = partial(Handler, directory=str(bg.REPO))
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    url = f"http://localhost:{args.port}/media/3d_outputs/gallery.html"
    n = len(bg.build_items())
    print(f"live gallery serving {n} GLBs (auto-refreshes on new generations)")
    print(f"-> {url}")
    print("   (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
