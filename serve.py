#!/usr/bin/env python3
"""Static server for the viewer that never lets the browser use stale files.

Every response carries `Cache-Control: no-cache`, so scripts, meshes and volumes are
revalidated (If-Modified-Since -> 304) on each load instead of being served from the
browser cache for hours after the pipeline regenerated them.

Usage: python3 serve.py [port]   (default 8765, binds 127.0.0.1, serves ./viewer)
"""
import http.server
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "viewer"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def json_response(self, value, status=200):
        data=json.dumps(value,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

    def do_POST(self):
        self.json_response({'error':'Viewer is read-only. Prepare CFD and thermal recordings with the offline scripts.'},405)

    def log_message(self, fmt, *args):  # quiet: only errors
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"viewer: http://127.0.0.1:{PORT}/  (serving {ROOT})", flush=True)
        httpd.serve_forever()
