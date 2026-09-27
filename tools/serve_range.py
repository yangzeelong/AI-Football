#!/usr/bin/env python3
"""Static file server with HTTP Range support, for watching big videos remotely.

`python3 -m http.server` always answers with the whole file and never sends
Accept-Ranges, so a video player cannot seek: it has to pull the entire file in
one response, and seeking or closing the tab shows up as a wall of
ConnectionResetError tracebacks. This one serves `206 Partial Content`
properly, so the player streams, seeks, and reconnects without noise.

    python3 tools/serve_range.py --directory output/touch_eval_moving_pass --port 8080
    # then open http://localhost:8080/ (through the IDE's port forwarding)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

RANGE_PATTERN = re.compile(r"bytes=(\d*)-(\d*)$")

CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}

CHUNK_BYTES = 256 * 1024


class RangeRequestHandler(BaseHTTPRequestHandler):
    server_version = "nexusflow-range/1.0"
    protocol_version = "HTTP/1.1"
    directory = "."

    def log_message(self, fmt: str, *args) -> None:
        # One line per request instead of the default noisy format; errors are
        # handled where they happen.
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- request entry points ------------------------------------------------

    def do_HEAD(self) -> None:  # noqa: N802 (http.server naming)
        self._serve(head_only=True)

    def do_GET(self) -> None:  # noqa: N802
        self._serve(head_only=False)

    # -- implementation -----------------------------------------------------

    def _resolve(self) -> str | None:
        path = unquote(urlparse(self.path).path)
        if path.endswith("/"):
            path += "index.html"
        root = os.path.abspath(self.directory)
        full = os.path.abspath(os.path.join(root, path.lstrip("/")))
        if full != root and not full.startswith(root + os.sep):
            return None  # path traversal
        return full

    def _parse_range(self, size: int) -> tuple[int, int] | None:
        """(start, end) for a satisfiable range, None when there is no range."""
        header = self.headers.get("Range", "").strip()
        match = RANGE_PATTERN.match(header)
        if not header or not match:
            return None
        first, last = match.group(1), match.group(2)
        if first == "":
            if last == "":
                return None
            # "bytes=-N": the last N bytes.
            start = max(0, size - int(last))
            end = size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
        if start >= size or start > end:
            return (-1, -1)  # unsatisfiable
        return (start, min(end, size - 1))

    def _serve(self, head_only: bool) -> None:
        full = self._resolve()
        if not full or not os.path.isfile(full):
            self.send_error(HTTPStatus.NOT_FOUND, "no such file")
            return

        size = os.path.getsize(full)
        extension = os.path.splitext(full)[1].lower()
        content_type = CONTENT_TYPES.get(extension, "application/octet-stream")

        status = HTTPStatus.OK
        start, end = 0, max(0, size - 1)
        if size > 0:
            requested = self._parse_range(size)
            if requested == (-1, -1):
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if requested is not None:
                start, end = requested
                status = HTTPStatus.PARTIAL_CONTENT

        length = 0 if size == 0 else end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        # The player needs this to know it may ask for pieces.
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if head_only or length == 0:
            return

        try:
            with open(full, "rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = handle.read(min(CHUNK_BYTES, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            # The player seeked or closed the tab mid-response. Expected while
            # scrubbing a video, and not worth a traceback.
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=".",
                        help="Directory to serve (default: current).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="0.0.0.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        print(f"[serve] not a directory: {directory}")
        return 1

    RangeRequestHandler.directory = directory
    server = ThreadingHTTPServer((args.bind, args.port), RangeRequestHandler)
    server.daemon_threads = True

    print(f"[serve] {directory}")
    print(f"[serve] http://{args.bind}:{args.port}/  (Range enabled, 206 on seek)")
    print("[serve] stop with Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
