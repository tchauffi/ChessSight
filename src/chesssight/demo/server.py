"""A small local server that reads a position off a photograph you drop on it.

Deliberately built on the standard library. The demo exists to show the
pipeline working, and a demo that drags in a web framework is a demo that
stops working when the framework moves; ``http.server`` is enough for one
person pointing a browser at their own machine.

The handler takes any object with a ``read(image) -> dict`` method, which is
what :class:`chesssight.train.predict_position.PositionReader` is, so the
server can be tested without loading a checkpoint.
"""

from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from io import BytesIO
from typing import Any, Protocol

from PIL import Image

from chesssight.demo.render import board_svg, jpeg_uri, overlay

#: Refuse a body larger than this. A phone photograph is a few megabytes; far
#: more than this is a mistake or a probe, and either way it should not be read
#: into memory first.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

#: Width the returned overlay is rendered at. Large enough to see a piece box,
#: small enough that the base64 stays in the hundreds of kilobytes.
OVERLAY_WIDTH = 900


class Reader(Protocol):
    """What the server needs from the pipeline: one image in, one dict out."""

    def read(self, image: Image.Image) -> dict[str, Any]: ...


@dataclass
class Demo:
    """The loaded pipeline plus the lock that serialises access to it.

    One GPU and one model: concurrent requests would interleave inside the
    same module, so they queue instead. A demo serving one person does not
    notice, and a demo serving several stays correct.
    """

    reader: Reader
    lock: threading.Lock

    def read_bytes(self, payload: bytes) -> dict[str, Any]:
        """Read one uploaded image, returning the JSON the page expects.

        "No board found" is an answer rather than an error, and comes back as
        a normal 200 with ``found: false`` -- the page has something to say in
        that case, and it is not a failure of the request.
        """
        image = Image.open(BytesIO(payload)).convert("RGB")
        with self.lock:
            result = self.reader.read(image)

        detections = list(result.get("detections") or [])
        corners = result.get("corners")
        pieces = [d for d in detections if str(d["name"]) != "board"]
        response: dict[str, Any] = {
            "found": result.get("fen") is not None,
            "fen": result.get("fen"),
            "pieces": len(pieces),
            "on_board": (
                sum(1 for row in result["grid"] for value in row if value)
                if result.get("grid")
                else 0
            ),
            "size": list(image.size),
            "overlay": jpeg_uri(
                overlay(image, detections, corners, width=OVERLAY_WIDTH)
            ),
        }
        if response["found"]:
            response["diagram"] = board_svg(str(result["fen"]))
        return response


def page_html() -> bytes:
    """The single-page UI, read from the package rather than built here."""
    return resources.files("chesssight.demo").joinpath("page.html").read_bytes()


class Handler(BaseHTTPRequestHandler):
    """Two routes: the page, and the thing the page posts an image to."""

    demo: Demo
    server_version = "chesssight-demo"

    def log_message(self, format: str, *args: Any) -> None:
        # BaseHTTPRequestHandler logs every request to stderr; the useful line
        # is printed by the reader instead.
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(
            code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8"
        )

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's spelling
        if self.path in ("/", "/index.html"):
            self._send(200, page_html(), "text/html; charset=utf-8")
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's spelling
        if self.path != "/read":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_json(400, {"error": "no image in the request body"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._send_json(
                413,
                {
                    "error": f"image is larger than "
                    f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
                },
            )
            return

        payload = self.rfile.read(length)
        try:
            self._send_json(200, self.demo.read_bytes(payload))
        except OSError:
            # PIL raises this for anything it cannot decode as an image.
            self._send_json(400, {"error": "that file is not an image"})
        except Exception:  # noqa: BLE001 -- a demo should not die on one photo
            traceback.print_exc()
            self._send_json(500, {"error": "the pipeline failed on that image"})


def serve(reader: Reader, *, host: str = "127.0.0.1", port: int = 7860) -> None:
    """Serve the demo until interrupted."""
    handler = type("BoundHandler", (Handler,), {"demo": Demo(reader, threading.Lock())})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"  serving on http://{host}:{server.server_port}")
    print("  ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        server.server_close()
