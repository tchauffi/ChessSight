"""The demo server, exercised with a stand-in for the pipeline.

The point of these is that they need neither torch nor a checkpoint: the
handler talks to anything with a ``read`` method, so the HTTP behaviour can be
tested on its own.
"""

from __future__ import annotations

import json
import threading
from io import BytesIO
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from PIL import Image

from chesssight.demo.render import board_svg, fit, jpeg_uri, overlay
from chesssight.demo.server import Demo, Handler, page_html

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


class FakeReader:
    """Returns a fixed reading, so the tests measure the server, not a model."""

    def __init__(self, *, found: bool = True) -> None:
        self.found = found
        self.seen: list[tuple[int, int]] = []

    def read(self, image: Image.Image) -> dict[str, Any]:
        self.seen.append(image.size)
        if not self.found:
            return {"corners": None, "grid": None, "fen": None, "detections": []}
        return {
            "corners": [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
            "grid": [[1] * 8] + [[0] * 8] * 7,
            "fen": START_FEN,
            "detections": [
                {"name": "board", "score": 0.9, "box": [5.0, 5.0, 95.0, 95.0]},
                {"name": "white_pawn", "score": 0.8, "box": [20.0, 20.0, 30.0, 40.0]},
                {"name": "black_king", "score": 0.7, "box": [60.0, 20.0, 70.0, 45.0]},
            ],
        }


def photo(size: tuple[int, int] = (200, 150)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (120, 130, 110)).save(buffer, "JPEG")
    return buffer.getvalue()


def test_fit_downscales_and_reports_its_factor() -> None:
    image = Image.new("RGB", (900, 450))
    scaled, factor = fit(image, 300)
    assert scaled.size == (300, 150)
    assert factor == pytest.approx(1 / 3)


def test_fit_leaves_a_small_image_alone() -> None:
    image = Image.new("RGB", (120, 80))
    scaled, factor = fit(image, 300)
    assert scaled.size == (120, 80)
    assert factor == 1.0


def test_overlay_draws_without_corners() -> None:
    # A reading with no board still returns an overlay; it just has no outline.
    image = Image.new("RGB", (100, 100))
    drawn = overlay(image, FakeReader().read(image)["detections"], None, width=100)
    assert drawn.size == (100, 100)


def test_board_svg_is_in_the_demo_colours() -> None:
    svg = board_svg(START_FEN)
    assert svg.startswith("<svg")
    # The default python-chess board is orange; this one must not be.
    assert "#ffce9e" not in svg
    assert "#e6e8df" in svg


def test_jpeg_uri_round_trips() -> None:
    uri = jpeg_uri(Image.new("RGB", (10, 10)))
    assert uri.startswith("data:image/jpeg;base64,")


def test_read_bytes_reports_a_position() -> None:
    demo = Demo(FakeReader(), threading.Lock())
    response = demo.read_bytes(photo())

    assert response["found"] is True
    assert response["fen"] == START_FEN
    assert response["pieces"] == 2  # the board box is not a piece
    assert response["on_board"] == 8  # occupied squares in the fake grid
    assert response["size"] == [200, 150]
    assert response["overlay"].startswith("data:image/jpeg;base64,")
    assert response["diagram"].startswith("<svg")


def test_read_bytes_says_so_when_there_is_no_board() -> None:
    demo = Demo(FakeReader(found=False), threading.Lock())
    response = demo.read_bytes(photo())

    assert response["found"] is False
    assert response["fen"] is None
    assert "diagram" not in response
    # An overlay still comes back, so the page can show what was looked at.
    assert response["overlay"].startswith("data:image/jpeg;base64,")


def test_page_html_is_a_document() -> None:
    assert page_html().lstrip().startswith(b"<!doctype html>")


@pytest.fixture
def server() -> Any:
    from http.server import ThreadingHTTPServer

    reader = FakeReader()
    handler = type("BoundHandler", (Handler,), {"demo": Demo(reader, threading.Lock())})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", reader
    httpd.shutdown()
    httpd.server_close()


def test_get_serves_the_page(server: Any) -> None:
    base, _ = server
    with urlopen(base + "/") as response:
        assert response.status == 200
        assert b"ChessSight" in response.read()


def test_post_reads_an_image(server: Any) -> None:
    base, reader = server
    with urlopen(base + "/read", data=photo((320, 240))) as response:
        payload = json.load(response)

    assert payload["found"] is True
    assert payload["fen"] == START_FEN
    assert reader.seen == [(320, 240)]


def test_post_rejects_a_non_image(server: Any) -> None:
    base, _ = server
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/read", data=b"not a picture")
    assert error.value.code == 400


def test_unknown_paths_are_404(server: Any) -> None:
    base, _ = server
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/elsewhere")
    assert error.value.code == 404
