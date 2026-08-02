"""The local demo: drop a photograph in a browser, get the position back."""

from __future__ import annotations

from chesssight.demo.render import board_svg, overlay
from chesssight.demo.server import serve

__all__ = ["board_svg", "overlay", "serve"]
