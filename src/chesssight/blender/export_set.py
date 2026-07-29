"""Export the procedural set as OBJ files plus a manifest.

Two uses. It gives a working, correctly-oriented example of the manifest format to
start from when adapting a downloaded set. And it exercises the import path
end-to-end -- exporting the built-in pieces and reading them back through
:mod:`chesssight.blender.assets` proves the normalisation is right without needing
any third-party asset.

Run via::

    blender --background --factory-startup \
            --python src/chesssight/blender/export_set.py -- --out DIR
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bpy  # noqa: E402

from chesssight.blender import bl_utils, pieces  # noqa: E402
from chesssight.synth import profiles  # noqa: E402

PIECE_FILENAMES = {
    "P": "pawn",
    "N": "knight",
    "B": "bishop",
    "R": "rook",
    "Q": "queen",
    "K": "king",
}


def export(out_dir: pathlib.Path, *, segments: int = 48, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    style = pieces.PieceStyle(
        square_size=1.0,
        height_scale=1.0,
        radius_scale=1.0,
        bevel_width=0.006,
        lathe_segments=segments,
    )
    provider = pieces.ProceduralProvider()
    entries = {}

    for letter in profiles.PIECE_LETTERS:
        bl_utils.reset_scene()
        obj = provider.build(letter, style, random.Random(seed))

        for other in bpy.context.scene.objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        path = out_dir / f"{PIECE_FILENAMES[letter]}.obj"
        bpy.ops.wm.obj_export(
            filepath=str(path),
            export_selected_objects=True,
            export_materials=False,
            # Keep Blender's own axes so the manifest below is the identity case.
            forward_axis="Y",
            up_axis="Z",
        )
        entries[letter] = {"file": path.name}
        print(f"[chesssight] exported {letter} -> {path.name}", flush=True)

    manifest = {
        "name": "procedural-export",
        "source": "chesssight built-in procedural set",
        "license": "same as this repository",
        "forward_axis": "+Y",
        "up_axis": "+Z",
        "king_height": profiles.PIECE_HEIGHTS["K"],
        "pieces": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit("expected arguments after `--`, e.g. -- --out DIR")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--segments", type=int, default=48)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    out_dir = pathlib.Path(args.out).expanduser()
    export(out_dir, segments=args.segments)
    print(f"[chesssight] wrote manifest to {out_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
