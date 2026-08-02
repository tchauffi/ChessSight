"""Pre-normalise an external chess set into a compact, render-ready one.

Importing a print-ready set costs far more than rendering it: the STLs behind a
Staunton set run to hundreds of megabytes and hundreds of thousands of triangles per
piece, and :mod:`chesssight.blender.entry` resets the scene between jobs, so that
cost is paid *per image* rather than once.

Baking does the expensive work once -- import, decimate, orient, scale into board
units, re-centre the origin -- and writes the result as small OBJ files with a
manifest that needs no further processing.

Run via::

    blender --background --factory-startup \
            --python src/chesssight/blender/bake_set.py \
            -- --manifest IN/manifest.json --out OUT
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import bpy  # noqa: E402

from chesssight.blender import assets, bl_utils  # noqa: E402
from chesssight.synth import asset_spec  # noqa: E402

FILENAMES = {
    "P": "pawn",
    "N": "knight",
    "B": "bishop",
    "R": "rook",
    "Q": "queen",
    "K": "king",
}


def bake(
    manifest_path: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    decimate: float | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = asset_spec.load_manifest(manifest_path)
    if decimate is not None:
        manifest = {**manifest, "decimate": decimate}
    provider = assets.AssetLibraryProvider(manifest, manifest_path.parent)

    entries: dict[str, dict] = {}
    king_height = 0.0

    for letter in asset_spec.PIECE_LETTERS:
        bl_utils.reset_scene()
        # Re-create the provider's cache per piece so a reset scene cannot leave it
        # holding freed datablocks.
        provider._cache.clear()
        obj = provider._import_piece(letter)
        bl_utils.link(obj)

        for other in bpy.context.scene.objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        height = obj.dimensions.z
        if letter == "K":
            king_height = height

        path = out_dir / f"{FILENAMES[letter]}.obj"
        bpy.ops.wm.obj_export(
            filepath=str(path),
            export_selected_objects=True,
            export_materials=False,
            forward_axis="Y",
            up_axis="Z",
        )
        entries[letter] = {"file": path.name}
        print(
            f"[chesssight] baked {letter}: {len(obj.data.polygons)} faces, "
            f"{height:.3f} squares tall -> {path.name}",
            flush=True,
        )

    baked = {
        "name": manifest["name"],
        # Provenance has to survive the bake: a baked set is what actually gets
        # rendered, so if the licence and credit stop here they stop everywhere.
        "source": manifest.get("source"),
        "license": manifest.get("license"),
        "attribution": manifest.get("attribution"),
        # Already oriented, decimated and scaled into board units, so the importer
        # has nothing left to do beyond reading the file.
        "forward_axis": "+Y",
        "up_axis": "+Z",
        "scale_mode": "uniform",
        "king_height": round(king_height, 6),
        "pieces": entries,
        "baked_from": str(manifest_path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(baked, indent=2) + "\n", encoding="utf-8"
    )
    return baked


def main() -> int:
    if "--" not in sys.argv:
        raise SystemExit("expected arguments after `--`")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--decimate",
        type=float,
        default=None,
        help="Override the manifest's face-keep fraction.",
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    out_dir = pathlib.Path(args.out).expanduser()
    bake(pathlib.Path(args.manifest).expanduser(), out_dir, decimate=args.decimate)
    print(f"[chesssight] wrote {out_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
