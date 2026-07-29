"""Import an external chess set and normalise it to the pipeline's conventions.

A downloaded set arrives in whatever units, orientation and origin its author used.
The renderer and the label pass both assume the contract in
:class:`chesssight.blender.pieces.PieceProvider`: origin at the base centre, standing
on ``z = 0``, sized in board squares, facing ``+Y``. Everything needed to get from
one to the other is done here, once, at import.

The importers are all present in Blender 5.2 (verified): ``wm.obj_import``,
``import_scene.gltf``, ``wm.fbx_import``, ``wm.collada_import``, ``wm.stl_import``,
``wm.usd_import`` and ``wm.append`` for ``.blend`` files.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

from chesssight.blender import bl_utils, pieces
from chesssight.synth import asset_spec
from chesssight.synth.asset_spec import AssetError


def _existing_objects() -> set[str]:
    return {obj.name for obj in bpy.data.objects}


def _import_file(path: Path, kind: str) -> list[bpy.types.Object]:
    """Import a file and return only the objects it added."""
    before = _existing_objects()
    filepath = str(path)

    if kind == "obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif kind == "gltf":
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif kind == "fbx":
        bpy.ops.wm.fbx_import(filepath=filepath)
    elif kind == "collada":
        bpy.ops.wm.collada_import(filepath=filepath)
    elif kind == "stl":
        bpy.ops.wm.stl_import(filepath=filepath)
    elif kind == "usd":
        bpy.ops.wm.usd_import(filepath=filepath)
    else:
        raise AssetError(f"unsupported import kind {kind!r} for {path}")

    return [obj for obj in bpy.data.objects if obj.name not in before]


def _append_from_blend(path: Path, object_name: str) -> list[bpy.types.Object]:
    before = _existing_objects()
    bpy.ops.wm.append(
        filepath=str(path / "Object" / object_name),
        directory=str(path) + "/Object/",
        filename=object_name,
    )
    added = [obj for obj in bpy.data.objects if obj.name not in before]
    for obj in added:
        if obj.name not in {o.name for o in bpy.context.scene.objects}:
            bl_utils.link(obj)
    return added


def _select_meshes(
    added: list[bpy.types.Object], wanted: str | None, source: str
) -> list[bpy.types.Object]:
    """Pick the mesh objects that make up one piece, discarding the rest."""
    meshes = [obj for obj in added if obj.type == "MESH"]
    if not meshes:
        raise AssetError(f"{source} contained no mesh objects")

    if wanted:
        # glTF and FBX importers often suffix names on collision, so match loosely.
        matched = [
            obj
            for obj in meshes
            if obj.name == wanted or obj.name.startswith(f"{wanted}.")
        ]
        if not matched:
            available = ", ".join(sorted(obj.name for obj in meshes))
            raise AssetError(
                f"{source}: no object named {wanted!r}; it contains: {available}"
            )
        meshes = matched

    # Anything imported but not wanted (cameras, lights, other pieces from a set
    # packed into one file) must go, or it would be rendered and get a mask id.
    for obj in added:
        if obj not in meshes:
            bpy.data.objects.remove(obj, do_unlink=True)
    return meshes


def _combine(meshes: list[bpy.types.Object], name: str) -> bpy.types.Object:
    target = meshes[0]
    target.name = name
    if len(meshes) > 1:
        bl_utils.join(target, meshes[1:])
    return target


def _decimate(obj: bpy.types.Object, ratio: float) -> None:
    """Reduce triangle count in place."""
    modifier = obj.modifiers.new("Decimate", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = ratio
    bl_utils.apply_modifiers(obj)


def _orient(obj: bpy.types.Object, forward_axis: str, up_axis: str) -> None:
    """Stand a piece upright and turn it to face the opponent."""
    for other in bpy.context.scene.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.rotation_euler = (
        math.radians(asset_spec.up_pitch(up_axis)),
        0.0,
        math.radians(asset_spec.forward_yaw(forward_axis)),
    )
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)


def _normalise(obj: bpy.types.Object, *, scale_factor: float) -> None:
    """Scale an already-oriented piece into board units and centre its base.

    Orientation happens earlier, in ``_load_mesh``, because the scale factor has to
    be measured along the *upright* axis. Rotating after scaling would give pieces
    sized by their width rather than their height.
    """
    for other in bpy.context.scene.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 3. Put the origin at the centre of the base, standing on z = 0. The mesh data
    #    is moved rather than the object, so the object transform stays identity and
    #    a placement can set `location` directly.
    obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()

    corners = [Vector(corner) for corner in obj.bound_box]
    lowest = min(corner.z for corner in corners)
    center_x = (
        min(corner.x for corner in corners) + max(corner.x for corner in corners)
    ) / 2.0
    center_y = (
        min(corner.y for corner in corners) + max(corner.y for corner in corners)
    ) / 2.0

    obj.data.transform(Matrix.Translation((-center_x, -center_y, -lowest)))
    obj.data.update()


class AssetLibraryProvider:
    """Builds pieces from an external set described by a manifest.

    Registered under the manifest's own name, so a config can select it with
    ``pieces.provider: "<manifest name>"`` and nothing else in the renderer changes.
    """

    def __init__(self, manifest: dict, root: Path) -> None:
        self.manifest = manifest
        self.root = Path(root)
        self.name = manifest["name"]
        self.forward_axis = manifest.get("forward_axis", "+Y")
        self.up_axis = manifest.get("up_axis", "+Z")
        self.scale_mode = asset_spec.scale_mode(manifest)
        self.decimate = asset_spec.decimate_ratio(manifest)
        self._cache: dict[str, bpy.types.Object] = {}
        self._uniform_factor: float | None = None

    def _load_mesh(self, letter: str) -> bpy.types.Object:
        """Import one piece's raw geometry, oriented but not yet scaled."""
        asset = self.manifest["pieces"][letter]
        path = (self.root / asset["file"]).resolve()
        if not path.is_file():
            raise AssetError(f"{letter}: missing file {path}")

        object_name = asset.get("object")
        kind = asset_spec.kind_for_file(asset["file"], self.manifest.get("kind"))
        if kind == "blend":
            if not object_name:
                raise AssetError(f"{letter}: a .blend asset needs an `object` name")
            added = _append_from_blend(path, object_name)
        else:
            added = _import_file(path, kind)

        meshes = _select_meshes(added, object_name, f"{path.name} ({letter})")
        obj = _combine(meshes, f"Asset_{letter}")
        if self.decimate:
            _decimate(obj, self.decimate)
        _orient(obj, self.forward_axis, self.up_axis)
        return obj

    def _scale_factor(self, letter: str, obj: bpy.types.Object) -> float:
        """Factor taking an oriented piece from file units into board squares."""
        height = obj.dimensions.z
        if height <= 1e-9:
            raise AssetError(f"{letter}: zero height after orienting; check `up_axis`")

        if self.scale_mode == "per_piece":
            return asset_spec.target_height(self.manifest, letter) / height

        # Uniform: measure the king once, then apply that factor to everything, so
        # the set keeps the proportions it was designed with.
        if self._uniform_factor is None:
            if letter == "K":
                king_dimension = height
            else:
                probe = self._load_mesh("K")
                king_dimension = probe.dimensions.z
                bl_utils.unlink(probe)
                bpy.data.objects.remove(probe, do_unlink=True)
            if king_dimension <= 1e-9:
                raise AssetError("K: zero height after orienting; check `up_axis`")
            self._uniform_factor = (
                asset_spec.king_height(self.manifest) / king_dimension
            )
        return self._uniform_factor

    def _import_piece(self, letter: str) -> bpy.types.Object:
        obj = self._load_mesh(letter)
        _normalise(obj, scale_factor=self._scale_factor(letter, obj))
        # Imported sets bring their own materials; drop them so the scene's colour
        # randomisation applies uniformly, and leave one slot for the override.
        obj.data.materials.clear()
        obj.data.materials.append(None)

        # The cache is a template, not scene content. Leaving it linked puts an
        # untagged copy of every piece at the world origin -- an unlabelled pile in
        # the middle of the board, and a bogus id in the mask pass.
        bl_utils.unlink(obj)
        return obj

    def build(
        self, letter: str, style: pieces.PieceStyle, rng: random.Random
    ) -> bpy.types.Object:
        """Return a prototype for ``letter``, scaled by the scene's style."""
        del rng  # an imported set has no procedural variation to draw

        if letter not in self._cache:
            self._cache[letter] = self._import_piece(letter)
        prototype = self._cache[letter]

        obj = prototype.copy()
        obj.data = prototype.data.copy()
        bl_utils.link(obj)  # only the instance belongs in the scene

        # Apply the scene's style jitter on top, so an imported set still varies
        # between images rather than being pixel-identical every time.
        obj.scale = (
            style.square_size * style.radius_scale,
            style.square_size * style.radius_scale,
            style.square_size * style.height_scale,
        )
        for other in bpy.context.scene.objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return obj


def register_manifest(path: Path) -> AssetLibraryProvider:
    """Load a manifest and register its provider, returning it."""
    manifest_path = Path(path).expanduser()
    manifest = asset_spec.load_manifest(manifest_path)
    provider = AssetLibraryProvider(manifest, manifest_path.parent)
    pieces.register_provider(provider)
    return provider
