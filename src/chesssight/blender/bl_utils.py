"""Small helpers over the Blender data API.

``bpy.ops`` is avoided wherever a ``bpy.data`` equivalent exists: operators depend on
context that is awkward in background mode, and they are considerably slower when
called tens of thousands of times.
"""

from __future__ import annotations

import math

import bpy
from mathutils import Euler, Matrix, Vector

#: Custom-property keys used to tag objects so the label pass can find them again.
ROLE_KEY = "cs_role"
INSTANCE_KEY = "cs_instance_id"
CLASS_KEY = "cs_class_id"
RANK_KEY = "cs_rank"
FILE_KEY = "cs_file"
CORNER_KEY = "cs_corner_id"
CAPTURED_KEY = "cs_captured"

#: Role codes written into the green channel of the id pass.
ROLE_CODES = {
    "piece": 1,
    "board": 2,
    "table": 3,
    "backdrop": 4,
    "distractor": 5,
}


def reset_scene() -> None:
    """Return Blender to a completely empty scene.

    Called between jobs in a batch so one render cannot leak geometry, materials or
    settings into the next.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)


def purge_orphans() -> None:
    """Drop datablocks with no users, keeping memory flat across a long batch."""
    for _ in range(3):
        removed = False
        for collection in (
            bpy.data.meshes,
            bpy.data.materials,
            bpy.data.lights,
            bpy.data.cameras,
            bpy.data.images,
            bpy.data.node_groups,
        ):
            for datablock in list(collection):
                if datablock.users == 0:
                    collection.remove(datablock)
                    removed = True
        if not removed:
            break


def link(obj: bpy.types.Object) -> bpy.types.Object:
    """Link an object into the active scene collection."""
    bpy.context.scene.collection.objects.link(obj)
    return obj


def unlink(obj: bpy.types.Object) -> bpy.types.Object:
    """Remove an object from every collection, keeping its datablock alive.

    This is how piece *prototypes* are kept out of the render. Merely setting
    ``hide_render`` is not enough to rely on: a prototype that stays in the scene is
    one property away from being drawn at the world origin, where it appears as an
    unlabelled pile in the middle of the board and takes an instance id in the mask
    pass that belongs to nothing. Unlinking makes that impossible rather than
    unlikely.
    """
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    return obj


def renderable_meshes() -> list[bpy.types.Object]:
    """Mesh objects that will actually appear in a render."""
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.hide_render
    ]


def assert_all_tagged() -> None:
    """Fail if anything renderable is missing its role tag.

    Untagged geometry is worse than a visual glitch: it appears in the image with no
    label, and the id pass decodes its default white ``object.color`` as instance
    255, which is not a piece. Better to lose one sample loudly than to poison a
    dataset quietly.
    """
    untagged = [obj.name for obj in renderable_meshes() if ROLE_KEY not in obj]
    if untagged:
        raise RuntimeError(
            f"untagged renderable objects would be rendered without labels: "
            f"{sorted(untagged)}"
        )


def new_mesh_object(
    name: str,
    vertices: list[tuple[float, float, float]],
    edges: list[tuple[int, int]],
    faces: list[tuple[int, ...]],
) -> bpy.types.Object:
    """Build an object from raw geometry and link it into the scene."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, edges, faces)
    mesh.update()
    return link(bpy.data.objects.new(name, mesh))


def tag(
    obj: bpy.types.Object,
    role: str,
    *,
    instance_id: int = 0,
    class_id: int = 0,
    rank_index: int | None = None,
    file_index: int | None = None,
) -> bpy.types.Object:
    """Tag an object for the label and id passes.

    ``instance_id`` goes into the red channel of the id pass and must be unique
    within a scene; ``role`` goes into the green channel.
    """
    if role not in ROLE_CODES:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(ROLE_CODES)}")
    obj[ROLE_KEY] = role
    obj[INSTANCE_KEY] = instance_id
    obj[CLASS_KEY] = class_id
    if rank_index is not None:
        obj[RANK_KEY] = rank_index
    if file_index is not None:
        obj[FILE_KEY] = file_index
    # object.color is what the Workbench id pass reads; alpha must stay 1.0.
    obj.color = (instance_id / 255.0, ROLE_CODES[role] / 255.0, 0.0, 1.0)
    return obj


def objects_with_role(role: str) -> list[bpy.types.Object]:
    """Every linked object tagged with ``role``, in scene order."""
    return [obj for obj in bpy.context.scene.objects if obj.get(ROLE_KEY) == role]


def look_at(
    obj: bpy.types.Object,
    target: Vector | tuple[float, float, float],
    roll_deg: float = 0.0,
) -> None:
    """Aim an object's -Z axis at ``target``, then roll it about its own axis.

    Used for the camera. A track-to constraint would also work but would have to be
    evaluated before the label pass could read the camera matrix, which is an easy
    source of one-frame-stale bugs.
    """
    direction = Vector(target) - obj.location
    if direction.length < 1e-9:
        raise ValueError("cannot aim an object at its own location")
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    if roll_deg:
        rolled = obj.rotation_euler.to_matrix() @ Matrix.Rotation(
            math.radians(roll_deg), 3, "Z"
        )
        obj.rotation_euler = rolled.to_euler()


def apply_modifiers(obj: bpy.types.Object) -> None:
    """Evaluate and bake in an object's modifier stack.

    Needed before joining meshes: joining leaves each source object's modifiers
    behind, which would silently drop the lathe.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated)
    old_mesh = obj.data
    obj.modifiers.clear()
    obj.data = mesh
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def join(target: bpy.types.Object, others: list[bpy.types.Object]) -> bpy.types.Object:
    """Join ``others`` into ``target``, applying their modifiers first."""
    if not others:
        return target
    for obj in [target, *others]:
        apply_modifiers(obj)

    bpy.context.view_layer.objects.active = target
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    target.select_set(True)
    for obj in others:
        obj.select_set(True)
    bpy.ops.object.join()
    return target


def set_euler(
    obj: bpy.types.Object, x: float = 0.0, y: float = 0.0, z: float = 0.0
) -> None:
    """Set an object's rotation from degrees."""
    obj.rotation_euler = Euler(
        (math.radians(x), math.radians(y), math.radians(z)), "XYZ"
    )
