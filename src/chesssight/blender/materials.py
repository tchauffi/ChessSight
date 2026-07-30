"""Procedural materials.

Node names follow the Principled BSDF v2 socket set shipped in Blender 5.2 --
``Specular IOR Level`` and ``Coat Weight`` rather than the pre-4.0 ``Specular`` and
``Clearcoat``. ``ShaderNodeTexMusgrave`` was removed in 4.1, so wood grain is built
from ``ShaderNodeTexNoise`` and ``ShaderNodeTexWave`` instead.
"""

from __future__ import annotations

import bpy

RGB = tuple[float, float, float]


def _principled(material: bpy.types.Material) -> bpy.types.Node:
    return next(
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    )


def new_material(name: str) -> tuple[bpy.types.Material, bpy.types.Node]:
    """Create a node-based material and return it with its Principled BSDF."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    return material, _principled(material)


def _object_noise(
    tree: bpy.types.NodeTree, *, scale: float, detail: float = 4.0
) -> bpy.types.Node:
    """A noise texture in object space, so it does not swim with the camera."""
    coords = tree.nodes.new("ShaderNodeTexCoord")
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    tree.links.new(coords.outputs["Object"], noise.inputs["Vector"])
    return noise


def vary_roughness(
    material: bpy.types.Material,
    *,
    base: float,
    amount: float = 0.12,
    scale: float = 30.0,
) -> None:
    """Break a constant roughness up with procedural variation.

    A single roughness value across a whole surface is one of the strongest tells
    that an image was rendered: it makes every specular highlight the same size and
    sharpness, so the light reads as a clean mathematical lobe rather than as a real
    surface. Real objects vary -- a moulded piece has glossier and duller patches
    from the mould and from handling, and varnish is never laid down evenly.
    """
    tree = material.node_tree
    noise = _object_noise(tree, scale=scale)
    spread = tree.nodes.new("ShaderNodeMapRange")
    spread.inputs["From Min"].default_value = 0.25
    spread.inputs["From Max"].default_value = 0.75
    spread.inputs["To Min"].default_value = max(0.02, base - amount)
    spread.inputs["To Max"].default_value = min(1.0, base + amount)
    spread.clamp = True
    tree.links.new(noise.outputs["Fac"], spread.inputs["Value"])
    tree.links.new(spread.outputs["Result"], _principled(material).inputs["Roughness"])


def add_bump(
    material: bpy.types.Material, *, strength: float = 0.08, scale: float = 90.0
) -> None:
    """Give a surface fine relief.

    Everything here is built from lathes and quads, so without this every surface is
    geometrically perfect to the pixel -- no turning marks on the pieces, no grain
    standing up on the board, no orange peel in the varnish. The eye reads that
    perfection as synthetic long before it notices anything about the shapes.
    """
    tree = material.node_tree
    noise = _object_noise(tree, scale=scale, detail=6.0)
    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = 0.01
    tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], _principled(material).inputs["Normal"])


def solid(
    name: str,
    color: RGB,
    *,
    roughness: float = 0.4,
    metallic: float = 0.0,
    coat: float = 0.0,
) -> bpy.types.Material:
    """A plain Principled material."""
    material, bsdf = new_material(name)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Coat Weight"].default_value = coat
    return material


def wood(
    name: str,
    color: RGB,
    *,
    roughness: float = 0.4,
    grain_scale: float = 12.0,
    grain_strength: float = 0.12,
    coat: float = 0.0,
) -> bpy.types.Material:
    """A wood-ish material: stretched noise darkens the base colour into grain.

    Cheap in both engines and enough to break up the flat colour that makes
    synthetic renders obvious.
    """
    material, bsdf = new_material(name)
    tree = material.node_tree

    coords = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    # Squash along one axis so the noise reads as directional grain, not blobs.
    mapping.inputs["Scale"].default_value = (
        grain_scale,
        grain_scale * 0.06,
        grain_scale,
    )

    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.55

    ramp = tree.nodes.new("ShaderNodeValToRGB")
    dark = tuple(max(0.0, channel * (1.0 - grain_strength)) for channel in color)
    ramp.color_ramp.elements[0].color = (*dark, 1.0)
    ramp.color_ramp.elements[1].color = (*color, 1.0)

    tree.links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Coat Weight"].default_value = coat
    # Grain that only changes colour reads as a printed decal. Standing it up as
    # relief and letting the varnish pool unevenly over it is what makes it wood.
    vary_roughness(material, base=roughness, amount=0.10, scale=grain_scale * 2.0)
    add_bump(material, strength=0.06, scale=grain_scale * 6.0)
    return material


def piece_material(
    name: str, color: RGB, *, roughness: float = 0.35
) -> bpy.types.Material:
    """Turned-and-lacquered look for the pieces.

    A little coat and subsurface keeps light pieces from reading as flat plastic,
    which is what makes synthetic boxwood look wrong.
    """
    material, bsdf = new_material(name)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Coat Weight"].default_value = 0.25
    bsdf.inputs["Coat Roughness"].default_value = 0.15
    bsdf.inputs["Subsurface Weight"].default_value = 0.06
    bsdf.inputs["Subsurface Radius"].default_value = (0.3, 0.2, 0.12)
    # Pieces are the most scrutinised surface in the frame and the one a detector
    # has to tell apart by shape, so they get the finest relief: turning marks from
    # the lathe, and patches worn glossy where a piece is picked up.
    vary_roughness(material, base=roughness, amount=0.13, scale=45.0)
    add_bump(material, strength=0.05, scale=160.0)
    return material


def assign(obj: bpy.types.Object, *materials: bpy.types.Material) -> None:
    """Replace an object's material slots."""
    obj.data.materials.clear()
    for material in materials:
        obj.data.materials.append(material)


def assign_object_level(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    """Override an object's material without touching its mesh.

    This is what lets one lathed mesh serve both the white and the black set: the
    mesh datablock is shared by linked duplicates, and only the object-level slot
    differs.
    """
    if not obj.data.materials:
        obj.data.materials.append(None)
    obj.material_slots[0].link = "OBJECT"
    obj.material_slots[0].material = material
