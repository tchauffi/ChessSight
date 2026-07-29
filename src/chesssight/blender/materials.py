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

    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = coat
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
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = 0.25
    bsdf.inputs["Coat Roughness"].default_value = 0.15
    bsdf.inputs["Subsurface Weight"].default_value = 0.06
    bsdf.inputs["Subsurface Radius"].default_value = (0.3, 0.2, 0.12)
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
