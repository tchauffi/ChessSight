"""Procedural materials.

Node names follow the Principled BSDF v2 socket set shipped in Blender 5.2 --
``Specular IOR Level`` and ``Coat Weight`` rather than the pre-4.0 ``Specular`` and
``Clearcoat``. ``ShaderNodeTexMusgrave`` was removed in 4.1, so wood grain is built
from ``ShaderNodeTexNoise`` and ``ShaderNodeTexWave`` instead.
"""

from __future__ import annotations

import math

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


def add_bevel(material: bpy.types.Material, *, radius: float = 0.004) -> None:
    """Round the shading normal along edges, without touching geometry.

    A perfect 90-degree edge is the most reliable tell that an image was rendered.
    Nothing manufactured has one: a board is chamfered or at least sanded, and even
    a sharp arris catches a thin highlight. The Bevel node fakes that by bending the
    shading normal near an edge, so the highlight appears with no extra polygons --
    and, importantly here, no change to the silhouette the masks and boxes come
    from. A real bevel modifier would alter geometry and therefore the labels.

    Chains into whatever already drives the normal, so it composes with
    :func:`add_bump` rather than silently replacing its relief.
    """
    tree = material.node_tree
    bsdf = _principled(material)
    bevel = tree.nodes.new("ShaderNodeBevel")
    bevel.inputs["Radius"].default_value = radius
    bevel.samples = 2

    normal_input = bsdf.inputs["Normal"]
    if normal_input.is_linked:
        upstream = normal_input.links[0].from_node
        if "Normal" in upstream.inputs:
            tree.links.new(bevel.outputs["Normal"], upstream.inputs["Normal"])
            return
    tree.links.new(bevel.outputs["Normal"], normal_input)


def vary_per_instance(
    material: bpy.types.Material,
    *,
    value: float = 0.10,
    saturation: float = 0.08,
    roughness: float = 0.06,
) -> None:
    """Give every object sharing this material its own slight tint and finish.

    Pieces are instances of one prototype with one material, so without this a rank
    of pawns is *pixel-identical* eight times over. Real sets never are: timber
    varies board to board, moulded pieces vary batch to batch, and handling wears
    them unevenly. That repetition is a strong part of why a render reads as
    synthetic even when every individual object looks right.

    ``ObjectInfo.Random`` is a stable per-object value, so the variation is
    consistent for an object across the frame rather than flickering per sample, and
    it costs no extra materials.
    """
    tree = material.node_tree
    bsdf = _principled(material)
    info = tree.nodes.new("ShaderNodeObjectInfo")

    base_input = bsdf.inputs["Base Color"]
    hue = tree.nodes.new("ShaderNodeHueSaturation")
    if base_input.is_linked:
        source = base_input.links[0]
        tree.links.new(source.from_socket, hue.inputs["Color"])
        tree.links.remove(source)
    else:
        hue.inputs["Color"].default_value = base_input.default_value
    tree.links.new(hue.outputs["Color"], base_input)

    # Random is in [0, 1); centre it so instances vary either side of the nominal
    # colour rather than all drifting one way.
    for socket, amount in (("Value", value), ("Saturation", saturation)):
        spread = tree.nodes.new("ShaderNodeMapRange")
        spread.inputs["To Min"].default_value = 1.0 - amount
        spread.inputs["To Max"].default_value = 1.0 + amount
        spread.clamp = True
        tree.links.new(info.outputs["Random"], spread.inputs["Value"])
        tree.links.new(spread.outputs["Result"], hue.inputs[socket])

    if roughness:
        rough_input = bsdf.inputs["Roughness"]
        # Only when roughness is a plain value; if a procedural already drives it,
        # leave that alone rather than discarding the variation it provides.
        if not rough_input.is_linked:
            spread = tree.nodes.new("ShaderNodeMapRange")
            base = rough_input.default_value
            spread.inputs["To Min"].default_value = max(0.02, base - roughness)
            spread.inputs["To Max"].default_value = min(1.0, base + roughness)
            spread.clamp = True
            tree.links.new(info.outputs["Random"], spread.inputs["Value"])
            tree.links.new(spread.outputs["Result"], rough_input)


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


def _object_mapping(tree, *, scale: float, rotation: tuple = (0.0, 0.0, 0.0)):
    """Object-space coordinates, scaled and rotated. The input to every procedural.

    Object space rather than generated or UV: a lathed piece has no sensible UV
    layout, and generated coordinates are bounding-box relative, so two pieces of
    different heights would get differently-stretched grain from the same settings.
    """
    coords = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    mapping.inputs["Rotation"].default_value = rotation
    tree.links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    return mapping


def _warp(
    tree: bpy.types.NodeTree,
    vector,
    *,
    scale: float,
    strength: float,
    detail: float = 6.0,
    noise_type: str = "FBM",
):
    """Displace a coordinate field by 3D noise -- *domain warping*.

    This is the technique that separates a procedural that looks computed from one
    that looks grown. Instead of perturbing the *output* of a pattern (which is what
    the Wave node's own Distortion does, and it reads as a uniform wobble), the
    coordinate itself is pushed around before the pattern is evaluated. Bands then
    stretch, pinch and drift the way figure does in a real solid, because the whole
    space they live in is deformed rather than the stripes being shaken.

    Uses the noise's ``Color`` output, which is three decorrelated channels, so the
    displacement is a genuine 3D vector rather than the same scalar on every axis --
    the latter only slides the pattern diagonally and looks like nothing much.
    """
    noise = tree.nodes.new("ShaderNodeTexNoise")
    noise.noise_type = noise_type
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    noise.inputs["Roughness"].default_value = 0.55
    tree.links.new(vector, noise.inputs["Vector"])

    # Noise Color is in [0,1]^3; centre it so the warp pushes both ways.
    centre = tree.nodes.new("ShaderNodeVectorMath")
    centre.operation = "SUBTRACT"
    centre.inputs[1].default_value = (0.5, 0.5, 0.5)
    tree.links.new(noise.outputs["Color"], centre.inputs[0])

    amount = tree.nodes.new("ShaderNodeVectorMath")
    amount.operation = "SCALE"
    amount.inputs["Scale"].default_value = strength
    tree.links.new(centre.outputs["Vector"], amount.inputs[0])

    warped = tree.nodes.new("ShaderNodeVectorMath")
    warped.operation = "ADD"
    tree.links.new(vector, warped.inputs[0])
    tree.links.new(amount.outputs["Vector"], warped.inputs[1])
    return warped.outputs["Vector"]


def marble(
    name: str,
    color: RGB,
    vein_color: RGB,
    *,
    scale: float = 3.0,
    distortion: float = 3.0,
    vein_width: float = 0.12,
    rotation: tuple = (0.0, 0.0, 0.0),
    roughness: float = 0.25,
    coat: float = 0.35,
) -> bpy.types.Material:
    """Veined marble: mostly plain stone, crossed occasionally by a thin vein.

    The proportions are the whole thing, and getting them wrong is what made the
    first attempt read as zebra camouflage. Real marble is roughly nine parts body
    stone to one part vein, so ``vein_width`` maps only the top sliver of the wave
    to the vein colour and everything below it to plain stone. A ramp spanning the
    full range instead paints half the surface dark, which is a pattern, not a rock.

    Distortion has to stay moderate for the same reason. It is what stops the bands
    reading as a barber's pole, but pushed hard it curls them into closed blobs and
    the veins stop looking like fractures running through a solid.
    """
    material, bsdf = new_material(name)
    tree = material.node_tree
    mapping = _object_mapping(tree, scale=scale, rotation=rotation)

    # One warp, at the scale of the form itself. A second finer warp was tried and
    # removed: it turned the veins into dense contour lines like a topographic map,
    # because warping an already-warped field at high frequency folds the bands back
    # over themselves many times within a single piece.
    # Strength is in *band periods*, because the warp is applied after the mapping
    # scale and the wave's period is about one unit there. Anything approaching 1.0
    # displaces the field by a whole band and folds it back on itself, which draws
    # concentric contour rings like a topographic map -- the failure this replaced.
    # A little under half a period bends a vein convincingly without folding it.
    warped = _warp(
        tree,
        mapping.outputs["Vector"],
        scale=1.2,
        strength=min(0.45, distortion * 0.09),
        detail=4.0,
    )

    wave = tree.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    # Diagonal rather than along an axis: axis-aligned bands on a lathed piece come
    # out as near-vertical lines running top to bottom, which read as drips down the
    # piece. A vein in cut stone crosses the form at whatever angle the slab was
    # sawn, and the per-scene rotation puts that angle somewhere different each time.
    wave.bands_direction = "DIAGONAL"
    wave.wave_profile = "SIN"
    # The node's own distortion is left off: it perturbs the output rather than the
    # coordinate, which reads as a uniform wobble on top of straight bands. The
    # domain warp above does the same job in a way that deforms the whole field.
    wave.inputs["Distortion"].default_value = 0.0
    wave.inputs["Detail"].default_value = 2.0
    tree.links.new(warped, wave.inputs["Vector"])

    # Body stone everywhere, vein only in the narrow band near the top of the wave.
    # Both stops sit high, so the vein is a minority of the surface however the
    # wave happens to fall.
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "EASE"
    ramp.color_ramp.elements[0].position = min(0.95, 1.0 - vein_width * 2.0)
    ramp.color_ramp.elements[0].color = (*color, 1.0)
    ramp.color_ramp.elements[1].position = min(1.0, 1.0 - vein_width * 0.4)
    ramp.color_ramp.elements[1].color = (*vein_color, 1.0)
    tree.links.new(wave.outputs["Factor"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Polished stone: a clear coat over a fairly smooth body, and a little
    # subsurface so light pieces do not read as painted plaster.
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.08
    bsdf.inputs["Subsurface Weight"].default_value = 0.08
    bsdf.inputs["Subsurface Radius"].default_value = (0.4, 0.35, 0.3)
    vary_roughness(material, base=roughness, amount=0.08, scale=60.0)
    return material


def wood_grain(
    name: str,
    color: RGB,
    *,
    dark_color: RGB | None = None,
    rings: bool = True,
    scale: float = 4.0,
    distortion: float = 2.0,
    rotation: tuple = (0.0, 0.0, 0.0),
    roughness: float = 0.35,
    coat: float = 0.25,
) -> bpy.types.Material:
    """Wood grain, as rings on a turned piece or as bands on a sawn flat surface.

    ``rings=True`` gives growth rings concentric with the object's Z axis, which is
    the axis a piece was turned on -- the grain then wraps a lathed profile the way
    it does on the real thing. ``rings=False`` gives the straight, wandering bands
    of a sawn board, which is what a flat square needs: rings on a flat surface come
    out as a bullseye, and that was plainly visible on the first boards rendered.

    The contrast is deliberately low. Grain is a *modulation* of one timber colour,
    perhaps 15-30% between earlywood and latewood; the first version darkened by
    half and produced something closer to a barcode than to wood. An EASE ramp keeps
    the transition soft, because a linear one gives every ring a hard edge.
    """
    material, bsdf = new_material(name)
    tree = material.node_tree
    mapping = _object_mapping(tree, scale=scale, rotation=rotation)

    # A much gentler warp than marble uses. Growth rings really are near-concentric,
    # so the coordinate is nudged, not dragged: at marble's strength the rings stop
    # reading as rings at all and the piece looks like polished stone instead of
    # timber. One warp only, for the same reason it is one on marble.
    warped = _warp(
        tree,
        mapping.outputs["Vector"],
        scale=1.6,
        strength=distortion * 0.10,
        detail=4.0,
    )

    grain = tree.nodes.new("ShaderNodeTexWave")
    grain.wave_type = "RINGS" if rings else "BANDS"
    grain.rings_direction = "Z"
    grain.bands_direction = "X"
    # A saw profile rather than a sine: earlywood fades gradually into latewood and
    # then the next ring starts abruptly. A sine is symmetric and reads as a painted
    # stripe; the asymmetry is most of what makes the eye call it timber.
    grain.wave_profile = "SAW"
    grain.inputs["Distortion"].default_value = 0.0
    grain.inputs["Detail"].default_value = 4.0
    grain.inputs["Detail Scale"].default_value = 2.0
    grain.inputs["Detail Roughness"].default_value = 0.6
    tree.links.new(warped, grain.inputs["Vector"])

    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "EASE"
    darker = dark_color or [max(0.0, channel * 0.78) for channel in color]
    ramp.color_ramp.elements[0].color = (*darker, 1.0)
    ramp.color_ramp.elements[1].color = (*color, 1.0)
    # Pulled apart from the ends so the extremes are rare: most of the surface sits
    # mid-ramp, as timber does, rather than alternating between two flat tones.
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[1].position = 0.85
    tree.links.new(grain.outputs["Factor"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.12
    bsdf.inputs["Subsurface Weight"].default_value = 0.05
    bsdf.inputs["Subsurface Radius"].default_value = (0.3, 0.2, 0.12)
    # Grain that only changes colour reads as a printed decal; standing it up as
    # relief is what makes the eye read it as figure in a solid.
    vary_roughness(material, base=roughness, amount=0.10, scale=50.0)
    add_bump(material, strength=0.04, scale=140.0)
    return material


def textured(
    name: str,
    maps: dict,
    *,
    scale: float = 1.0,
    rotation: float = 0.0,
    tint: RGB | None = None,
    roughness_shift: float = 0.0,
    projection: str = "FLAT",
    hue_shift: float = 0.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
) -> bpy.types.Material:
    """A photographed surface: diffuse, roughness and normal maps wired up.

    Procedural noise makes a surface non-uniform; it does not make it a material.
    Wood has figure that runs, joins between boards and wear along an edge, and none
    of that is reachable from a noise node -- which is why the table, the largest
    surface in frame after the board, read as flat colour no matter how much noise
    was layered on it.

    ``scale`` and ``rotation`` are per-scene randomisation: the same texture laid
    down at one size and angle for every image would just be a different constant
    for the detector to memorise. ``tint`` and ``roughness_shift`` push the same map
    across a range of finishes, so twelve downloads cover more than twelve tables.

    The normal map is loaded as non-colour data. Left as sRGB it is silently gamma
    decoded and the relief comes out shallow and wrongly angled -- a mistake that
    looks like a weak normal map rather than a broken one.
    """
    material, bsdf = new_material(name)
    tree = material.node_tree

    coords = tree.nodes.new("ShaderNodeTexCoord")
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, rotation)
    tree.links.new(coords.outputs["Object"], mapping.inputs["Vector"])

    def image_node(path: str, *, non_color: bool) -> bpy.types.Node:
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(path, check_existing=True)
        if non_color:
            node.image.colorspace_settings.name = "Non-Color"
        node.extension = "REPEAT"
        node.projection = projection
        if projection == "BOX":
            # Triplanar: the texture is projected down all three axes and blended by
            # surface normal. This is what allows a photographed texture on the
            # pieces at all -- they are lathed surfaces of revolution with no UV
            # map, so a flat projection has nothing to sample against. The blend
            # softens the seams where two projections meet.
            node.projection_blend = 0.3
        tree.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    diffuse = image_node(maps["Diffuse"], non_color=False)
    colour = diffuse.outputs["Color"]

    if hue_shift or saturation != 1.0 or brightness != 1.0:
        # Only two veneers exist that are usable on a piece, and a handful of table
        # textures, so without this every wooden set in the dataset is the same
        # rosewood against the same boxwood. Shifting hue and brightness per scene
        # turns two timbers into a range of them -- walnut, cherry, mahogany -- from
        # the same maps, which is the cheapest diversity available here.
        #
        # Blender's Hue input is centred on 0.5, so a shift is applied as an offset
        # from there rather than as an absolute value; 0.0 would rotate the hue a
        # full half-turn and make oak blue.
        adjust = tree.nodes.new("ShaderNodeHueSaturation")
        adjust.inputs["Hue"].default_value = 0.5 + hue_shift
        adjust.inputs["Saturation"].default_value = saturation
        adjust.inputs["Value"].default_value = brightness
        tree.links.new(colour, adjust.inputs["Color"])
        colour = adjust.outputs["Color"]

    if tint is not None:
        mix = tree.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        tree.links.new(colour, mix.inputs[6])
        mix.inputs[7].default_value = (*tint, 1.0)
        colour = mix.outputs[2]

    tree.links.new(colour, bsdf.inputs["Base Color"])

    rough = image_node(maps["Rough"], non_color=True)
    if roughness_shift:
        adjust = tree.nodes.new("ShaderNodeMath")
        adjust.operation = "ADD"
        adjust.use_clamp = True
        adjust.inputs[1].default_value = roughness_shift
        tree.links.new(rough.outputs["Color"], adjust.inputs[0])
        tree.links.new(adjust.outputs["Value"], bsdf.inputs["Roughness"])
    else:
        tree.links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    normal = image_node(maps["nor_gl"], non_color=True)
    tree.links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
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


def accent_for(color: RGB, contrast: float, *, warm: bool = True) -> list[float]:
    """The figure colour for a body colour: darker on light stock, lighter on dark.

    Resolved against the colour actually being shaded rather than fixed per scene.
    A white set and a black set share one style, and an absolute accent that reads
    as subtle grain on near-white boxwood is many times lighter than near-black
    ebony -- which produced pieces striped like a zebra while the white ones looked
    correct.

    Dark stock lightens instead of darkening, because there is no room below 0.07 to
    darken into; and the shift is warmer than neutral for wood, since latewood loses
    far more blue than red.
    """
    luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    bias = (1.0, 0.93, 0.82) if warm else (1.0, 0.98, 0.96)
    if luminance > 0.25:
        factors = [(1.0 - contrast) * channel for channel in bias]
    else:
        # Dark stock lightens by a small additive step rather than a ratio. The step
        # has to be *small*: ebony sits near 0.07, so even +0.15 is a threefold
        # ratio, and dark values are perceptually stretched enough that such a jump
        # reads as banding. +0.04 is a visible figure and no more.
        return [
            min(1.0, max(0.0, channel + contrast * 0.15 * tint))
            for channel, tint in zip(color, bias, strict=True)
        ]
    return [
        min(1.0, max(0.0, channel * factor))
        for channel, factor in zip(color, factors, strict=True)
    ]


def styled(
    name: str,
    color: RGB,
    style: dict | None,
    *,
    roughness: float = 0.35,
    coat: float = 0.25,
    flat: bool = False,
    maps: dict | None = None,
) -> bpy.types.Material:
    """Build whichever material a resolved :class:`MaterialStyle` asks for.

    The single place the ``kind`` string turns into a node graph, so the renderer
    never branches on it and a new material means one more arm here. ``None`` and
    unknown kinds fall through to the plain solid: an unrecognised style should
    render a valid board, not raise mid-batch.

    ``flat`` distinguishes a sawn surface from a turned one. It is the caller that
    knows which it has -- squares and the frame are flat, pieces are turned -- and
    getting it wrong puts concentric bullseyes on the board.
    """
    kind = (style or {}).get("kind", "plain")
    scale = (style or {}).get("scale", 3.0)
    distortion = (style or {}).get("distortion", 4.0)
    contrast = (style or {}).get("contrast", 0.2)
    vein_width = (style or {}).get("vein_width", 0.14)
    rotation = tuple(
        math.radians(angle) for angle in (style or {}).get("rotation_deg", (0, 0, 0))
    )

    if kind == "textured" and maps:
        # Flat for a board, triplanar for a piece. The board is a quad and projects
        # cleanly; a lathed piece has no UV map at all, so box projection is the
        # only way a photograph reaches it.
        #
        # The surface colour is multiplied *into* the photograph rather than
        # replaced by it. Light and dark squares share one texture, so without this
        # they come out identical and the board loses its chequer entirely -- an
        # image with no visible grid, still carrying labels that say where every
        # square is. Tinting keeps the photographed grain while restoring the
        # contrast, which is also how a real board is made: one timber, two stains.
        return textured(
            name,
            maps,
            scale=scale,
            projection="FLAT" if flat else "BOX",
            tint=color,
            hue_shift=(style or {}).get("hue_shift", 0.0),
            saturation=(style or {}).get("saturation", 1.0),
            brightness=(style or {}).get("brightness", 1.0),
        )
    if kind == "marble":
        return marble(
            name,
            color,
            accent_for(color, contrast, warm=False),
            scale=scale,
            distortion=distortion,
            vein_width=vein_width,
            rotation=rotation,
            roughness=roughness,
            coat=max(coat, 0.3),
        )
    if kind == "wood" and maps:
        # A photographed veneer beats the procedural on a piece, and box projection
        # is what makes it reachable without a UV map. Only veneers are used here:
        # a floor texture's plank joins wrap the piece as hoops, which is why the
        # texture set for pieces is curated separately from the table's.
        return textured(
            name,
            maps,
            scale=2.5,
            projection="BOX",
            hue_shift=(style or {}).get("hue_shift", 0.0),
            saturation=(style or {}).get("saturation", 1.0),
            brightness=(style or {}).get("brightness", 1.0),
        )
    if kind == "wood":
        return wood_grain(
            name,
            color,
            dark_color=accent_for(color, contrast, warm=True),
            rings=not flat,
            scale=scale,
            distortion=distortion,
            rotation=rotation,
            roughness=roughness,
            coat=coat,
        )
    if kind == "plastic":
        return piece_material(name, color, roughness=roughness)
    return solid(name, color, roughness=roughness, coat=coat)


def organic(
    material: bpy.types.Material,
    *,
    bevel_radius: float = 0.004,
    instance_value: float = 0.10,
    instance_saturation: float = 0.08,
) -> bpy.types.Material:
    """The two finishing touches every surface in the scene gets.

    Applied last, after the material's own node graph exists, because both hook
    into sockets the graph has already wired: the bevel chains onto the existing
    normal chain, and the per-instance tint splices in front of the base colour.
    """
    add_bevel(material, radius=bevel_radius)
    vary_per_instance(material, value=instance_value, saturation=instance_saturation)
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
