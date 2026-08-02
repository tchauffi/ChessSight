"""Render settings and the two render passes.

Beauty pass
-----------
EEVEE or Cycles, per the job spec. Cycles is pointed at OPTIX when available; note
that setting ``compute_device_type`` alone is not enough -- the individual devices
must also be enabled, and Blender keeps listing the CPU and CUDA entries afterwards.

Id pass
-------
A Workbench render in flat object-colour mode, where ``object.color`` *is* the output
pixel. Every one of the settings in :func:`configure_id_pass` is load-bearing:
anti-aliasing and dithering both blend neighbouring ids into values that decode to
pieces that do not exist, and the default ``AgX`` view transform re-encodes the
values entirely. Verified on this Blender: ids 1, 7, 33, 128, 200 and 255 round-trip
to exactly those bytes.
"""

from __future__ import annotations

import bpy

#: Settings the id pass overrides and the beauty pass needs back.
_SAVED_KEYS = (
    "engine",
    "dither_intensity",
    "film_transparent",
    "file_format",
    "color_mode",
    "quality",
    "view_transform",
    "look",
    "exposure",
    "gamma",
    "use_dof",
    "use_motion_blur",
)


def configure_devices(spec: dict) -> str:
    """Point Cycles at the GPU. Returns a short description of what was selected."""
    if spec["engine"] != "CYCLES" or not spec["use_gpu"]:
        return "cpu"

    preferences = bpy.context.preferences.addons["cycles"].preferences
    requested = spec["compute_device_type"]
    if requested == "NONE":
        return "cpu"

    try:
        preferences.compute_device_type = requested
    except TypeError:
        return "cpu (requested device type unavailable)"

    preferences.refresh_devices()
    # Blender still lists CPU and CUDA entries after selecting OPTIX, and any of
    # them left enabled would be used, so set `use` explicitly on every device.
    enabled = []
    for device in preferences.devices:
        device.use = device.type == requested
        if device.use:
            enabled.append(device.name)

    if not enabled:
        return "cpu (no matching device)"
    return f"{requested}: {', '.join(enabled)}"


def configure_render(spec: dict) -> None:
    """Apply the beauty-pass render settings from a job spec."""
    scene = bpy.context.scene
    render = scene.render

    render.engine = spec["engine"]
    render.resolution_x, render.resolution_y = spec["resolution"]
    render.resolution_percentage = 100
    render.film_transparent = False

    if spec["engine"] == "CYCLES":
        scene.cycles.samples = spec["samples"]
        scene.cycles.use_denoising = spec["denoise"]
        scene.cycles.device = "GPU" if spec["use_gpu"] else "CPU"
        if spec["denoise"]:
            try:
                # The denoiser enum is populated dynamically, so an OPTIX-only build
                # detail can make this raise rather than fall back.
                scene.cycles.denoiser = "OPTIX"
            except TypeError:
                pass
    else:
        scene.eevee.taa_render_samples = spec["samples"]

    image = render.image_settings
    image.file_format = spec["image_format"]
    if spec["image_format"] == "JPEG":
        image.quality = spec["jpeg_quality"]
        image.color_mode = "RGB"
    else:
        image.color_mode = "RGBA" if render.film_transparent else "RGB"


def render_to(path: str) -> None:
    """Render the current scene to ``path``."""
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def _snapshot() -> dict:
    scene = bpy.context.scene
    camera = scene.camera.data if scene.camera else None
    return {
        "engine": scene.render.engine,
        "dither_intensity": scene.render.dither_intensity,
        "film_transparent": scene.render.film_transparent,
        "file_format": scene.render.image_settings.file_format,
        "color_mode": scene.render.image_settings.color_mode,
        "quality": scene.render.image_settings.quality,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
        "use_dof": camera.dof.use_dof if camera else False,
        "use_motion_blur": scene.render.use_motion_blur,
    }


def _restore(saved: dict) -> None:
    scene = bpy.context.scene
    scene.render.engine = saved["engine"]
    scene.render.dither_intensity = saved["dither_intensity"]
    scene.render.film_transparent = saved["film_transparent"]
    scene.render.image_settings.file_format = saved["file_format"]
    scene.render.image_settings.color_mode = saved["color_mode"]
    scene.render.image_settings.quality = saved["quality"]
    scene.view_settings.view_transform = saved["view_transform"]
    scene.view_settings.look = saved["look"]
    scene.view_settings.exposure = saved["exposure"]
    scene.view_settings.gamma = saved["gamma"]
    scene.render.use_motion_blur = saved["use_motion_blur"]
    if scene.camera:
        scene.camera.data.dof.use_dof = saved["use_dof"]


def render_id_pass(path: str) -> None:
    """Render the instance-id pass, then restore every setting it changed."""
    scene = bpy.context.scene
    saved = _snapshot()
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "OBJECT"
        scene.display.render_aa = "OFF"

        # Anything that blends neighbouring pixels would invent ids that are not
        # real objects, so both are forced off regardless of the beauty settings.
        scene.render.dither_intensity = 0.0
        scene.render.use_motion_blur = False
        if scene.camera:
            scene.camera.data.dof.use_dof = False

        scene.render.film_transparent = False
        scene.view_settings.view_transform = "Raw"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0

        image = scene.render.image_settings
        image.file_format = "PNG"
        image.color_mode = "RGB"
        image.color_depth = "8"

        render_to(path)
    finally:
        _restore(saved)


def blender_version() -> str:
    return ".".join(str(part) for part in bpy.app.version)
