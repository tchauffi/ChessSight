"""Train-time augmentation.

Distinct from the domain randomisation the generator already does. That varies
camera, lighting, materials and placement *when an image is rendered*, so a given
sample looks the same every epoch. This varies each sample every time it is seen,
and -- more to the point -- it can simulate things Blender does not: sensor noise,
JPEG artefacts, white-balance drift, motion blur.

What is deliberately absent
---------------------------
**No horizontal or vertical flip, and no 90-degree rotation.** Every detection
recipe reaches for flips first, and all three are wrong here. Mirroring a board
puts the light square on the player's left, which never happens on a correctly set
up board -- it is precisely the defect :func:`chesssight.data.geometry.is_mirrored`
exists to reject, so augmenting with it would inject the thing the generator
guarantees against. Flips and quarter turns also remap squares, which breaks the
downstream board-reading task even where the detector would not care.

Rotation is kept small for the same reason: a few degrees is camera roll, which is
real. Ninety is a different board.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

#: Boxes smaller than this after a geometric transform are dropped rather than kept
#: as slivers, which the matcher would otherwise waste queries on.
MIN_BOX_SIZE_PX = 4


@dataclass(frozen=True)
class AugmentConfig:
    """Strength of each augmentation family.

    Set any probability to zero to switch that family off.
    """

    #: Brightness, contrast, saturation and hue. The highest-value family for
    #: sim-to-real: real photographs vary in exposure and white balance far more
    #: than a renderer with a fixed tone curve does.
    photometric_probability: float = 0.8
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.05

    #: Sensor realism. Renders are noise-free and unnaturally sharp; photographs
    #: are neither, and a detector trained only on clean images leans on detail
    #: that a phone camera does not deliver.
    noise_probability: float = 0.3
    noise_sigma: float = 0.02
    blur_probability: float = 0.2
    blur_kernel: int = 5
    jpeg_probability: float = 0.3
    jpeg_quality: tuple[int, int] = (40, 90)

    #: Geometry. The crop always runs -- it is what brings an image down to the
    #: working resolution, and its scale range is what makes it an augmentation
    #: rather than a plain resize. Small scales also produce partially cropped
    #: boards, which real photographs do constantly.
    #:
    #: The upper bound is capped below the inscribed square of a rotated image
    #: (~0.886 linear, ~0.78 area at 8 degrees) so the crop never includes the
    #: wedges of fill that rotation leaves in the corners.
    crop_scale: tuple[float, float] = (0.45, 0.72)
    crop_ratio: tuple[float, float] = (0.85, 1.18)
    #: Working canvas before the crop, as a multiple of ``image_size``. Everything
    #: after the initial resize runs at roughly the output resolution, which is the
    #: difference between 6 ms and 330 ms per image on a 3072px photograph.
    work_scale: float = 1.2
    #: Camera roll, in degrees. Small on purpose: see the module docstring.
    rotation_degrees: float = 8.0
    rotation_probability: float = 0.3
    #: Grey the rotation fills exposed corners with. Rotating without expanding
    #: leaves wedges of fill at the edges, and the default black reads as a hard
    #: shadow that no photograph contains -- the crop that follows removes most of
    #: it, and a mid grey is unobtrusive where any survives.
    rotation_fill: int = 114

    image_size: int = 512


def geometric_steps(config: AugmentConfig) -> list[v2.Transform]:
    """Resize, roll and crop. Shared by every task's pipeline.

    Kept separate from the photometric half so that a second target type -- board
    corners as keypoints rather than pieces as boxes -- reuses this geometry
    instead of restating it. The rules here (crop *after* rotation, work at
    roughly output resolution, no flips) are the ones that must not diverge
    between tasks: they are about what a chessboard photograph can look like, not
    about what the model predicts.
    """
    work = int(round(config.image_size * config.work_scale))

    # Downscale first. ChessReD photographs are 3072 square and everything here
    # is resolution-bound: blurring, JPEG round-tripping and noising at native
    # size costs 330 ms an image against 6 ms at working size, and every one of
    # those pixels is discarded by the resize a moment later anyway.
    steps: list[v2.Transform] = [v2.Resize(size=(work, work), antialias=True)]

    # Rotation comes *before* the crop, not after. Rotating without expanding
    # leaves wedges of fill in the corners; cropping afterwards removes them.
    # The other order bakes those wedges into the final image.
    if config.rotation_probability > 0 and config.rotation_degrees > 0:
        steps.append(
            v2.RandomApply(
                [
                    v2.RandomRotation(
                        degrees=config.rotation_degrees,
                        expand=False,
                        fill=config.rotation_fill,
                    )
                ],
                p=config.rotation_probability,
            )
        )
    steps.append(
        v2.RandomResizedCrop(
            size=(config.image_size, config.image_size),
            scale=config.crop_scale,
            ratio=config.crop_ratio,
            antialias=True,
        )
    )
    return steps


def photometric_steps(config: AugmentConfig) -> list[v2.Transform]:
    """Colour, blur, JPEG and sensor noise. Target-agnostic by construction."""
    steps: list[v2.Transform] = []
    if config.photometric_probability > 0:
        steps.append(
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=config.brightness,
                        contrast=config.contrast,
                        saturation=config.saturation,
                        hue=config.hue,
                    )
                ],
                p=config.photometric_probability,
            )
        )
    if config.blur_probability > 0:
        steps.append(
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=config.blur_kernel)],
                p=config.blur_probability,
            )
        )
    if config.jpeg_probability > 0:
        steps.append(
            v2.RandomApply(
                [v2.JPEG(quality=config.jpeg_quality)], p=config.jpeg_probability
            )
        )
    if config.noise_probability > 0:
        # GaussianNoise wants float input, so the conversion is bracketed here
        # rather than left to leak into the rest of the pipeline.
        steps.append(
            v2.RandomApply(
                [
                    v2.Compose(
                        [
                            v2.ToDtype(torch.float32, scale=True),
                            v2.GaussianNoise(sigma=config.noise_sigma),
                            v2.ToDtype(torch.uint8, scale=True),
                        ]
                    )
                ],
                p=config.noise_probability,
            )
        )
    return steps


def build_transform(config: AugmentConfig) -> v2.Transform:
    """Compose the detection pipeline: geometry, then photometry, then sanitising.

    Geometry runs first so that photometric and sensor effects apply to the final
    framing -- blurring before a crop would sharpen the result back up by
    resampling, which is not what a real camera does.
    """
    steps = geometric_steps(config) + photometric_steps(config)

    # Geometric transforms push boxes out of frame and shrink others to slivers.
    # Sanitising afterwards is what keeps the targets honest.
    steps.append(v2.ClampBoundingBoxes())
    # The labels getter is explicit rather than left to the default heuristic,
    # which searches the input for a key it recognises and raises on anything else.
    steps.append(
        v2.SanitizeBoundingBoxes(
            min_size=MIN_BOX_SIZE_PX,
            labels_getter=lambda inputs: inputs["labels"],
        )
    )
    return v2.Compose(steps)


def apply(
    transform: v2.Transform,
    image: Image.Image,
    boxes_xywh: list[list[float]],
    labels: list[int],
) -> tuple[Image.Image, list[list[float]], list[int]]:
    """Augment a PIL image and its COCO-format boxes together.

    Boxes go in and come out as ``xywh`` because that is what the detector's
    processor expects; XYXY is used internally because it is what torchvision's
    box transforms operate on.
    """
    tensor = tv_tensors.Image(v2.functional.pil_to_tensor(image))
    height, width = tensor.shape[-2:]

    xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes_xywh] or torch.zeros(
        (0, 4)
    ).tolist()
    boxes = tv_tensors.BoundingBoxes(
        torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=(height, width),
    )
    payload = {
        "image": tensor,
        "boxes": boxes,
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    out = transform(payload)

    result = [
        [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
        for x0, y0, x1, y1 in out["boxes"].tolist()
    ]
    return (
        v2.functional.to_pil_image(out["image"]),
        result,
        [int(value) for value in out["labels"].tolist()],
    )


def build_corner_transform(config: AugmentConfig) -> v2.Transform:
    """The same pipeline for keypoint targets.

    No clamping and no sanitising: a corner pushed out of frame by the crop is
    *information*, not a defective label. Clamping it to the border would teach
    the model that a corner sits wherever the image happens to end, which is the
    one thing a corner detector must never learn. The caller drops out-of-frame
    points instead, and knows it did.
    """
    return v2.Compose(geometric_steps(config) + photometric_steps(config))


def apply_corners(
    transform: v2.Transform, image: Image.Image, points: list[list[float]]
) -> tuple[Image.Image, list[list[float]]]:
    """Augment an image and its corner points together.

    Points ride through as ``tv_tensors.KeyPoints`` so torchvision applies the
    identical geometry to image and label -- the alternative, replaying the
    sampled crop and rotation by hand, is exactly the kind of second
    implementation that silently drifts from the first.
    """
    tensor = tv_tensors.Image(v2.functional.pil_to_tensor(image))
    height, width = tensor.shape[-2:]
    keypoints = tv_tensors.KeyPoints(
        torch.tensor(points, dtype=torch.float32).reshape(-1, 2),
        canvas_size=(height, width),
    )
    out = transform({"image": tensor, "keypoints": keypoints})
    return (
        v2.functional.to_pil_image(out["image"]),
        [[float(x), float(y)] for x, y in out["keypoints"].tolist()],
    )
