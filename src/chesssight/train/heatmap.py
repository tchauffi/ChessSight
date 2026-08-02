"""A corner detector that predicts a heatmap instead of boxes.

Why not keep using the detector
-------------------------------
The RT-DETR checkpoint already carries corners, as a 14th class of one-square
boxes, and on ChessReD photographs that works: four corners on 306 of 306 test
frames, 0.23 squares of mean error. But a corner is a *point*, and asking a box
detector for one costs three things that matter:

* **Resolution.** A box is decoded from a query embedding, so localisation is
  limited by what the transformer chose to encode. A heatmap is a spatial
  argmax over a stride-4 grid, refined by a local soft-argmax, so precision is
  bounded by pixels rather than by query capacity.
* **Duplicates.** Nothing stops the detector putting two boxes on one physical
  corner, and four points containing a duplicate give a *degenerate* homography
  rather than an obviously wrong one -- :func:`chesssight.train.corners.select_quad`
  exists solely to police that. Peak suppression on a heatmap makes it structural.
* **Honest confidence.** A per-corner peak value says how sure the model is about
  *that corner*, which is what a downstream consumer needs in order to fall back.

One channel, not four
---------------------
Four separate channels would name the corners -- a8, h8, h1, a1 -- and that
naming is not learnable from a board's appearance: rotate the photograph and the
image is equally consistent with any assignment. So the model predicts a single
"corner-ness" map with four peaks, and ordering is left to the geometry in
:mod:`chesssight.train.corners`, which is where it already lives.

Out-of-frame corners are not predicted. A cropped board really is missing that
information, and a model that guesses at it would be inventing geometry; the
training targets simply omit those points and the evaluation reports how often
it happens.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#: Output stride. 4 keeps the map cheap while putting a cell well inside a
#: square: at 448 input on a board filling the frame, one cell is about an eighth
#: of a square, and the soft-argmax refinement below resolves inside a cell.
DEFAULT_STRIDE = 4

#: Gaussian spread of a target peak, in heatmap cells. Wide enough that the loss
#: has a gradient for a nearly-right prediction, narrow enough that two corners
#: of a small board do not merge into one blob.
DEFAULT_SIGMA = 2.0

#: The corner count is fixed by what a chessboard is, not by a tuning decision.
CORNERS = 4


@dataclass
class HeatmapConfig:
    """Everything needed to rebuild the model and reproduce the run."""

    data_roots: list[Path] = field(default_factory=list)
    output_dir: Path = Path("runs/corners")
    backbone: str = "resnet18"
    pretrained: bool = True
    image_size: int = 448
    stride: int = DEFAULT_STRIDE
    sigma: float = DEFAULT_SIGMA
    channels: int = 64
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    num_workers: int = 8
    seed: int = 0
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    augment: bool = True
    #: Crop area range, tighter than the detector's (0.45, 0.72). A detector
    #: benefits from partly-cropped boards -- it still finds the pieces that
    #: remain -- but a corner model needs all four points to yield a homography,
    #: and at the detector's setting only 48% of augmented boards keep all four
    #: against 68% here. The upper bound stays at 0.78 because above it the crop
    #: starts including the fill wedges that rotation leaves in the image
    #: corners, which is precisely a false corner cue.
    crop_scale: tuple[float, float] = (0.62, 0.78)
    limit: int | None = None
    eval_dataset: int = 0
    eval_split: str = "val"
    #: A dataset to validate on that is *not* trained on. Without this the
    #: checkpoint is selected on renders, and synthetic corner error plateaus
    #: around 4px while real-photo error is still moving -- so selection past
    #: that point is driven by noise in the wrong domain. Point this at the real
    #: set: its val split selects, its test split stays untouched.
    val_root: Path | None = None

    def to_json(self) -> str:
        payload = asdict(self)
        payload["data_roots"] = [str(root) for root in self.data_roots]
        payload["output_dir"] = str(self.output_dir)
        payload["val_root"] = str(self.val_root) if self.val_root else None
        return json.dumps(payload, indent=2)


def gaussian_peak(
    heatmap: torch.Tensor, x: float, y: float, sigma: float = DEFAULT_SIGMA
) -> None:
    """Splat one Gaussian into ``heatmap`` in place, combining by maximum.

    Maximum rather than sum: two corners close together in a steeply-angled view
    must not produce a single brighter blob between them, which is what addition
    gives and what a peak decoder would then read as one corner in the wrong
    place.
    """
    height, width = heatmap.shape
    radius = int(math.ceil(3 * sigma))
    left, right = max(0, int(x) - radius), min(width, int(x) + radius + 1)
    top, bottom = max(0, int(y) - radius), min(height, int(y) + radius + 1)
    if left >= right or top >= bottom:
        return

    xs = torch.arange(left, right, dtype=torch.float32)
    ys = torch.arange(top, bottom, dtype=torch.float32)
    grid = torch.exp(
        -(((xs[None, :] - x) ** 2) + ((ys[:, None] - y) ** 2)) / (2 * sigma * sigma)
    )
    patch = heatmap[top:bottom, left:right]
    torch.maximum(patch, grid, out=patch)

    # The nearest cell is set to exactly 1. The focal loss below counts positives
    # by `target == 1`, so a peak that merely approaches 1 contributes no positive
    # term at all and the model learns only to predict background.
    cx, cy = int(round(x)), int(round(y))
    if 0 <= cx < width and 0 <= cy < height:
        heatmap[cy, cx] = 1.0


def render_target(
    points: list[list[float]],
    size: int,
    *,
    stride: int = DEFAULT_STRIDE,
    sigma: float = DEFAULT_SIGMA,
) -> torch.Tensor:
    """The ``1 x size/stride x size/stride`` target for one image's corners.

    Points are in input-image pixels. Those outside the image are dropped rather
    than clamped: see the module docstring.
    """
    cells = size // stride
    heatmap = torch.zeros((cells, cells), dtype=torch.float32)
    for x, y in points:
        if not (0 <= x < size and 0 <= y < size):
            continue
        gaussian_peak(heatmap, x / stride, y / stride, sigma)
    return heatmap.unsqueeze(0)


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, *, alpha: float = 2.0, beta: float = 4.0
) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss.

    Plain BCE fails here for a structural reason: a stride-4 map has ~12500 cells
    and four of them are positive, so predicting zero everywhere is already
    99.97% correct. The ``(1 - target) ** beta`` factor discounts negatives near a
    peak -- being one cell off is nearly right and should not be punished like
    firing on a player's sleeve -- and normalising by the positive count keeps the
    scale independent of how many corners happen to be in frame.
    """
    probability = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    positive = target.eq(1.0).float()
    negative = 1.0 - positive

    positive_loss = -((1 - probability) ** alpha) * torch.log(probability) * positive
    negative_loss = (
        -((1 - target) ** beta)
        * (probability**alpha)
        * torch.log(1 - probability)
        * negative
    )
    count = positive.sum()
    if count == 0:
        # An image with no corner in frame is all negatives. Dividing by zero
        # positives would return NaN and poison the epoch, so it contributes its
        # background term alone.
        return negative_loss.sum() / max(1, logits.shape[0])
    return (positive_loss.sum() + negative_loss.sum()) / count


#: Feature strides the top-down path consumes, finest first.
PYRAMID = (4, 8, 16, 32)


class CornerHeatmapNet(nn.Module):
    """A backbone pyramid with a top-down path, ending in one heatmap channel.

    The backbone is any ``timm`` model that exposes a stride-4/8/16/32 pyramid --
    ResNet, ConvNeXt, Swin. Plain ViT is deliberately *not* usable here and that
    is a property of the task, not an oversight: a patch-16 ViT's finest feature
    map is stride 16, and upsampling it 4x to reach the heatmap grid throws away
    exactly the localisation this design exists to provide. Swin's patch-4 stem
    gives stride 4 natively, which is why it is the transformer that fits.

    The head stays small either way. Finding four high-contrast corners is a
    low-level geometric task, and the capacity question is whether the backbone
    transfers across domains, not whether the head can express the answer.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        *,
        pretrained: bool = True,
        channels: int = 64,
        stride: int = DEFAULT_STRIDE,
        image_size: int = 448,
    ) -> None:
        super().__init__()
        if stride != DEFAULT_STRIDE:
            raise ValueError(
                f"stride must be {DEFAULT_STRIDE}: the top-down path is built "
                f"from the backbone's own stages, whose finest is stride 4"
            )
        import timm

        # Windowed-attention backbones bake the input size into their relative
        # position tables, so they need it at construction; convolutional ones
        # reject the argument. Asking forgiveness keeps one code path.
        # Typed loosely: timm's factory returns a different class per backbone,
        # and the pyramid metadata below lives on the feature wrapper rather than
        # on nn.Module.
        net: Any
        try:
            net = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                img_size=image_size,
            )
        except TypeError:
            net = timm.create_model(backbone, pretrained=pretrained, features_only=True)

        reductions = list(net.feature_info.reduction())
        try:
            self.taps = [reductions.index(step) for step in PYRAMID]
        except ValueError as error:
            raise ValueError(
                f"backbone {backbone!r} exposes strides {reductions}, which do "
                f"not include the {list(PYRAMID)} pyramid this head needs. A "
                f"patch-16 ViT fails here for that reason -- use a hierarchical "
                f"model such as swin_tiny_patch4_window7_224."
            ) from error

        # Read the widths before assigning the module: afterwards `self.backbone`
        # is typed as Module and `feature_info` is invisible to the checker.
        channel_counts = list(net.feature_info.channels())
        self.backbone = net
        self.stride = stride
        widths = [channel_counts[index] for index in self.taps]
        self.widths = widths
        self.lateral = nn.ModuleList(
            nn.Conv2d(width, channels, kernel_size=1) for width in widths
        )
        self.smooth = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1) for _ in range(3)
        )
        predictor = nn.Conv2d(channels, 1, kernel_size=1)
        # A prior of ~0.01 on the final bias. Without it the first steps push every
        # cell down hard from 0.5, and the focal loss spends its early budget
        # rediscovering that corners are rare.
        assert predictor.bias is not None  # Conv2d carries one unless bias=False
        nn.init.constant_(predictor.bias, -4.6)
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            predictor,
        )

    @staticmethod
    def _as_nchw(feature: torch.Tensor, width: int) -> torch.Tensor:
        """Put a feature map in NCHW, whichever layout the backbone produced.

        Swin returns NHWC from ``features_only`` and ignores ``output_fmt``, so
        the layout has to be established from the data rather than assumed. The
        channel count is the discriminator: it is known from ``feature_info``,
        and on a square feature map the spatial axes cannot be told apart by
        shape alone.
        """
        if feature.shape[1] == width:
            return feature
        if feature.shape[-1] == width:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"feature map {tuple(feature.shape)} has {width} channels on no axis"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.backbone(images)
        features = [
            self._as_nchw(maps[tap], width)
            for tap, width in zip(self.taps, self.widths, strict=True)
        ]

        top = self.lateral[3](features[3])
        for index in range(3):
            lateral = self.lateral[2 - index](features[2 - index])
            top = F.interpolate(top, size=lateral.shape[-2:], mode="nearest") + lateral
            top = self.smooth[index](top)
        return self.head(top)


def decode(
    logits: torch.Tensor,
    *,
    count: int = CORNERS,
    stride: int = DEFAULT_STRIDE,
    radius: int = 2,
) -> list[tuple[float, float, float]]:
    """Peaks of one heatmap as ``(x, y, score)`` in input-image pixels.

    Two stages, and both are needed. A 3x3 maximum filter keeps only cells that
    dominate their neighbourhood, so one broad blob yields one point instead of
    nine adjacent ones -- this is what makes duplicate corners structurally
    impossible rather than something to filter afterwards. Then a local
    intensity-weighted mean over a small window puts the point *between* cells,
    which matters: at stride 4 a whole-cell answer is quantised to 4 pixels of
    the input, and a board photographed at 3072px scales that up by nearly seven.
    """
    if logits.dim() == 4:
        logits = logits[0]
    scores = torch.sigmoid(logits[0])
    pooled = F.max_pool2d(scores[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    peaks = scores * (scores >= pooled).float()

    height, width = peaks.shape
    flat = peaks.flatten()
    take = min(count, flat.numel())
    values, indices = torch.topk(flat, take)

    found: list[tuple[float, float, float]] = []
    for value, index in zip(values.tolist(), indices.tolist(), strict=True):
        cy, cx = divmod(int(index), width)
        left, right = max(0, cx - radius), min(width, cx + radius + 1)
        top, bottom = max(0, cy - radius), min(height, cy + radius + 1)
        window = scores[top:bottom, left:right]
        weight = window.sum()
        if weight > 0:
            xs = torch.arange(left, right, dtype=torch.float32, device=window.device)
            ys = torch.arange(top, bottom, dtype=torch.float32, device=window.device)
            x = float((window.sum(dim=0) * xs).sum() / weight)
            y = float((window.sum(dim=1) * ys).sum() / weight)
        else:
            x, y = float(cx), float(cy)
        found.append(((x + 0.5) * stride, (y + 0.5) * stride, float(value)))
    return found


def peaks_in_image(
    logits: torch.Tensor,
    *,
    size: tuple[int, int],
    input_size: int,
    stride: int = DEFAULT_STRIDE,
    count: int = CORNERS,
) -> list[tuple[float, float, float]]:
    """Decoded peaks rescaled to the original image's pixels, best first.

    ``size`` is the original ``(width, height)``; the model saw a square resize of
    it, so the two axes scale independently.
    """
    width, height = size
    return [
        (x * width / input_size, y * height / input_size, score)
        for x, y, score in decode(logits, stride=stride, count=count)
    ]


def quad_from_logits(
    logits: torch.Tensor,
    *,
    size: tuple[int, int],
    input_size: int,
    stride: int = DEFAULT_STRIDE,
    min_score: float = 0.0,
) -> list[list[float]] | None:
    """An ordered quad in the original image's pixels, or None.

    ``min_score`` is what makes this able to answer "no". Peak extraction always
    returns four points -- ``topk`` does not care whether they are corners -- so
    without a floor the model can never report that it did not find the board,
    and a "four corners found on 100%" figure would be an artefact of the decoder
    rather than a property of the model.
    """
    from chesssight.train.corners import order_clockwise

    found = [
        point
        for point in peaks_in_image(
            logits, size=size, input_size=input_size, stride=stride
        )
        if point[2] >= min_score
    ]
    if len(found) < CORNERS:
        return None
    return order_clockwise([(x, y) for x, y, _ in found[:CORNERS]])


def save(model: CornerHeatmapNet, config: HeatmapConfig, path: Path) -> None:
    """Write weights and the config needed to rebuild them side by side."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(config.to_json(), encoding="utf-8")


def load(path: Path, device: torch.device | str = "cpu"):
    """Reload a saved corner model and the config it was built with."""
    path = Path(path)
    payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    val_root = payload.get("val_root")
    config = HeatmapConfig(
        **{
            key: value
            for key, value in payload.items()
            if key not in ("data_roots", "output_dir", "val_root")
        },
        data_roots=[Path(root) for root in payload.get("data_roots", [])],
        output_dir=Path(payload.get("output_dir", ".")),
        val_root=Path(val_root) if val_root else None,
    )
    # Rebuilding with pretrained=True would download ImageNet weights only to
    # overwrite every one of them a line later.
    model = CornerHeatmapNet(
        config.backbone,
        pretrained=False,
        channels=config.channels,
        stride=config.stride,
        image_size=config.image_size,
    )
    state = torch.load(path / "model.pt", map_location="cpu")
    # Checkpoints written before the backbone was generalised to timm carry the
    # torchvision layout, `stem.*` and `layer1..4`, and cannot be mapped onto a
    # timm pyramid one-for-one. Say so, rather than emitting two hundred lines of
    # missing keys that bury the actual cause.
    if any(key.startswith(("stem.", "layer1.")) for key in state):
        raise RuntimeError(
            f"{path} was saved by the torchvision-backbone version of this model "
            f"and cannot be loaded now that backbones come from timm. Retrain it, "
            f"or check out the commit that wrote it."
        )
    model.load_state_dict(state)
    return model.to(device).eval(), config


def preprocess(image, size: int) -> torch.Tensor:
    """One PIL image to a normalised ``1x3xSxS`` tensor, ImageNet statistics."""
    from torchvision.transforms import v2

    tensor = v2.functional.pil_to_tensor(image.convert("RGB"))
    tensor = v2.functional.resize(tensor, [size, size], antialias=True)
    tensor = v2.functional.to_dtype(tensor, torch.float32, scale=True)
    return v2.functional.normalize(
        tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ).unsqueeze(0)


@torch.no_grad()
def predict_quad(
    model: CornerHeatmapNet,
    image,
    device: torch.device,
    *,
    input_size: int = 448,
    stride: int = DEFAULT_STRIDE,
    min_score: float = 0.0,
) -> list[list[float]] | None:
    """Board corners for one PIL image, in that image's own pixels."""
    logits = model(preprocess(image, input_size).to(device))
    return quad_from_logits(
        logits.float().cpu(),
        size=image.size,
        input_size=input_size,
        stride=stride,
        min_score=min_score,
    )


def square_size(corners: list[list[float]]) -> float:
    """Mean side of a square, in pixels, from the board's corners.

    Corner error means nothing in isolation -- 30 pixels is excellent on a
    close-up and hopeless on a distant board -- so every error this project
    reports is also divided by this.
    """
    points = np.asarray(corners, dtype=np.float64)
    sides = [
        float(np.linalg.norm(points[index] - points[(index + 1) % len(points)]))
        for index in range(len(points))
    ]
    return float(np.mean(sides)) / 8.0
