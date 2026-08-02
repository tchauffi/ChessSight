"""Corners regressed from the board's box, so they can lie outside the picture.

The heatmap model cannot place a corner that is not in the frame -- a peak has to
land on a pixel -- and that is not a shortcoming of the implementation but of the
representation. It matters because cropped boards are the dominant real failure:
measured over 48 frames of tournament video, the board *box* was found on 48 and
four corners on 2 with the detector's corner class, 8 with the Swin heatmap. The
signal that survives is the box.

So this model takes the box as given and predicts where the four corners sit
*relative to it*, in units of the box's own width and height. A corner one square
below the bottom of the frame is simply a target with ``y > 1``. Nothing about
that is special-cased; it falls out of choosing coordinates the answer can be
expressed in.

Why a plain ViT works here and not for the heatmap
--------------------------------------------------
The heatmap needs stride-4 spatial features, which a patch-16 ViT does not have.
This head pools the backbone to a single vector and regresses eight numbers, so
spatial resolution is not on the critical path and any classifier trunk will do --
ViT included. Same task, opposite architectural constraint, because the output
representation changed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from chesssight.train.heatmap import CORNERS

#: How far beyond the board box the crop reaches, as a fraction of the box. The
#: corners of a steeply-angled board sit outside its axis-aligned box, and a crop
#: that stops at the box would show the model none of the context it needs to
#: place them. Also absorbs the slop in a *detected* box at inference.
DEFAULT_MARGIN = 0.25


@dataclass
class BoxCornerConfig:
    """Everything needed to rebuild the model and reproduce the run."""

    data_roots: list[Path] = field(default_factory=list)
    output_dir: Path = Path("runs/boxcorners")
    backbone: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    margin: float = DEFAULT_MARGIN
    #: Random scale and shift applied to the box during training, as a fraction
    #: of its size. Not decoration: the model is trained on annotation-derived
    #: boxes and deployed on detected ones, and without jitter it learns to trust
    #: an exactness the detector does not provide.
    jitter_scale: float = 0.12
    jitter_shift: float = 0.08
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    num_workers: int = 8
    seed: int = 0
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    augment: bool = True
    limit: int | None = None
    eval_dataset: int = 0
    eval_split: str = "val"
    val_root: Path | None = None

    def to_json(self) -> str:
        payload = asdict(self)
        payload["data_roots"] = [str(root) for root in self.data_roots]
        payload["output_dir"] = str(self.output_dir)
        payload["val_root"] = str(self.val_root) if self.val_root else None
        return json.dumps(payload, indent=2)


def crop_box(
    box: tuple[float, float, float, float], margin: float = DEFAULT_MARGIN
) -> tuple[float, float, float, float]:
    """Grow a board box by ``margin`` on every side, as xyxy.

    Deliberately *not* clipped to the image. The crop is allowed to hang off the
    edge, and the padding that results is what tells the model the board runs out
    of frame there -- clipping would shift the crop back inside and silently move
    every target with it.
    """
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return x0 - dx, y0 - dy, x1 + dx, y1 + dy


def to_box_space(
    corners: list[list[float]], box: tuple[float, float, float, float]
) -> np.ndarray:
    """Corners in units of the crop box: (0,0) its top-left, (1,1) its bottom-right.

    Values outside [0, 1] are the point of the exercise, not an error to clamp.
    """
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate crop box {box}")
    points = np.asarray(corners, dtype=np.float64)
    return np.stack([(points[:, 0] - x0) / width, (points[:, 1] - y0) / height], axis=1)


def to_image_space(
    normalised: np.ndarray, box: tuple[float, float, float, float]
) -> list[list[float]]:
    """The inverse of :func:`to_box_space`."""
    x0, y0, x1, y1 = box
    points = np.asarray(normalised, dtype=np.float64).reshape(-1, 2)
    return [[float(x0 + u * (x1 - x0)), float(y0 + v * (y1 - y0))] for u, v in points]


def jitter(
    box: tuple[float, float, float, float],
    rng: np.random.Generator,
    *,
    scale: float,
    shift: float,
) -> tuple[float, float, float, float]:
    """Perturb a box the way a detector would get it slightly wrong."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    grow = 1.0 + rng.uniform(-scale, scale)
    cx = (x0 + x1) / 2 + rng.uniform(-shift, shift) * width
    cy = (y0 + y1) / 2 + rng.uniform(-shift, shift) * height
    half_w, half_h = width * grow / 2, height * grow / 2
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


class BoxCornerNet(nn.Module):
    """Pooled backbone features to eight numbers: four corners in box space."""

    def __init__(
        self,
        backbone: str = "resnet18",
        *,
        pretrained: bool = True,
        image_size: int = 224,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        import timm

        net: Any
        try:
            net = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0, img_size=image_size
            )
        except TypeError:
            net = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        features = int(net.num_features)
        self.backbone = net
        self.head = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, CORNERS * 2),
        )
        # Start by predicting the box's own corners. That is the answer for a
        # board seen square-on, so the model begins near a sensible geometry
        # rather than at the origin, which is a corner of the crop.
        predictor = self.head[-1]
        assert isinstance(predictor, nn.Linear)
        with torch.no_grad():
            predictor.bias.copy_(
                torch.tensor(
                    [
                        DEFAULT_MARGIN,
                        DEFAULT_MARGIN,
                        1 - DEFAULT_MARGIN,
                        DEFAULT_MARGIN,
                        1 - DEFAULT_MARGIN,
                        1 - DEFAULT_MARGIN,
                        DEFAULT_MARGIN,
                        1 - DEFAULT_MARGIN,
                    ],
                    dtype=torch.float32,
                )
            )
            predictor.weight.mul_(0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images)).view(-1, CORNERS, 2)


def save(model: BoxCornerNet, config: BoxCornerConfig, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(config.to_json(), encoding="utf-8")


def load(path: Path, device: torch.device | str = "cpu"):
    """Reload a saved box-corner model and its config."""
    path = Path(path)
    payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    val_root = payload.get("val_root")
    config = BoxCornerConfig(
        **{
            key: value
            for key, value in payload.items()
            if key not in ("data_roots", "output_dir", "val_root")
        },
        data_roots=[Path(root) for root in payload.get("data_roots", [])],
        output_dir=Path(payload.get("output_dir", ".")),
        val_root=Path(val_root) if val_root else None,
    )
    model = BoxCornerNet(
        config.backbone, pretrained=False, image_size=config.image_size
    )
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    return model.to(device).eval(), config


def prepare(image, box: tuple[float, float, float, float], size: int) -> torch.Tensor:
    """Crop, pad and normalise one image for the model.

    ``Image.crop`` with coordinates outside the image pads rather than clipping,
    which is exactly the behaviour wanted: a board running off the frame keeps
    its geometry and gains a blank margin.
    """
    from torchvision.transforms import v2

    crop = image.convert("RGB").crop(tuple(int(round(v)) for v in box))
    tensor = v2.functional.pil_to_tensor(crop)
    tensor = v2.functional.resize(tensor, [size, size], antialias=True)
    tensor = v2.functional.to_dtype(tensor, torch.float32, scale=True)
    return v2.functional.normalize(
        tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ).unsqueeze(0)


@torch.no_grad()
def predict_quad(
    model: BoxCornerNet,
    image,
    board_box: tuple[float, float, float, float],
    device: torch.device,
    *,
    image_size: int = 224,
    margin: float = DEFAULT_MARGIN,
) -> list[list[float]]:
    """Board corners in image pixels, given the detected board box.

    Always returns four points: unlike the heatmap there is no confidence to
    threshold on, because the model is asked where the corners are rather than
    whether they are visible. The caller's gate is the *box* detection.
    """
    box = crop_box(board_box, margin)
    predicted = model(prepare(image, box, image_size).to(device))
    return to_image_space(predicted[0].float().cpu().numpy(), box)
