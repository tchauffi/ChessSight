"""Read all 64 squares at once from a rectified board.

The detector answers "what pieces are in this picture and where", and the answer
is then projected onto squares. That is more than the question needs. Once the
board is rectified every square is at a known, fixed place, so the model can be
asked the question actually wanted -- what is on each square -- and answer it in
one forward pass with no boxes, no NMS, no foot-point heuristic.

Why this is worth trying, in numbers from this project
-----------------------------------------------------
Geometry is already solved: swapping annotated corners for the model's own costs
1.07 points of per-square accuracy. The remaining 2.55 points, and the gap
between 31% board-exact and 100%, is piece identification. That is what this
model attacks, with two advantages the detector does not have -- every square
arrives at the same scale and orientation, and the model sees a square's
*neighbours*, so it can use the fact that boards are not random.

The head is aligned, not pooled
-------------------------------
Rectification makes square (rank, file) land in a fixed sub-rectangle of the
image, so the output grid can be read straight off the feature map with
``roi_align`` on the playing area. Pooling the backbone to one vector and
regressing 64x13 numbers would work too, and would make the model relearn a
correspondence that is already exact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from chesssight.data.fen import BOARD_SIZE, NUM_CLASSES
from chesssight.train.rectify import FAR_MARGIN, SIDE_MARGIN, playing_area_px


@dataclass
class GridConfig:
    """Everything needed to rebuild the model and reproduce the run."""

    data_roots: list[Path] = field(default_factory=list)
    output_dir: Path = Path("runs/grid")
    backbone: str = "resnet34"
    pretrained: bool = True
    image_size: int = 448
    side_margin: float = SIDE_MARGIN
    far_margin: float = FAR_MARGIN
    #: Corners are perturbed by this fraction of a square during training. The
    #: model is trained on annotated corners and deployed on predicted ones,
    #: which land about 0.2 squares out; without jitter it learns to trust an
    #: alignment it will not get.
    corner_jitter: float = 0.25
    channels: int = 128
    #: Feature cells sampled per square before the context convolutions reduce
    #: them to one prediction each. One cell per square was the first version's
    #: other mistake: it threw away within-square detail before anything could
    #: use it.
    cells_per_square: int = 4
    #: Backbone stride the board is read off. Stride 8 gave five feature
    #: positions per square and produced 12.65% right-colour-wrong-piece errors;
    #: stride 4 doubles that on each axis.
    feature_stride: int = 4
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    #: Higher than the 0.1 used elsewhere. A rectified board looks nothing like
    #: an ImageNet photograph -- flat, top-down, repeating texture -- so the
    #: backbone has a long way to travel, and at a tenth of the head's rate it
    #: barely moved: training loss was still falling at the end of a 25-epoch run
    #: that had plateaued at 90% per-square.
    backbone_lr_scale: float = 0.3
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


class GridNet(nn.Module):
    """Rectified board in, ``13 x 8 x 8`` logits out."""

    def __init__(
        self,
        backbone: str = "resnet34",
        *,
        pretrained: bool = True,
        image_size: int = 448,
        channels: int = 128,
        cells_per_square: int = 4,
        feature_stride: int = 4,
        side_margin: float = SIDE_MARGIN,
        far_margin: float = FAR_MARGIN,
    ) -> None:
        super().__init__()
        import timm

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
        widths = list(net.feature_info.channels())
        # Which feature map to read the board off. This is the resolution knob
        # that matters, and stride 8 was too coarse: at 448 input a square spans
        # about 40 pixels, so stride 8 leaves five feature positions per square
        # to describe a piece's whole silhouette. The measured consequence was
        # that 12.65% of occupied squares got the right colour and the wrong
        # piece -- n->b, p->b, r->q -- pieces of similar height confused with
        # each other, which is what reading a streak's length rather than its
        # shape looks like. Stride 4 doubles the detail on both axes.
        self.tap = min(
            range(len(reductions)),
            key=lambda index: abs(reductions[index] - feature_stride),
        )
        self.stride = reductions[self.tap]
        self.width = widths[self.tap]
        self.backbone = net
        self.image_size = image_size
        self.side_margin = side_margin
        self.far_margin = far_margin

        self.cells_per_square = cells_per_square
        self.neck = nn.Sequential(
            nn.Conv2d(self.width, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Two things the first version got wrong, both about the smear.
        #
        # It pooled each square to a single cell and classified it with a 1x1
        # convolution, so a square was identified from its own cell alone. But a
        # piece does not sit over its own square in a rectified image -- it
        # streaks away from the camera, and its identifying detail (the head, the
        # silhouette) lands on *neighbouring* cells while its own cell holds
        # mostly board surface. The head was denied exactly the evidence the
        # rectification was supposed to hand it.
        #
        # So: sample finer than one cell per square, mix across neighbours with
        # 3x3 convolutions, and only then reduce to one prediction per square.
        # The result sees a 5x5 square neighbourhood, which covers a piece
        # leaning in from two squares away.
        def block(kernel: int, stride: int = 1, pad: int = 0) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    channels, channels, kernel_size=kernel, stride=stride, padding=pad
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )

        # Two convolutions at *square* resolution rather than one, so a square's
        # prediction can draw on a 2-square neighbourhood. The smear was measured
        # at 1.28 squares on average and 2.50 at p90, so one square of context
        # does not reach the far end of a tall piece's streak.
        self.context = nn.Sequential(
            block(3, pad=1),
            block(cells_per_square, stride=cells_per_square),
            block(3, pad=1),
            block(3, pad=1),
        )
        self.classifier = nn.Conv2d(channels, NUM_CLASSES, kernel_size=1)

    def _as_nchw(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.shape[1] == self.width:
            return feature
        if feature.shape[-1] == self.width:
            return feature.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"feature map {tuple(feature.shape)} has {self.width} channels nowhere"
        )

    def pool_cells(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone features resampled onto the board's own grid.

        Exposed so the alignment tests exercise the model's real path. When the
        test built its own ``roi_align`` call the two could drift, and a readout
        that no longer matched the board is the one failure this whole design
        has to rule out.
        """
        from torchvision.ops import roi_align

        features = self.neck(self._as_nchw(self.backbone(images)[self.tap]))

        # The playing area is a known rectangle of the *input*; roi_align maps it
        # onto the feature map, so output cell (rank, file) corresponds to that
        # square by construction rather than by the model having learned where
        # the board is.
        x0, y0, x1, y1 = playing_area_px(
            self.image_size, side=self.side_margin, far=self.far_margin
        )
        boxes = torch.tensor(
            [[x0, y0, x1, y1]], dtype=features.dtype, device=features.device
        ).repeat(features.shape[0], 1)
        index = torch.arange(
            features.shape[0], dtype=features.dtype, device=features.device
        ).unsqueeze(1)
        size = BOARD_SIZE * self.cells_per_square
        return roi_align(
            features,
            torch.cat([index, boxes], dim=1),
            output_size=(size, size),
            spatial_scale=1.0 / self.stride,
            sampling_ratio=2,
            aligned=True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.context(self.pool_cells(images)))


def save(model: GridNet, config: GridConfig, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(config.to_json(), encoding="utf-8")


def load(path: Path, device: torch.device | str = "cpu"):
    path = Path(path)
    payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    val_root = payload.get("val_root")
    config = GridConfig(
        **{
            key: value
            for key, value in payload.items()
            if key not in ("data_roots", "output_dir", "val_root")
        },
        data_roots=[Path(root) for root in payload.get("data_roots", [])],
        output_dir=Path(payload.get("output_dir", ".")),
        val_root=Path(val_root) if val_root else None,
    )
    model = GridNet(
        config.backbone,
        pretrained=False,
        image_size=config.image_size,
        channels=config.channels,
        cells_per_square=config.cells_per_square,
        feature_stride=config.feature_stride,
        side_margin=config.side_margin,
        far_margin=config.far_margin,
    )
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    return model.to(device).eval(), config


def to_tensor(image) -> torch.Tensor:
    """A rectified PIL board to a uint8 ``3xHxW`` tensor.

    Separate from :func:`prepare` because augmentation belongs between the two:
    the sensor-realism transforms are tensor-only -- ``GaussianNoise`` raises on
    a PIL image -- so anything applying them has to be handed this, not a PIL.
    """
    from torchvision.transforms import v2

    return v2.functional.pil_to_tensor(image.convert("RGB"))


def prepare(image, size: int) -> torch.Tensor:
    """A rectified board, PIL or uint8 tensor, to a normalised batch of one."""
    from torchvision.transforms import v2

    tensor = image if isinstance(image, torch.Tensor) else to_tensor(image)
    if tensor.shape[-1] != size or tensor.shape[-2] != size:
        tensor = v2.functional.resize(tensor, [size, size], antialias=True)
    tensor = v2.functional.to_dtype(tensor, torch.float32, scale=True)
    return v2.functional.normalize(
        tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ).unsqueeze(0)


@torch.no_grad()
def read_grid(
    model: GridNet, image, corners: list[list[float]], device: torch.device
) -> list[list[int]]:
    """Corners plus a photograph to an 8x8 grid of class ids."""
    from chesssight.train.rectify import rectify

    warped = rectify(
        image,
        corners,
        size=model.image_size,
        side=model.side_margin,
        far=model.far_margin,
    )
    logits = model(prepare(warped, model.image_size).to(device))
    return logits[0].argmax(dim=0).cpu().tolist()
