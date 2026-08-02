"""Train and evaluate the corner heatmap model.

Kept apart from :mod:`chesssight.train.run` because almost nothing is shared: no
processor, no Hungarian matcher, no COCO metric, and a checkpoint that is a
state dict rather than a transformers model. What *is* shared -- the split rule
and the corner-error metric -- is imported rather than restated.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from chesssight.data.dataset import DatasetReader
from chesssight.train.corners import corner_error

if TYPE_CHECKING:
    from chesssight.train.gate import Gate
from chesssight.train.dataset import (
    CornerHeatmapDataset,
    SplitSpec,
    collate_corners,
    describe_split,
)
from chesssight.train.engine import cosine_schedule, resolve_device
from chesssight.train.heatmap import (
    CORNERS,
    CornerHeatmapNet,
    HeatmapConfig,
    decode,
    focal_loss,
    quad_from_logits,
    save,
    square_size,
)


def build_loaders(config: HeatmapConfig) -> tuple[DataLoader, DataLoader]:
    from chesssight.train.augment import AugmentConfig, build_corner_transform

    spec = SplitSpec(
        val_fraction=config.val_fraction, test_fraction=config.test_fraction
    )
    transform = (
        build_corner_transform(
            AugmentConfig(image_size=config.image_size, crop_scale=config.crop_scale)
        )
        if config.augment
        else None
    )

    def make(split: str, augmented: bool):
        parts = [
            CornerHeatmapDataset(
                root,
                image_size=config.image_size,
                stride=config.stride,
                sigma=config.sigma,
                split=split,
                split_spec=spec,
                limit=config.limit,
                transform=transform if augmented else None,
            )
            for root in config.data_roots
        ]
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    train_loader = DataLoader(
        make("train", config.augment),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_corners,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    # A held-out real set when one is given, otherwise a split of the training
    # data. The distinction matters more than it looks: selecting on renders
    # optimises for the domain that is not the target.
    validation = CornerHeatmapDataset(
        config.val_root or config.data_roots[config.eval_dataset],
        image_size=config.image_size,
        stride=config.stride,
        sigma=config.sigma,
        split=config.eval_split,
        split_spec=spec,
        limit=config.limit,
    )
    val_loader = DataLoader(
        validation,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_corners,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model, loader, optimizer, scheduler, device, *, epoch: int
) -> float:
    model.train()
    total, batches = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        images = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            loss = focal_loss(model(images).float(), target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        total += float(loss.detach())
        batches += 1
        if step % 100 == 0:
            print(
                f"[chesssight] epoch {epoch} step {step}/{len(loader)} "
                f"loss {total / batches:.4f}",
                flush=True,
            )
    return total / max(1, batches)


@torch.no_grad()
def validate(model, loader, device, *, stride: int) -> dict:
    """Loss plus what the loss cannot say: how far the peaks actually land.

    Measured in the model's own input pixels against the augmented points, so
    this number is comparable across epochs but not across image sizes; the
    real-photo evaluation below reports squares, which is comparable across
    everything.

    ``scorable_rate`` is a property of the *data*, not of the model: the decoder
    returns four peaks from any input, so this counts boards that had four
    corners in frame to compare against. On train5's validation split that is
    93.9%, and the metric sits there from the first epoch. It is here to explain
    which boards the error was averaged over, and reading it as accuracy would
    be reading the dataset's framing as the model's skill.
    """
    model.eval()
    losses: list[float] = []
    errors: list[float] = []
    complete = 0
    boards = 0

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        logits = model(images).float()
        losses.append(float(focal_loss(logits, target)))

        for index in range(images.shape[0]):
            visible = batch["visible"][index]
            truth = batch["points"][index][visible].tolist()
            boards += 1
            if len(truth) < CORNERS:
                continue  # a cropped board cannot be scored on all four
            found = decode(logits[index].cpu(), stride=stride)
            if len(found) < CORNERS:
                continue
            complete += 1
            error = corner_error([[x, y] for x, y, _ in found], truth)
            if error is not None:
                errors.append(error)

    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "corner_px": float(np.mean(errors)) if errors else float("nan"),
        "corner_px_median": float(np.median(errors)) if errors else float("nan"),
        "scorable_rate": complete / max(1, boards),
    }


def train(config: HeatmapConfig, device: str | None = None) -> dict:
    torch.manual_seed(config.seed)
    resolved = resolve_device(device)
    spec = SplitSpec(
        val_fraction=config.val_fraction, test_fraction=config.test_fraction
    )
    for root in config.data_roots:
        counts = describe_split(root, spec)
        print(
            f"[chesssight] {counts['total']} samples from {root} "
            f"(train {counts['train']}, val {counts['val']}, test {counts['test']})",
            flush=True,
        )
    validating_on = config.val_root or config.data_roots[config.eval_dataset]
    print(
        f"[chesssight] validating on {validating_on} split {config.eval_split!r}"
        f"{'' if config.val_root else ' (a split of the training data)'}",
        flush=True,
    )
    print(
        f"[chesssight] corner heatmap: {config.backbone} at {config.image_size}px, "
        f"stride {config.stride}, sigma {config.sigma} on {resolved}",
        flush=True,
    )

    model = CornerHeatmapNet(
        config.backbone,
        pretrained=config.pretrained,
        channels=config.channels,
        stride=config.stride,
        image_size=config.image_size,
    ).to(resolved)
    train_loader, val_loader = build_loaders(config)

    # The pretrained backbone starts far closer to useful than the randomly
    # initialised head, so it moves at a tenth of the rate; letting both run at
    # the head's rate destroys the ImageNet features in the first few hundred
    # steps.
    backbone_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        target = backbone_params if name.startswith("backbone.") else head_params
        target.append(parameter)
    if not backbone_params:
        raise RuntimeError(
            "no parameters matched the backbone prefix; the whole model would "
            "train at the head's rate and destroy the pretrained features"
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": config.learning_rate * config.backbone_lr_scale,
            },
            {"params": head_params, "lr": config.learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = cosine_schedule(
        optimizer, max(1, len(train_loader) * config.epochs), config.warmup_fraction
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(config.to_json(), encoding="utf-8")

    history: list[dict] = []
    best = float("inf")
    started = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, resolved, epoch=epoch
        )
        metrics = validate(model, val_loader, resolved, stride=config.stride)
        record = {"epoch": epoch, "train_loss": loss, **metrics}
        history.append(record)
        print(
            f"[chesssight] epoch {epoch}: train {loss:.4f} "
            f"val {metrics['val_loss']:.4f}  corner {metrics['corner_px']:.2f}px "
            f"(median {metrics['corner_px_median']:.2f})  "
            f"scorable on {metrics['scorable_rate']:.1%}",
            flush=True,
        )

        # Selected on pixel error, not on loss. The loss is a proxy for the shape
        # of a blob; the error is the thing the homography actually depends on.
        score = metrics["corner_px"]
        if not np.isnan(score) and score < best:
            best = score
            save(model, config, config.output_dir / "best")
            print(f"[chesssight] new best corner error {score:.2f}px", flush=True)
        save(model, config, config.output_dir / "last")
        (config.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    elapsed = time.monotonic() - started
    print(
        f"[chesssight] finished {config.epochs} epochs in {elapsed / 60:.1f} min",
        flush=True,
    )
    return {"history": history, "best_corner_px": best, "elapsed_seconds": elapsed}


@torch.no_grad()
def evaluate_samples(
    model,
    reader: DatasetReader,
    device: torch.device,
    *,
    split: str = "test",
    input_size: int = 448,
    stride: int = 4,
    limit: int | None = None,
    min_score: float = 0.0,
    split_spec: SplitSpec | None = None,
    progress=None,
) -> dict:
    """Corner error on real photographs, in pixels *and* in squares.

    Squares is the number that transfers. A 40-pixel error is a fifth of a square
    on a close-up and two squares on a board across the room, and only one of
    those reads a position correctly.

    ``weakest_peak`` is reported alongside because ``found_rate`` at
    ``min_score=0`` is 100% by construction -- the decoder returns its four best
    cells whether or not they are corners. The confidence of the *fourth* peak is
    what says whether the model actually found a board, so it is measured rather
    than assumed, and it is what a threshold should be set from.
    """
    from PIL import Image

    from chesssight.train.dataset import select_entries
    from chesssight.train.heatmap import peaks_in_image, preprocess

    entries, _ = select_entries(
        reader.entries(), split=split, spec=split_spec or SplitSpec()
    )
    if limit is not None:
        entries = entries[:limit]

    pixels: list[float] = []
    squares: list[float] = []
    weakest: list[float] = []
    found = 0
    for index, entry in enumerate(entries, start=1):
        sample = reader.load(entry.id)
        image = Image.open(reader.root / sample.image).convert("RGB")
        truth = [[float(x), float(y)] for x, y in sample.board.corners_px]

        logits = model(preprocess(image, input_size).to(device)).float().cpu()
        peaks = peaks_in_image(
            logits, size=image.size, input_size=input_size, stride=stride
        )
        weakest.append(min(score for _, _, score in peaks))
        quad = quad_from_logits(
            logits,
            size=image.size,
            input_size=input_size,
            stride=stride,
            min_score=min_score,
        )
        if quad is None:
            continue
        found += 1
        error = corner_error(quad, truth)
        if error is None:
            continue
        pixels.append(error)
        squares.append(error / square_size(truth))
        if progress and index % 50 == 0:
            progress(f"[chesssight] {index}/{len(entries)} photographs")

    return {
        "boards": len(entries),
        "found_rate": found / max(1, len(entries)),
        "weakest_peak": float(np.mean(weakest)) if weakest else float("nan"),
        "weakest_peak_p10": (
            float(np.percentile(weakest, 10)) if weakest else float("nan")
        ),
        "corner_px": float(np.mean(pixels)) if pixels else float("nan"),
        "corner_px_median": float(np.median(pixels)) if pixels else float("nan"),
        "corner_px_p90": float(np.percentile(pixels, 90)) if pixels else float("nan"),
        "corner_squares": float(np.mean(squares)) if squares else float("nan"),
        "corner_squares_median": (
            float(np.median(squares)) if squares else float("nan")
        ),
    }


@torch.no_grad()
def fit_gate(
    model,
    reader: DatasetReader,
    device: torch.device,
    *,
    split: str = "val",
    input_size: int = 448,
    stride: int = 4,
    limit: int | None = None,
    split_spec: SplitSpec | None = None,
) -> "Gate":
    """Fit the peak-confidence threshold that decides when to refuse a board.

    Fit on validation, never on test: the threshold is a parameter like any
    other, and choosing it on the split the result is quoted from would make
    that result a description of the fitting data.
    """
    from PIL import Image

    from chesssight.train.dataset import select_entries
    from chesssight.train.gate import fit as fit_threshold
    from chesssight.train.heatmap import peaks_in_image, preprocess, square_size

    entries, _ = select_entries(
        reader.entries(), split=split, spec=split_spec or SplitSpec()
    )
    if limit is not None:
        entries = entries[:limit]

    scores: list[float] = []
    errors: list[float | None] = []
    for entry in entries:
        sample = reader.load(entry.id)
        image = Image.open(reader.root / sample.image).convert("RGB")
        truth = [[float(x), float(y)] for x, y in sample.board.corners_px]
        logits = model(preprocess(image, input_size).to(device)).float().cpu()
        peaks = peaks_in_image(
            logits, size=image.size, input_size=input_size, stride=stride
        )
        scores.append(min(score for _, _, score in peaks))
        quad = quad_from_logits(
            logits, size=image.size, input_size=input_size, stride=stride
        )
        error = corner_error(quad, truth) if quad else None
        errors.append(None if error is None else error / square_size(truth))

    return fit_threshold(scores, errors, split=split)


def format_report(metrics: dict) -> str:
    return (
        f"  boards {metrics['boards']}   four corners found on "
        f"{metrics['found_rate']:.1%}\n"
        f"  weakest of the four peaks: mean {metrics['weakest_peak']:.3f}  "
        f"p10 {metrics['weakest_peak_p10']:.3f}\n"
        f"  error  mean {metrics['corner_px']:.1f}px  "
        f"median {metrics['corner_px_median']:.1f}px  "
        f"p90 {metrics['corner_px_p90']:.1f}px\n"
        f"  error  mean {metrics['corner_squares']:.3f} squares  "
        f"median {metrics['corner_squares_median']:.3f} squares"
    )
