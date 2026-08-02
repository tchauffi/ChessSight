"""Train and evaluate the box-relative corner regressor."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from chesssight.data.dataset import DatasetReader
from chesssight.train.boxcorners import (
    BoxCornerConfig,
    BoxCornerNet,
    crop_box,
    predict_quad,
    save,
)
from chesssight.train.corners import corner_error
from chesssight.train.dataset import (
    BoxCornerDataset,
    SplitSpec,
    collate_box_corners,
    describe_split,
)
from chesssight.train.engine import cosine_schedule, resolve_device
from chesssight.train.heatmap import square_size


def build_loaders(config: BoxCornerConfig) -> tuple[DataLoader, DataLoader]:
    spec = SplitSpec(
        val_fraction=config.val_fraction, test_fraction=config.test_fraction
    )

    def make(root: Path, split: str, jittered: bool) -> BoxCornerDataset:
        return BoxCornerDataset(
            root,
            image_size=config.image_size,
            margin=config.margin,
            jitter_scale=config.jitter_scale if jittered else 0.0,
            jitter_shift=config.jitter_shift if jittered else 0.0,
            split=split,
            split_spec=spec,
            limit=config.limit,
            seed=config.seed,
        )

    parts = [make(root, "train", config.augment) for root in config.data_roots]
    train_loader = DataLoader(
        parts[0] if len(parts) == 1 else ConcatDataset(parts),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_box_corners,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        make(
            config.val_root or config.data_roots[config.eval_dataset],
            config.eval_split,
            False,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_box_corners,
        pin_memory=True,
    )
    return train_loader, val_loader


def loss_fn(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Smooth L1 on box-space coordinates, minimised over the four rotations.

    Smooth rather than plain L2 because the interesting cases -- corners well
    outside the crop -- are exactly the large residuals, and squaring them lets a
    handful of extreme boards dominate every batch they appear in.

    Rotation-invariant because no canonical ordering of four interchangeable
    corners is continuous. Ordering by "nearest the crop's top-left" flips
    between two near-equidistant corners, so two photographs that look the same
    carry different labels and the regression is asked to fit a discontinuity
    that is not in the image. Scoring against whichever rotation fits best
    removes the demand entirely; which corner is a8 is decided downstream by
    :mod:`chesssight.train.orientation`, as it already is for every other path.
    """
    losses = torch.stack(
        [
            F.smooth_l1_loss(
                predicted, target.roll(shifts, dims=1), beta=0.05, reduction="none"
            ).mean(dim=(1, 2))
            for shifts in range(predicted.shape[1])
        ],
        dim=0,
    )
    return losses.min(dim=0).values.mean()


@torch.no_grad()
def validate(model, loader, device) -> dict:
    """Loss, plus the error in units of a *square*, which is what transfers."""
    model.eval()
    losses, errors = [], []
    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        predicted = model(images).float()
        losses.append(float(loss_fn(predicted, target)))

        # A board spans 8 squares, so one square is an eighth of the mean side of
        # the true quad -- computed in box space, where the model works.
        for index in range(images.shape[0]):
            truth = target[index].cpu().numpy().tolist()
            guess = predicted[index].cpu().numpy().tolist()
            error = corner_error(guess, truth)
            if error is not None:
                errors.append(error / max(1e-6, square_size(truth)))

    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "corner_squares": float(np.mean(errors)) if errors else float("nan"),
        "corner_squares_median": (float(np.median(errors)) if errors else float("nan")),
    }


def train(config: BoxCornerConfig, device: str | None = None) -> dict:
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
    print(
        f"[chesssight] box corners: {config.backbone} at {config.image_size}px, "
        f"margin {config.margin}, jitter {config.jitter_scale}/{config.jitter_shift} "
        f"on {resolved}",
        flush=True,
    )

    model = BoxCornerNet(
        config.backbone, pretrained=config.pretrained, image_size=config.image_size
    ).to(resolved)
    train_loader, val_loader = build_loaders(config)

    backbone_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        (backbone_params if name.startswith("backbone.") else head_params).append(
            parameter
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
        model.train()
        total, batches = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["pixel_values"].to(resolved, non_blocking=True)
            target = batch["target"].to(resolved, non_blocking=True)
            with torch.autocast(
                resolved.type, dtype=torch.bfloat16, enabled=resolved.type == "cuda"
            ):
                loss = loss_fn(model(images).float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach())
            batches += 1
            if step % 100 == 0:
                print(
                    f"[chesssight] epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss {total / batches:.5f}",
                    flush=True,
                )

        metrics = validate(model, val_loader, resolved)
        record = {"epoch": epoch, "train_loss": total / max(1, batches), **metrics}
        history.append(record)
        print(
            f"[chesssight] epoch {epoch}: train {record['train_loss']:.5f} "
            f"val {metrics['val_loss']:.5f}  corner {metrics['corner_squares']:.4f} sq "
            f"(median {metrics['corner_squares_median']:.4f})",
            flush=True,
        )
        score = metrics["corner_squares"]
        if not np.isnan(score) and score < best:
            best = score
            save(model, config, config.output_dir / "best")
            print(f"[chesssight] new best corner error {score:.4f} squares", flush=True)
        save(model, config, config.output_dir / "last")
        (config.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    elapsed = time.monotonic() - started
    print(
        f"[chesssight] finished {config.epochs} epochs in {elapsed / 60:.1f} min",
        flush=True,
    )
    return {"history": history, "best_corner_squares": best, "elapsed_seconds": elapsed}


@torch.no_grad()
def evaluate_samples(
    model,
    reader: DatasetReader,
    device: torch.device,
    *,
    split: str = "test",
    image_size: int = 224,
    margin: float = 0.25,
    limit: int | None = None,
    detector: Path | None = None,
    split_spec: SplitSpec | None = None,
    progress=None,
) -> dict:
    """Corner error on a dataset, in squares.

    ``detector`` supplies the board box the way deployment would. Without it the
    annotated box is used, which measures the regressor alone and *overstates*
    the pipeline -- so which was used is reported rather than left implicit.
    """
    from PIL import Image

    from chesssight.data.export import board_bbox
    from chesssight.train.dataset import select_entries

    entries, _ = select_entries(
        reader.entries(), split=split, spec=split_spec or SplitSpec()
    )
    if limit is not None:
        entries = entries[:limit]

    detect = None
    if detector is not None:
        from chesssight.train.calibrate import Calibration
        from chesssight.train.labels import BOARD_INDEX
        from chesssight.train.run import load_trained
        from chesssight.train.visualize import predict

        net, processor, _ = load_trained(detector, None)
        calibration = Calibration.load(detector)

        def detect(image):  # noqa: F811
            found = predict(
                net,
                processor,
                image,
                device,
                threshold=calibration.threshold if calibration else 0.3,
                calibration=calibration,
            )
            boards = [d for d in found if d["label"] == BOARD_INDEX]
            if not boards:
                return None
            best_board = max(boards, key=lambda d: float(d["score"]))
            return tuple(float(v) for v in best_board["box"])

    squares, pixels = [], []
    missing_box = 0
    for index, entry in enumerate(entries, start=1):
        sample = reader.load(entry.id)
        image = Image.open(reader.root / sample.image).convert("RGB")
        truth = [[float(x), float(y)] for x, y in sample.board.corners_px]

        if detect is not None:
            box = detect(image)
        else:
            annotated = board_bbox(sample)
            box = (
                None
                if annotated is None
                else (
                    annotated.x,
                    annotated.y,
                    annotated.x + annotated.width,
                    annotated.y + annotated.height,
                )
            )
        if box is None:
            missing_box += 1
            continue

        quad = predict_quad(
            model, image, box, device, image_size=image_size, margin=margin
        )
        error = corner_error(quad, truth)
        if error is None:
            continue
        pixels.append(error)
        squares.append(error / square_size(truth))
        if progress and index % 50 == 0:
            progress(f"[chesssight] {index}/{len(entries)}")

    return {
        "boards": len(entries),
        "box_source": "detector" if detector else "annotation",
        "missing_box": missing_box,
        "corner_px": float(np.mean(pixels)) if pixels else float("nan"),
        "corner_px_median": float(np.median(pixels)) if pixels else float("nan"),
        "corner_squares": float(np.mean(squares)) if squares else float("nan"),
        "corner_squares_median": (
            float(np.median(squares)) if squares else float("nan")
        ),
        "corner_squares_p90": (
            float(np.percentile(squares, 90)) if squares else float("nan")
        ),
    }


def format_report(metrics: dict) -> str:
    return (
        f"  boards {metrics['boards']}   box from {metrics['box_source']}"
        f"   no box on {metrics['missing_box']}\n"
        f"  error  mean {metrics['corner_px']:.1f}px  "
        f"median {metrics['corner_px_median']:.1f}px\n"
        f"  error  mean {metrics['corner_squares']:.3f} sq  "
        f"median {metrics['corner_squares_median']:.3f} sq  "
        f"p90 {metrics['corner_squares_p90']:.3f} sq"
    )


__all__ = ["build_loaders", "crop_box", "evaluate_samples", "format_report", "train"]
