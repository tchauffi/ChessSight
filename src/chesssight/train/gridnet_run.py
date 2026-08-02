"""Train and evaluate the rectified-board grid classifier."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from chesssight.data.dataset import DatasetReader
from chesssight.data.fen import BOARD_SIZE
from chesssight.train.dataset import (
    RectifiedBoardDataset,
    SplitSpec,
    collate_grid,
    describe_split,
)
from chesssight.train.engine import cosine_schedule, resolve_device
from chesssight.train.gridnet import GridConfig, GridNet, read_grid, save


def build_loaders(config: GridConfig) -> tuple[DataLoader, DataLoader]:
    spec = SplitSpec(
        val_fraction=config.val_fraction, test_fraction=config.test_fraction
    )
    photometric = None
    if config.augment:
        from torchvision.transforms import v2

        from chesssight.train.augment import AugmentConfig, photometric_steps

        # Photometric only. Geometry is *already* canonical here, and a crop or
        # rotation after rectification would undo the alignment the whole design
        # rests on -- the corner jitter below is the geometric augmentation.
        photometric = v2.Compose(photometric_steps(AugmentConfig()))

    def make(root: Path, split: str, jittered: bool) -> RectifiedBoardDataset:
        return RectifiedBoardDataset(
            root,
            image_size=config.image_size,
            side_margin=config.side_margin,
            far_margin=config.far_margin,
            corner_jitter=config.corner_jitter if jittered else 0.0,
            split=split,
            split_spec=spec,
            limit=config.limit,
            transform=photometric if jittered else None,
            seed=config.seed,
        )

    parts = [make(root, "train", True) for root in config.data_roots]
    train_loader = DataLoader(
        parts[0] if len(parts) == 1 else ConcatDataset(parts),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_grid,
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
        collate_fn=collate_grid,
        pin_memory=True,
    )
    return train_loader, val_loader


def score(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Per-square accuracy and the fraction of boards that are exactly right."""
    predicted = logits.argmax(dim=1)
    correct = predicted.eq(target)
    per_square = float(correct.float().mean())
    exact = float(correct.flatten(1).all(dim=1).float().mean())
    return per_square, exact


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    losses: list[float] = []
    squares: list[float] = []
    exacts: list[float] = []
    occupied_correct = occupied_total = 0

    for batch in loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        logits = model(images).float()
        losses.append(float(F.cross_entropy(logits, target)))
        per_square, exact = score(logits, target)
        squares.append(per_square)
        exacts.append(exact)

        # Reported separately because empty squares are the majority class: a
        # model that predicted "empty" everywhere would score around 60% and
        # look like it had learned something.
        occupied = target > 0
        occupied_total += int(occupied.sum())
        occupied_correct += int(logits.argmax(dim=1).eq(target)[occupied].sum())

    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "per_square": float(np.mean(squares)) if squares else float("nan"),
        "board_exact": float(np.mean(exacts)) if exacts else float("nan"),
        "occupied": occupied_correct / max(1, occupied_total),
    }


def train(config: GridConfig, device: str | None = None) -> dict:
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
        f"[chesssight] grid classifier: {config.backbone} on rectified boards at "
        f"{config.image_size}px, corner jitter {config.corner_jitter} squares "
        f"on {resolved}",
        flush=True,
    )

    model = GridNet(
        config.backbone,
        pretrained=config.pretrained,
        image_size=config.image_size,
        channels=config.channels,
        cells_per_square=config.cells_per_square,
        feature_stride=config.feature_stride,
        side_margin=config.side_margin,
        far_margin=config.far_margin,
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
    best = -float("inf")
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
                loss = F.cross_entropy(model(images).float(), target)
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
                    f"loss {total / batches:.4f}",
                    flush=True,
                )

        metrics = validate(model, val_loader, resolved)
        record = {"epoch": epoch, "train_loss": total / max(1, batches), **metrics}
        history.append(record)
        print(
            f"[chesssight] epoch {epoch}: train {record['train_loss']:.4f} "
            f"val {metrics['val_loss']:.4f}  per-square {metrics['per_square']:.4%}  "
            f"occupied {metrics['occupied']:.4%}  "
            f"board-exact {metrics['board_exact']:.2%}",
            flush=True,
        )
        # Selected on board-exact: it is the thing the project is for, and it is
        # far more sensitive than per-square, where one wrong square in 64 is a
        # 1.6% move.
        if metrics["board_exact"] > best:
            best = metrics["board_exact"]
            save(model, config, config.output_dir / "best")
            print(f"[chesssight] new best board-exact {best:.2%}", flush=True)
        save(model, config, config.output_dir / "last")
        (config.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    elapsed = time.monotonic() - started
    print(
        f"[chesssight] finished {config.epochs} epochs in {elapsed / 60:.1f} min",
        flush=True,
    )
    return {"history": history, "best_board_exact": best, "elapsed_seconds": elapsed}


@torch.no_grad()
def evaluate_samples(
    model,
    reader: DatasetReader,
    device: torch.device,
    *,
    split: str = "test",
    limit: int | None = None,
    corner_model: Path | None = None,
    split_spec: SplitSpec | None = None,
    progress=None,
) -> dict:
    """Score the grid classifier on a dataset.

    ``corner_model`` supplies the geometry the way deployment would. Without it
    the annotated corners are used, which measures classification alone -- a
    useful number, but not the pipeline's.
    """
    from PIL import Image

    from chesssight.train.dataset import select_entries

    entries, _ = select_entries(
        reader.entries(), split=split, spec=split_spec or SplitSpec()
    )
    if limit is not None:
        entries = entries[:limit]

    corners_from = None
    if corner_model is not None:
        from chesssight.train.heatmap import load as load_corners
        from chesssight.train.heatmap import predict_quad

        net, corner_config = load_corners(corner_model, device)

        def corners_from(image):  # noqa: F811
            return predict_quad(
                net,
                image,
                device,
                input_size=corner_config.image_size,
                stride=corner_config.stride,
            )

    squares: list[float] = []
    exacts: list[float] = []
    no_geometry = 0
    for index, entry in enumerate(entries, start=1):
        sample = reader.load(entry.id)
        image = Image.open(reader.root / sample.image).convert("RGB")
        truth = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for square_index, square in enumerate(sample.squares):
            truth[square_index // BOARD_SIZE][square_index % BOARD_SIZE] = (
                square.occupant or 0
            )

        if corners_from is not None:
            corners = corners_from(image)
            if corners is None:
                no_geometry += 1
                continue
        else:
            corners = [[float(x), float(y)] for x, y in sample.board.corners_px]

        grid = read_grid(model, image, corners, device)
        correct = sum(
            1
            for rank in range(BOARD_SIZE)
            for file in range(BOARD_SIZE)
            if grid[rank][file] == truth[rank][file]
        )
        squares.append(correct / (BOARD_SIZE * BOARD_SIZE))
        exacts.append(float(correct == BOARD_SIZE * BOARD_SIZE))
        if progress and index % 50 == 0:
            progress(f"[chesssight] {index}/{len(entries)}")

    return {
        "boards": len(entries),
        "geometry_from": "model" if corner_model else "annotation",
        "no_geometry": no_geometry,
        "per_square": float(np.mean(squares)) if squares else float("nan"),
        "board_exact": float(np.mean(exacts)) if exacts else float("nan"),
    }


def format_report(metrics: dict) -> str:
    return (
        f"  boards {metrics['boards']}   geometry from {metrics['geometry_from']}"
        f"   no geometry on {metrics['no_geometry']}\n"
        f"  per-square {metrics['per_square']:.2%}   "
        f"board-exact {metrics['board_exact']:.2%}"
    )
