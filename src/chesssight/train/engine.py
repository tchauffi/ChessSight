"""Fine-tune an RT-DETR detector on a ChessSight run.

RT-DETR rather than plain DETR: same family, redesigned so it converges in tens of
epochs instead of hundreds. Plain DETR's slow convergence is a property of its
one-to-one Hungarian matching with dense attention, and it shows up most on small
objects -- which here means every piece on a board seen from across the table.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from chesssight.train.dataset import (
    ChessDetectionDataset,
    SplitSpec,
    annotates_off_board,
    build_mixed,
    collate,
)
from chesssight.train.labels import ID2LABEL, LABEL2ID, NUM_DETECTION_LABELS

DEFAULT_MODEL = "PekingU/rtdetr_r50vd_coco_o365"


@dataclass
class TrainConfig:
    """Everything that defines a fine-tuning run."""

    #: One or more run directories. Several are concatenated, which is how a
    #: synthetic set and real photographs get trained on together.
    data_roots: list[Path]
    output_dir: Path
    model_name: str = DEFAULT_MODEL
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    #: The pretrained backbone wants a much smaller step than the fresh head; using
    #: one rate for both either destroys the features or barely trains the head.
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    grad_clip: float = 0.1
    num_workers: int = 4
    val_fraction: float = 0.1
    #: Held out of training alongside the validation set and never scored during the
    #: run. Validation both selects the checkpoint and fits the score calibration,
    #: so it is no longer an unbiased estimate by the time the run ends.
    test_fraction: float = 0.1
    image_size: int = 640
    amp: bool = True
    #: Train-time augmentation. Off by default so a run without it is the baseline
    #: any measurement of its value is made against.
    augment: bool = False
    seed: int = 0
    limit: int | None = None
    #: Per-dataset oversampling. Real photographs are the target domain and there
    #: are far fewer of them, so one pass each per epoch wastes most of their value.
    repeats: list[int] = field(default_factory=list)
    #: Which dataset to validate against, by index into ``data_roots``. Defaults to
    #: the last, which is the real set in a synthetic-plus-real mix.
    eval_dataset: int = -1
    eval_split: str = "val"
    #: What "best" means. mAP by default, not loss: the DETR loss sums
    #: classification, L1 and GIoU weighted 1/5/2, so it is dominated by box
    #: regression while mAP depends on ranking quality the loss barely reflects.
    #: Measured on one run the two disagreed by 8 mAP points, and selecting on
    #: loss discarded the better model.
    select_metric: str = "map"
    #: Multiplier on RT-DETR's classification (varifocal) loss. The stock 1.0 is
    #: dwarfed by the 5.0 L1 + 2.0 GIoU box terms, which is a large part of why a
    #: short fine-tune ranks well but scores everything under 0.1: the box branch
    #: gets most of the gradient and the class logits never grow. None keeps the
    #: stock weight.
    cls_loss_weight: float | None = None
    #: Keep an exponential moving average of the weights and evaluate/save *that*,
    #: as the reference RT-DETR training does. The averaged model is what smooths
    #: out the step-to-step noise of a small-batch fine-tune; the official recipe
    #: credits it with a steady fraction of a point of mAP and much steadier logits.
    ema: bool = True
    ema_decay: float = 0.9999
    #: The ramp below means early steps average aggressively (the raw weights are
    #: still mostly the random init, not worth preserving) and the full decay only
    #: applies once training has found its footing.
    ema_warmup_steps: int = 2000
    eval_every: int = 1
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["data_roots"] = [str(root) for root in self.data_roots]
        payload["output_dir"] = str(self.output_dir)
        return json.dumps(payload, indent=2)


def resolve_device(explicit: str | None = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    model_name: str = DEFAULT_MODEL, *, cls_loss_weight: float | None = None
):
    """Load a pretrained detector and swap in a 13-class head.

    ``ignore_mismatched_sizes`` is what allows the COCO 80-class head to be
    replaced; without it the load fails rather than reinitialising.
    """
    overrides = {}
    if cls_loss_weight is not None:
        overrides["weight_loss_vfl"] = cls_loss_weight
    return AutoModelForObjectDetection.from_pretrained(
        model_name,
        num_labels=NUM_DETECTION_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        **overrides,
    )


def build_processor(model_name: str = DEFAULT_MODEL, image_size: int = 640):
    return AutoImageProcessor.from_pretrained(
        model_name,
        size={"height": image_size, "width": image_size},
        do_pad=True,
        use_fast=True,
    )


class ModelEma:
    """Exponential moving average of a model's parameters.

    The averaged copy never sees a gradient; it trails the live weights with
    ``decay`` mixed per update. ``decay`` itself ramps in over ``warmup_steps`` as
    ``decay * (1 - exp(-updates / warmup_steps))`` -- the same schedule as the
    reference implementation -- so the average forgets the earliest, still-random
    steps quickly instead of carrying them for the whole run.

    Buffers (batch-norm statistics and the like) are copied outright rather than
    averaged: an average of integer counters is not a meaningful state.
    """

    def __init__(self, model, *, decay: float = 0.9999, warmup_steps: int = 2000):
        import copy

        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.updates = 0

    def current_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return self.decay * (1.0 - math.exp(-self.updates / self.warmup_steps))

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        decay = self.current_decay()
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            target = ema_state[key]
            if value.dtype.is_floating_point:
                target.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                target.copy_(value)


def parameter_groups(model, config: TrainConfig) -> list[dict]:
    """Separate the pretrained backbone from everything trained from scratch."""
    backbone: list[torch.nn.Parameter] = []
    head: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if "backbone" in name else head).append(parameter)
    return [
        {"params": backbone, "lr": config.backbone_learning_rate},
        {"params": head, "lr": config.learning_rate},
    ]


def cosine_schedule(optimizer, total_steps: int, warmup_fraction: float):
    warmup = max(1, int(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_loaders(config: TrainConfig, processor) -> tuple[DataLoader, DataLoader]:
    spec = SplitSpec(
        val_fraction=config.val_fraction, test_fraction=config.test_fraction
    )
    repeats = config.repeats or [1] * len(config.data_roots)

    transform = None
    if config.augment:
        from chesssight.train.augment import AugmentConfig, build_transform

        transform = build_transform(AugmentConfig(image_size=config.image_size))

    train_set = build_mixed(
        config.data_roots,
        processor,
        split="train",
        split_spec=spec,
        repeats=repeats,
        transform=transform,
    )

    # Validate against one dataset rather than the mix. A loss averaged over
    # synthetic and real together tracks neither, and the number worth watching is
    # the one on the domain the model has to work in.
    eval_root = config.data_roots[config.eval_dataset]
    val_set = ChessDetectionDataset(
        eval_root,
        processor,
        split=config.eval_split,
        split_spec=spec,
        include_off_board=all(annotates_off_board(root) for root in config.data_roots),
        limit=config.limit,
    )

    return (
        DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=config.num_workers > 0,
        ),
        DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=config.num_workers > 0,
        ),
    )


def _move(batch: dict, device: torch.device) -> dict:
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "labels": [
            {key: value.to(device) for key, value in label.items()}
            for label in batch["labels"]
        ],
    }


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    config: TrainConfig,
    *,
    epoch: int,
    ema: ModelEma | None = None,
    log_every: int = 50,
) -> float:
    model.train()
    running = 0.0
    seen = 0
    started = time.monotonic()

    for step, batch in enumerate(loader, start=1):
        batch = _move(batch, device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"
        ):
            outputs = model(**batch)
            loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # DETR-family models are sensitive to gradient spikes early on; the
        # reference implementations all clip, and skipping it shows up as a loss
        # that diverges in the first few hundred steps.
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        running += loss.item()
        seen += 1
        if step % log_every == 0:
            rate = step * config.batch_size / (time.monotonic() - started)
            print(
                f"  epoch {epoch} step {step}/{len(loader)} "
                f"loss {running / seen:.4f} lr {scheduler.get_last_lr()[-1]:.2e} "
                f"{rate:.1f} img/s",
                flush=True,
            )
    return running / max(1, seen)


@torch.no_grad()
def validation_loss(
    model, loader: DataLoader, device: torch.device, config: TrainConfig
) -> float:
    model.train()  # the loss head only runs in train mode; grads are off regardless
    total, seen = 0.0, 0
    for batch in loader:
        batch = _move(batch, device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=config.amp and device.type == "cuda"
        ):
            total += model(**batch).loss.item()
        seen += 1
    return total / max(1, seen)
