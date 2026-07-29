"""Drive a fine-tuning run end to end."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from chesssight.train.dataset import SplitSpec, describe_split
from chesssight.train.engine import (
    TrainConfig,
    build_loaders,
    build_model,
    build_processor,
    cosine_schedule,
    parameter_groups,
    resolve_device,
    train_one_epoch,
    validation_loss,
)
from chesssight.train.evaluate import evaluate_samples, format_report


def train(config: TrainConfig, device: str | None = None) -> dict:
    torch.manual_seed(config.seed)
    resolved = resolve_device(device)

    repeats = config.repeats or [1] * len(config.data_roots)
    for root, repeat in zip(config.data_roots, repeats, strict=True):
        counts = describe_split(root, SplitSpec(config.val_fraction))
        suffix = f" x{repeat}" if repeat > 1 else ""
        print(
            f"[chesssight] {counts['total']} samples from {root}{suffix}",
            flush=True,
        )
    eval_root = config.data_roots[config.eval_dataset]
    print(
        f"[chesssight] validating on {eval_root} split {config.eval_split!r}",
        flush=True,
    )

    print(f"[chesssight] model {config.model_name} on {resolved}", flush=True)

    processor = build_processor(config.model_name, config.image_size)
    model = build_model(config.model_name).to(resolved)
    train_loader, val_loader = build_loaders(config, processor)

    # Validation scores detections the same way the reported result does. The
    # loader path cannot apply the on-board filter -- it yields tensors, not
    # samples -- so without this the selected checkpoint is chosen on a number
    # dominated by captured pieces being counted as false positives. Measured on
    # one checkpoint that was 0.80 against 0.88, and 0.49 against 0.89 for the
    # black king alone.
    from chesssight.data.dataset import DatasetReader
    from chesssight.train.dataset import annotates_off_board

    eval_reader = DatasetReader(eval_root)
    eval_on_board = not annotates_off_board(eval_root)
    print(
        f"[chesssight] validation on-board filter: "
        f"{'on' if eval_on_board else 'off'}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        parameter_groups(model, config), weight_decay=config.weight_decay
    )
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = cosine_schedule(optimizer, total_steps, config.warmup_fraction)
    # bfloat16 needs no loss scaling, but keeping a (disabled) scaler keeps one
    # code path for both precisions.
    scaler = torch.amp.GradScaler(enabled=False)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(config.to_json(), encoding="utf-8")

    history: list[dict] = []
    higher_is_better = config.select_metric != "val_loss"
    best = -float("inf") if higher_is_better else float("inf")
    started = time.monotonic()

    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            resolved,
            config,
            epoch=epoch,
        )
        record = {"epoch": epoch, "train_loss": loss}

        if epoch % config.eval_every == 0 or epoch == config.epochs:
            record["val_loss"] = validation_loss(model, val_loader, resolved, config)
            metrics = evaluate_samples(
                model,
                processor,
                eval_reader,
                resolved,
                split=config.eval_split,
                on_board=eval_on_board,
            )
            record.update(metrics)
            print(
                f"[chesssight] epoch {epoch}: train {loss:.4f} "
                f"val {record['val_loss']:.4f}",
                flush=True,
            )
            print(format_report(metrics), flush=True)

            score = record.get(config.select_metric)
            if score is None:
                raise KeyError(
                    f"select_metric {config.select_metric!r} is not in the "
                    f"reported metrics: {sorted(record)}"
                )
            improved = score > best if higher_is_better else score < best
            if improved:
                best = score
                model.save_pretrained(config.output_dir / "best")
                processor.save_pretrained(config.output_dir / "best")
                print(
                    f"[chesssight] new best {config.select_metric}={score:.4f}, "
                    f"saved to {config.output_dir / 'best'}",
                    flush=True,
                )
            # Keep the most recent epoch too, so a run that is stopped early still
            # leaves something to inspect besides the selected checkpoint.
            model.save_pretrained(config.output_dir / "last")
            processor.save_pretrained(config.output_dir / "last")
        else:
            print(f"[chesssight] epoch {epoch}: train {loss:.4f}", flush=True)

        history.append(record)
        (config.output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    model.save_pretrained(config.output_dir / "last")
    processor.save_pretrained(config.output_dir / "last")

    elapsed = time.monotonic() - started
    print(
        f"[chesssight] finished {config.epochs} epochs in {elapsed / 60:.1f} min",
        flush=True,
    )
    return {
        "history": history,
        "best_score": best,
        "select_metric": config.select_metric,
        "elapsed_seconds": elapsed,
    }


def load_trained(path: Path, device: str | None = None):
    """Reload a saved checkpoint with its processor."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    resolved = resolve_device(device)
    model = AutoModelForObjectDetection.from_pretrained(path).to(resolved).eval()
    processor = AutoImageProcessor.from_pretrained(path)
    return model, processor, resolved
