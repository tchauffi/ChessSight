"""``chesssight`` command line interface."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import typer

from chesssight.data.dataset import DatasetReader
from chesssight.data.geometry import is_mirrored
from chesssight.data.qa import contact_sheet, render_overlay
from chesssight.synth import runner
from chesssight.synth.config import GeneratorConfig

app = typer.Typer(
    help="Synthetic chess-board dataset generation.", no_args_is_help=True
)
synth_app = typer.Typer(help="Generate synthetic renders.", no_args_is_help=True)
qa_app = typer.Typer(help="Inspect a generated dataset.", no_args_is_help=True)
assets_app = typer.Typer(
    help="Use an external chess set instead of the procedural one.",
    no_args_is_help=True,
)
app.add_typer(synth_app, name="synth")
app.add_typer(qa_app, name="qa")
app.add_typer(assets_app, name="assets")

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="Path to a generator YAML config.")
]
OutOption = Annotated[
    Path | None, typer.Option("--out", "-o", help="Run directory (overrides config).")
]


def load_config(path: Path | None, **overrides: object) -> GeneratorConfig:
    config = GeneratorConfig.from_yaml(path) if path else GeneratorConfig()
    supplied = {key: value for key, value in overrides.items() if value is not None}
    if supplied:
        config = config.model_copy(update=supplied)
    return config


@app.command()
def doctor() -> None:
    """Check that Blender, the GPU and the output directory are usable.

    Run this first when something misbehaves -- most generation problems are an
    environment issue rather than a bug.
    """
    blender = shutil.which("blender")
    typer.echo(f"blender executable : {blender or 'NOT FOUND'}")
    if not blender:
        typer.echo("  install Blender, or pass --blender to `synth run`.")
        raise typer.Exit(1)

    probe = (
        "import bpy, sys;"
        "print('BLENDER_VERSION', '.'.join(str(p) for p in bpy.app.version));"
        "print('PYTHON', sys.version.split()[0]);"
        "prefs = bpy.context.preferences.addons['cycles'].preferences;"
        "prefs.refresh_devices();"
        "print('DEVICES', [(d.name, d.type) for d in prefs.devices])"
    )
    result = subprocess.run(
        [blender, "--background", "--factory-startup", "--python-expr", probe],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    for line in result.stdout.splitlines():
        if line.startswith(("BLENDER_VERSION", "PYTHON", "DEVICES")):
            key, _, value = line.partition(" ")
            typer.echo(f"{key.lower():<19}: {value}")

    default_root = GeneratorConfig().output.run_dir()
    try:
        default_root.mkdir(parents=True, exist_ok=True)
        typer.echo(f"output root        : {default_root} (writable)")
    except OSError as error:
        typer.echo(f"output root        : {default_root} NOT WRITABLE ({error})")
        raise typer.Exit(1) from error


@synth_app.command("plan")
def synth_plan(
    config_path: ConfigOption = None,
    out: OutOption = None,
    count: Annotated[int | None, typer.Option("--count", "-n")] = None,
    workers: Annotated[int, typer.Option("--workers", "-w")] = 1,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
) -> None:
    """Write the run metadata and job shards without rendering anything."""
    config = load_config(config_path, count=count, master_seed=seed)
    _, shards, planned = runner.plan(config, out, workers=workers)
    typer.echo(f"planned {planned} samples across {len(shards)} shard(s)")
    for shard in shards:
        typer.echo(f"  {shard}")


@synth_app.command("run")
def synth_run(
    config_path: ConfigOption = None,
    out: OutOption = None,
    count: Annotated[int | None, typer.Option("--count", "-n")] = None,
    workers: Annotated[int, typer.Option("--workers", "-w")] = 1,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Skip already-indexed samples.")
    ] = False,
    blender: Annotated[Path | None, typer.Option("--blender")] = None,
    no_masks: Annotated[
        bool, typer.Option("--no-masks", help="Skip storing RLE masks to save space.")
    ] = False,
) -> None:
    """Render a dataset."""
    config = load_config(config_path, count=count, master_seed=seed)
    result = runner.run(
        config,
        out,
        workers=workers,
        resume=resume,
        blender=str(blender) if blender else None,
        store_masks=not no_masks,
    )
    if result.failed:
        raise typer.Exit(1)


@synth_app.command("preview")
def synth_preview(
    config_path: ConfigOption = None,
    out: OutOption = None,
    count: Annotated[int, typer.Option("--count", "-n")] = 4,
    sheet: Annotated[Path | None, typer.Option("--sheet")] = None,
) -> None:
    """Render a handful of samples fast and write an annotated contact sheet.

    The quickest way to see whether a config change did what you expected.
    """
    config = load_config(config_path, count=count)
    config = config.model_copy(
        update={"render": config.render.model_copy(update={"engine": "BLENDER_EEVEE"})}
    )
    result = runner.run(config, out, workers=1)
    root = Path(out) if out else config.output.run_dir()

    reader = DatasetReader(root)
    overlays = [
        render_overlay(sample, root / sample.image) for sample in list(reader)[:count]
    ]
    target = Path(sheet) if sheet else root / "contact_sheet.png"
    contact_sheet(overlays, columns=min(4, len(overlays))).save(target)
    typer.echo(f"{result.rendered} samples -> {target}")


@synth_app.command("verify")
def synth_verify(
    out: Annotated[Path, typer.Argument(help="Run directory to check.")],
) -> None:
    """Re-check every sample in a dataset.

    Validation already runs at generation time, so this is for datasets that were
    copied, edited, or produced by an older version of the code.
    """
    reader = DatasetReader(out)
    problems: list[str] = []
    total = 0
    real_in_train = 0
    # A dataset that is entirely real photographs carries its own train/val/test
    # division, and training on it is the point. The thing worth warning about is
    # real images mixed into a *synthetic* training set, where they would stop the
    # held-out number meaning anything.
    all_real = reader.meta().source == "real"

    for sample in reader:
        total += 1
        if not (out / sample.image).is_file():
            problems.append(f"{sample.id}: missing image {sample.image}")
        if is_mirrored(sample.board.corners_px):
            problems.append(f"{sample.id}: board is mirrored")
        error = sample.board.reprojection_error_px
        if error is not None and error > 0.5:
            problems.append(f"{sample.id}: reprojection error {error:.3f} px")
        if sample.source == "real" and sample.split == "train":
            real_in_train += 1
            if not all_real:
                problems.append(
                    f"{sample.id}: real photograph in a synthetic training set; "
                    f"holding real data out is what keeps the number honest"
                )

    typer.echo(f"checked {total} samples, {len(problems)} problem(s)")
    if all_real and real_in_train:
        typer.echo(
            f"  note: {real_in_train} real samples in the train split, which is "
            f"expected for a real dataset with its own splits"
        )
    for problem in problems[:40]:
        typer.echo(f"  {problem}")
    if problems:
        raise typer.Exit(1)


@qa_app.command("overlay")
def qa_overlay(
    out: Annotated[Path, typer.Argument(help="Run directory.")],
    sample_id: Annotated[str | None, typer.Option("--id")] = None,
    count: Annotated[int, typer.Option("--count", "-n")] = 1,
    sheet: Annotated[Path | None, typer.Option("--sheet")] = None,
    names: Annotated[bool, typer.Option("--names/--no-names")] = False,
) -> None:
    """Draw labels onto rendered images."""
    reader = DatasetReader(out)
    samples = [reader.load(sample_id)] if sample_id else list(reader)[:count]
    if not samples:
        typer.echo("no samples found")
        raise typer.Exit(1)

    overlays = [
        render_overlay(sample, out / sample.image, show_names=names)
        for sample in samples
    ]
    if sheet or len(overlays) > 1:
        target = Path(sheet) if sheet else out / "contact_sheet.png"
        contact_sheet(overlays, columns=min(4, len(overlays))).save(target)
        typer.echo(f"wrote {target}")
    else:
        target = out / "overlays" / f"{samples[0].id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        overlays[0].save(target)
        typer.echo(f"wrote {target}")


@qa_app.command("stats")
def qa_stats(
    out: Annotated[Path, typer.Argument(help="Run directory.")],
) -> None:
    """Summarise a dataset: class balance, occupancy and visibility."""
    from chesssight.data.fen import CLASS_NAMES

    reader = DatasetReader(out)
    class_counts: dict[int, int] = {}
    occupied = squares = visible = pieces = 0
    partial_boards = 0
    total = 0

    for sample in reader:
        total += 1
        squares += len(sample.squares)
        occupied += sum(1 for square in sample.squares if square.occupant != 0)
        partial_boards += not sample.board.all_corners_in_frame
        for piece in sample.pieces:
            pieces += 1
            visible += piece.visible
            class_counts[piece.class_id] = class_counts.get(piece.class_id, 0) + 1

    if not total:
        typer.echo("dataset is empty")
        raise typer.Exit(1)

    typer.echo(f"samples          : {total}")
    typer.echo(f"occupied squares : {occupied / squares:.1%}")
    typer.echo(
        f"visible pieces   : {visible}/{pieces} ({visible / max(pieces, 1):.1%})"
    )
    typer.echo(f"partial boards   : {partial_boards} ({partial_boards / total:.1%})")
    typer.echo("class balance:")
    for class_id in sorted(class_counts):
        count = class_counts[class_id]
        typer.echo(f"  {CLASS_NAMES[class_id]:<14} {count:>7}  {count / pieces:.1%}")


if __name__ == "__main__":
    app()


@assets_app.command("template")
def assets_template(
    directory: Annotated[Path, typer.Argument(help="Where to write the manifest.")],
    name: Annotated[str, typer.Option("--name")] = "my-chess-set",
    extension: Annotated[str, typer.Option("--ext")] = ".obj",
) -> None:
    """Write a manifest skeleton to fill in with your own model files.

    Drop one file per piece into the same directory, then run `assets check`.
    """
    from chesssight.synth.assets import blank_manifest

    directory = Path(directory).expanduser()
    manifest = blank_manifest(name, extension)
    path = manifest.save(directory / "manifest.json")

    typer.echo(f"wrote {path}")
    typer.echo("\nNext:")
    typer.echo(f"  1. put your model files in {directory}")
    typer.echo("  2. set forward_axis / up_axis to match how the set is modelled")
    typer.echo(f"  3. uv run chesssight assets check {path}")


@assets_app.command("export")
def assets_export(
    directory: Annotated[Path, typer.Argument(help="Where to write the OBJ files.")],
    blender: Annotated[Path | None, typer.Option("--blender")] = None,
) -> None:
    """Export the built-in procedural set as OBJ files plus a working manifest.

    Useful as a correctly-oriented starting point: edit the pieces in Blender and
    re-export, or use the manifest as a reference when adapting a downloaded set.
    """
    executable = runner.find_blender(str(blender) if blender else None)
    script = Path(__file__).resolve().parent / "blender" / "export_set.py"
    directory = Path(directory).expanduser()

    result = subprocess.run(
        [
            executable,
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            "--out",
            str(directory),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if line.startswith("[chesssight]"):
            typer.echo(line)
    if result.returncode != 0:
        typer.echo(result.stderr[-2000:], err=True)
        raise typer.Exit(1)


@assets_app.command("check")
def assets_check(
    manifest_path: Annotated[Path, typer.Argument(help="Path to manifest.json")],
) -> None:
    """Validate a manifest and confirm every piece file exists and is importable."""
    from chesssight.synth.assets import AssetError, AssetManifest

    manifest_path = Path(manifest_path).expanduser()
    try:
        manifest = AssetManifest.load(manifest_path)
    except (AssetError, ValueError) as error:
        typer.echo(f"invalid manifest: {error}", err=True)
        raise typer.Exit(1) from error

    problems = manifest.check(manifest_path.parent)
    typer.echo(f"set        : {manifest.name}")
    typer.echo(f"license    : {manifest.license or 'UNSPECIFIED'}")
    if manifest.attribution:
        typer.echo(f"credit     : {manifest.attribution}")
    typer.echo(f"orientation: forward {manifest.forward_axis}, up {manifest.up_axis}")
    typer.echo(f"king height: {manifest.king_height} squares")

    if problems:
        typer.echo(f"\n{len(problems)} problem(s):")
        for problem in problems:
            typer.echo(f"  {problem}")
        raise typer.Exit(1)

    typer.echo("\nall 6 pieces resolve. Use it with:")
    typer.echo(f"  pieces.provider: {manifest.name}")
    typer.echo(f"  pieces.asset_manifest: {manifest_path}")

    from chesssight.synth.asset_spec import licence_warnings

    warnings = licence_warnings(manifest.model_dump())
    if warnings:
        typer.echo("\nlicence notes:")
        for warning in warnings:
            typer.echo(f"  - {warning}")
        typer.echo(
            "  Recorded into meta.json of every dataset built with this set, so "
            "the terms travel with the renders."
        )


@assets_app.command("bake")
def assets_bake(
    manifest_path: Annotated[Path, typer.Argument(help="Manifest to bake.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")],
    decimate: Annotated[
        float | None,
        typer.Option(
            "--decimate", help="Fraction of faces to keep, overriding the manifest."
        ),
    ] = None,
    blender: Annotated[Path | None, typer.Option("--blender")] = None,
) -> None:
    """Pre-normalise a set so rendering does not re-import it for every image.

    Importing a print-ready set costs far more than rendering it, and the scene is
    reset between jobs, so that cost is otherwise paid per image. Baking does the
    import, decimation, orientation and scaling once and writes small OBJ files with
    a manifest that needs no further processing.
    """
    executable = runner.find_blender(str(blender) if blender else None)
    script = Path(__file__).resolve().parent / "blender" / "bake_set.py"

    result = subprocess.run(
        [
            executable,
            "--background",
            "--factory-startup",
            "--python",
            str(script),
            "--",
            "--manifest",
            str(Path(manifest_path).expanduser()),
            "--out",
            str(Path(out).expanduser()),
            *(["--decimate", str(decimate)] if decimate is not None else []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if line.startswith("[chesssight]"):
            typer.echo(line)
    if result.returncode != 0:
        typer.echo(result.stderr[-3000:], err=True)
        raise typer.Exit(1)


data_app = typer.Typer(help="Export and inspect dataset labels.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("masks")
def data_masks(
    out: Annotated[Path, typer.Argument(help="Run directory.")],
    dest: Annotated[
        Path | None,
        typer.Option("--dest", help="Where to write (default <run>/masks)."),
    ] = None,
    previews: Annotated[
        bool,
        typer.Option("--previews", help="Also write colourised versions to look at."),
    ] = False,
    no_board: Annotated[
        bool,
        typer.Option("--no-board", help="Omit the board label from the semantic mask."),
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
) -> None:
    """Write per-pixel segmentation masks decoded from each sample's stored RLE.

    Two images per sample: `semantic/` holds class indices (0 background, 1-12
    pieces, 13 board) and `instance/` holds per-piece ids. Both are label images,
    so they look almost black in a viewer -- that is correct. Use `--previews` for
    colourised copies.
    """
    from chesssight.data.export import write_masks

    reader = DatasetReader(out)
    target = Path(dest) if dest else Path(out) / "masks"
    result = write_masks(
        reader,
        target,
        include_board=not no_board,
        previews=previews,
        limit=limit,
    )
    typer.echo(f"wrote {result['written']} mask pairs to {target}")
    if result["skipped"]:
        typer.echo(
            f"skipped {result['skipped']} samples with no stored masks "
            f"(generated with --no-masks, or real photographs)"
        )


@data_app.command("coco")
def data_coco(
    out: Annotated[Path, typer.Argument(help="Run directory.")],
    dest: Annotated[
        Path | None,
        typer.Option("--dest", help="Output path (default <run>/coco.json)."),
    ] = None,
    no_masks: Annotated[
        bool, typer.Option("--no-masks", help="Boxes only, omit segmentation.")
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
) -> None:
    """Write a COCO instance-segmentation JSON for detection frameworks."""
    from chesssight.data.export import write_coco

    reader = DatasetReader(out)
    target = Path(dest) if dest else Path(out) / "coco.json"
    result = write_coco(reader, target, with_masks=not no_masks, limit=limit)
    typer.echo(
        f"wrote {result['annotations']} annotations over {result['images']} "
        f"images to {target}"
    )


train_app = typer.Typer(
    help="Fine-tune a detector on a generated dataset.", no_args_is_help=True
)
app.add_typer(train_app, name="train")


@train_app.command("detr")
def train_detr(
    data: Annotated[
        list[Path],
        typer.Argument(
            help="One or more run directories. Several are trained on together, "
            "which is how synthetic renders and real photographs get mixed."
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Where to save checkpoints.")
    ],
    model: Annotated[str, typer.Option("--model", help="Pretrained checkpoint.")] = (
        "PekingU/rtdetr_r50vd_coco_o365"
    ),
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 20,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b")] = 8,
    lr: Annotated[float, typer.Option("--lr")] = 1e-4,
    backbone_lr: Annotated[float, typer.Option("--backbone-lr")] = 1e-5,
    image_size: Annotated[int, typer.Option("--image-size")] = 640,
    workers: Annotated[int, typer.Option("--workers", "-w")] = 4,
    val_fraction: Annotated[float, typer.Option("--val-fraction")] = 0.1,
    repeats: Annotated[
        str | None,
        typer.Option(
            "--repeats",
            help="Comma-separated oversampling per dataset, e.g. '1,5'. Real "
            "photographs are the target domain and there are far fewer of them.",
        ),
    ] = None,
    eval_dataset: Annotated[
        int,
        typer.Option(
            "--eval-dataset",
            help="Index of the dataset to validate on. Defaults to the last, "
            "which is the real set in a synthetic-plus-real mix.",
        ),
    ] = -1,
    eval_split: Annotated[str, typer.Option("--eval-split")] = "val",
    select_metric: Annotated[
        str,
        typer.Option(
            "--select-metric",
            help="Which metric picks the best checkpoint. 'map' by default; "
            "'val_loss' is available but tracks mAP poorly on DETR-family models.",
        ),
    ] = "map",
    cls_weight: Annotated[
        float | None,
        typer.Option(
            "--cls-weight",
            help="Multiplier on the classification loss. The stock 1.0 is dwarfed "
            "by the 5+2 box terms, which is why short fine-tunes rank well but "
            "score everything under 0.1. Try 3.0.",
        ),
    ] = None,
    augment: Annotated[
        bool,
        typer.Option(
            "--augment/--no-augment",
            help="Train-time photometric, sensor and crop augmentation. Never "
            "flips or quarter-turns: a mirrored board is a defect the generator "
            "guarantees against, and both remap squares.",
        ),
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Fine-tune an RT-DETR detector to find the board and the pieces.

    RT-DETR rather than plain DETR: same family, but redesigned to converge in
    tens of epochs rather than hundreds. Pass `--model facebook/detr-resnet-50`
    for the original if you want to compare.
    """
    from chesssight.train.engine import TrainConfig
    from chesssight.train.run import train as run_training

    parsed_repeats = [int(value) for value in repeats.split(",")] if repeats else []
    config = TrainConfig(
        data_roots=[Path(root) for root in data],
        output_dir=Path(out),
        repeats=parsed_repeats,
        eval_dataset=eval_dataset,
        eval_split=eval_split,
        select_metric=select_metric,
        cls_loss_weight=cls_weight,
        augment=augment,
        model_name=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        backbone_learning_rate=backbone_lr,
        image_size=image_size,
        num_workers=workers,
        val_fraction=val_fraction,
        limit=limit,
    )
    run_training(config, device=device)


@train_app.command("evaluate")
def train_evaluate(
    checkpoint: Annotated[Path, typer.Argument(help="Saved checkpoint directory.")],
    data: Annotated[Path, typer.Option("--data", help="Run directory to evaluate on.")],
    split: Annotated[str, typer.Option("--split")] = "val",
    batch_size: Annotated[int, typer.Option("--batch-size", "-b")] = 8,
    val_fraction: Annotated[float, typer.Option("--val-fraction")] = 0.1,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Report mAP for a saved checkpoint, overall and per class."""
    from torch.utils.data import DataLoader

    from chesssight.train.dataset import ChessDetectionDataset, SplitSpec, collate
    from chesssight.train.evaluate import evaluate as run_eval
    from chesssight.train.evaluate import format_report
    from chesssight.train.run import load_trained

    model, processor, resolved = load_trained(Path(checkpoint), device)
    dataset = ChessDetectionDataset(
        Path(data),
        processor,
        split=split,
        split_spec=SplitSpec(val_fraction=val_fraction),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate, num_workers=2
    )
    typer.echo(f"evaluating {len(dataset)} samples from split {split!r}")
    typer.echo(format_report(run_eval(model, loader, processor, resolved)))


@data_app.command("ingest-chessred")
def data_ingest_chessred(
    annotations: Annotated[Path, typer.Argument(help="ChessReD annotations.json")],
    images: Annotated[
        Path, typer.Option("--images", help="Extracted images/ directory.")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Run directory to create.")],
    subset: Annotated[str, typer.Option("--subset")] = "chessred2k",
    force_split: Annotated[
        str | None,
        typer.Option("--force-split", help="Put every sample in one split."),
    ] = None,
    copy_images: Annotated[
        bool, typer.Option("--copy-images", help="Copy instead of symlinking.")
    ] = False,
) -> None:
    """Ingest ChessReD real photographs into the ChessSight format.

    Only the annotated subset can be fully ingested: piece positions exist for all
    10,800 images, but the board corners and per-piece boxes this schema needs are
    provided for 2,078 of them.

    ChessReD is CC BY-NC-SA 4.0 -- non-commercial *and* ShareAlike. The terms are
    recorded in the run's meta.json.
    """
    from chesssight.data.chessred import ingest

    result = ingest(
        Path(annotations),
        Path(images),
        Path(out),
        subset=subset,
        force_split=force_split,
        link_images=not copy_images,
    )
    typer.echo(f"ingested {result['written']} samples, skipped {result['skipped']}")
    typer.echo(f"  -> {out}")


def _load_calibration(checkpoint: Path, threshold: float, top_k: int | None):
    """Attach a saved calibration, letting its fitted threshold be the default.

    With calibration the absolute scores mean something, so thresholding takes
    over from ranking; an explicit ``--top-k`` keeps ranking behaviour instead.
    """
    from chesssight.train.calibrate import Calibration

    calibration = Calibration.load(checkpoint)
    if calibration is None or top_k is not None:
        return calibration if top_k is None else None, threshold
    if threshold <= 0:
        threshold = calibration.threshold
    typer.echo(
        f"calibration: applied (fit on {calibration.fit_split!r}, "
        f"threshold {threshold:.2f}, F1 there {calibration.f1:.3f})"
    )
    return calibration, threshold


@train_app.command("predict")
def train_predict(
    checkpoint: Annotated[Path, typer.Argument(help="Saved checkpoint directory.")],
    data: Annotated[Path, typer.Option("--data", help="Run directory to draw from.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Contact sheet to write.")],
    split: Annotated[str, typer.Option("--split")] = "test",
    split_source: Annotated[
        str,
        typer.Option(
            "--split-source",
            help="'stored' uses the dataset's own splits, 'hash' reproduces the "
            "training-time hold-out, 'auto' picks per dataset. A synthetic run "
            "stores one split for everything, so 'stored' would show training "
            "images and overstate the result.",
        ),
    ] = "auto",
    count: Annotated[int, typer.Option("--count", "-n")] = 6,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Score floor. Defaults to 0 because ranking, not thresholding, is "
            "what works on a partly-trained detector.",
        ),
    ] = 0.0,
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            help="Draw the N highest-scoring boxes instead of thresholding. "
            "Use while a model is still training, when scores are low but the "
            "ranking is already good.",
        ),
    ] = None,
    columns: Annotated[int, typer.Option("--columns")] = 2,
    no_truth: Annotated[bool, typer.Option("--no-truth")] = False,
    on_board: Annotated[
        bool | None,
        typer.Option(
            "--on-board/--all-detections",
            help="Keep only detections standing on the board. Defaults to on for a "
            "dataset that annotates on-board pieces only (ChessReD), and off for "
            "one that annotates captured pieces too (our synthetic runs).",
        ),
    ] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Draw a detector's predictions onto images and write a contact sheet.

    Ground truth is drawn alongside in green, because the informative part is the
    difference: a red box with no green under it is a false positive, a green box
    with nothing on it is a miss.
    """
    from PIL import Image

    from chesssight.data.qa import contact_sheet
    from chesssight.train.dataset import SplitSpec
    from chesssight.train.run import load_trained
    from chesssight.train.visualize import (
        draw_predictions,
        on_board_only,
        predict,
        square_accuracy,
        take_top,
    )

    reader = DatasetReader(data)
    all_entries = reader.entries()
    stored_splits = {entry.split for entry in all_entries}

    source = split_source
    if source == "auto":
        source = "stored" if len(stored_splits) > 1 else "hash"

    if source == "stored":
        entries = [e for e in all_entries if split == "all" or e.split == split]
    else:
        # Reproduce the training-time hold-out exactly, so what is drawn is data
        # the model has genuinely not seen.
        spec = SplitSpec()
        wanted_val = split in ("val", "test")
        entries = [
            e for e in all_entries if split == "all" or spec.is_val(e.id) == wanted_val
        ]

    if not entries:
        typer.echo(f"no samples in split {split!r} using {source} splits")
        raise typer.Exit(1)
    typer.echo(f"{len(entries)} samples in split {split!r} ({source} splits)")

    model, processor, resolved = load_trained(Path(checkpoint), device)
    calibration, threshold = _load_calibration(Path(checkpoint), threshold, top_k)
    step = max(1, len(entries) // count)
    chosen = entries[::step][:count]

    if on_board is None:
        # If the dataset annotates pieces beside the board, filtering them out
        # would discard correct detections that its ground truth expects.
        probe = [reader.load(entry.id) for entry in chosen]
        annotates_off_board = any(
            not piece.on_board for sample in probe for piece in sample.pieces
        )
        on_board = not annotates_off_board
        why = (
            "dataset annotates on-board pieces only"
            if on_board
            else "dataset annotates captured pieces too"
        )
        typer.echo(f"on-board filter: {'on' if on_board else 'off'} ({why})")

    panels = []
    accuracies = []
    for entry in chosen:
        sample = reader.load(entry.id)
        image = Image.open(reader.root / sample.image).convert("RGB")
        predictions = predict(
            model,
            processor,
            image,
            resolved,
            threshold=threshold,
            calibration=calibration,
        )
        if on_board:
            predictions = on_board_only(predictions, sample)
        if calibration is None:
            # Without calibration, rank rather than threshold: a partly-trained
            # DETR orders boxes well long before its scores are confident, so a
            # fixed threshold shows an empty image and hides a model that works.
            k = top_k if top_k else sum(1 for p in sample.pieces if p.bbox)
            predictions = take_top(predictions, k)
        elif top_k:
            predictions = take_top(predictions, top_k)
        accuracy = square_accuracy(sample, predictions)
        accuracies.append(accuracy)
        n_predicted = sum(1 for p in predictions if p["name"] != "board")
        n_annotated = sum(1 for p in sample.pieces if p.bbox)
        panels.append(
            draw_predictions(
                image,
                predictions,
                sample=sample,
                show_truth=not no_truth,
                title=(
                    f"{sample.id}  "
                    f"{n_predicted} predicted / {n_annotated} annotated  "
                    f"occupied-square acc {accuracy['occupied_correct']:.0%}"
                ),
            )
        )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet(panels, columns=columns, cell_width=900).save(out)

    mean_occupied = sum(a["occupied_correct"] for a in accuracies) / len(accuracies)
    typer.echo(f"wrote {out}")
    typer.echo(
        f"  mean occupied-square accuracy over {len(chosen)} images: "
        f"{mean_occupied:.1%}"
    )


@train_app.command("calibrate")
def train_calibrate(
    checkpoint: Annotated[Path, typer.Argument(help="Saved checkpoint directory.")],
    data: Annotated[
        Path,
        typer.Option(
            "--data",
            help="Run directory whose val split the fit uses. Use the real set: "
            "calibrating on renders would tune the scores for the wrong domain.",
        ),
    ],
    split: Annotated[str, typer.Option("--split")] = "val",
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Fit Platt scaling so the detector's scores mean something.

    A short DETR-family fine-tune ranks boxes well while scoring everything
    under 0.1 -- this checkpoint's mAP was 0.85 with no score above 0.07. The fit
    is monotone, so mAP is untouched; it only remaps the numbers and picks an
    operating threshold. Saved into the checkpoint and applied by `train predict`
    automatically.
    """
    from chesssight.train.calibrate import calibrate
    from chesssight.train.dataset import ChessDetectionDataset
    from chesssight.train.run import load_trained

    model, processor, resolved = load_trained(Path(checkpoint), device)
    dataset = ChessDetectionDataset(
        Path(data), processor, split=split, split_source="auto"
    )
    typer.echo(f"fitting on {len(dataset)} samples from split {split!r}")

    result = calibrate(
        model, processor, dataset, resolved, fit_split=split, limit=limit
    )
    path = result.save(Path(checkpoint))

    typer.echo(f"wrote {path}")
    typer.echo(f"  scale {result.scale:.3f}  bias {result.bias:.3f}")
    typer.echo(
        f"  operating threshold {result.threshold:.3f} -> "
        f"precision {result.precision:.3f}  recall {result.recall:.3f}  "
        f"F1 {result.f1:.3f}  ({result.detections_used} detections used)"
    )


@train_app.command("video")
def train_video(
    checkpoint: Annotated[Path, typer.Argument(help="Saved checkpoint directory.")],
    source: Annotated[Path, typer.Option("--input", "-i", help="Video to annotate.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Annotated video to write.")],
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help="Score floor; defaults to the checkpoint's calibrated one.",
        ),
    ] = None,
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k", help="Draw the N best boxes per frame instead of thresholding."
        ),
    ] = None,
    stride: Annotated[
        int,
        typer.Option(
            "--stride",
            help="Detect every Nth frame, redrawing between. 2-3 is fine hand-held.",
        ),
    ] = 1,
    max_seconds: Annotated[float | None, typer.Option("--max-seconds")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Run the detector over a video and write an annotated copy.

    Shows detection only: boxes, classes and calibrated confidences. A per-square
    position readout needs the board corners, which the detector does not emit.
    """
    from chesssight.train.video import annotate_video

    result = annotate_video(
        Path(checkpoint),
        Path(source),
        Path(out),
        threshold=threshold,
        top_k=top_k,
        stride=max(1, stride),
        max_seconds=max_seconds,
        device=device,
        progress=typer.echo,
    )
    typer.echo(
        f"wrote {result['output']}: {result['frames']} frames at "
        f"{result['fps']:.1f} fps, mean {result['mean_pieces']:.1f} pieces/frame"
    )
