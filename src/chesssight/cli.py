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
    data: Annotated[Path, typer.Argument(help="Run directory to train on.")],
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

    config = TrainConfig(
        data_root=Path(data),
        output_dir=Path(out),
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
