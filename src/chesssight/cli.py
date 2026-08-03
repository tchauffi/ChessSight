"""``chesssight`` command line interface."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer

from chesssight.data.dataset import DatasetReader
from chesssight.data.geometry import is_mirrored
from chesssight.data.qa import contact_sheet, render_overlay
from chesssight.synth import runner
from chesssight.synth.config import GeneratorConfig
from chesssight.train.position import POSITION_THRESHOLD

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


@app.command()
def predict(
    images: Annotated[list[Path], typer.Argument(help="Photographs to read.")],
    detector: Annotated[
        Path, typer.Option("--detector", help="RT-DETR checkpoint directory.")
    ],
    corners: Annotated[
        Path, typer.Option("--corners", help="Corner heatmap checkpoint directory.")
    ],
    diagram: Annotated[
        Path | None,
        typer.Option(
            "--diagram",
            help="Write an SVG board diagram per image (a directory when several).",
        ),
    ] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Detection score floor. The default reads boards; the "
            "checkpoint's own calibration point is stricter and reads them "
            "worse (see the README).",
        ),
    ] = POSITION_THRESHOLD,
) -> None:
    """Read the position off each photograph: one FEN per line.

    This is the pipeline the rest of the repository exists to train: detector
    for the pieces, corner heatmap for the geometry, colour parity plus piece
    placement for which corner is a8. "No board found" prints as exactly that
    rather than a guess.
    """
    from PIL import Image

    from chesssight.train.predict_position import load_reader

    reader = load_reader(detector, corners, device, threshold)
    many = len(images) > 1
    if diagram and many:
        diagram.mkdir(parents=True, exist_ok=True)

    for path in images:
        result = reader.read(Image.open(path).convert("RGB"))
        if result["fen"] is None:
            typer.echo(f"{path}: no board found")
            continue
        typer.echo(f"{path}: {result['fen']}")
        if diagram:
            from chesssight.demo.render import board_svg

            target = diagram / f"{path.stem}.svg" if many else diagram
            target.write_text(board_svg(result["fen"]), encoding="utf-8")


onnx_app = typer.Typer(
    help="Export and run the pipeline without torch.", no_args_is_help=True
)
app.add_typer(onnx_app, name="onnx")


@onnx_app.command("export")
def onnx_export(
    detector: Annotated[
        Path, typer.Option("--detector", help="RT-DETR checkpoint directory.")
    ],
    corners: Annotated[
        Path, typer.Option("--corners", help="Corner heatmap checkpoint directory.")
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Bundle directory.")],
    fp16: Annotated[
        bool,
        typer.Option(
            "--fp16/--no-fp16",
            help="Store weights as float16 (io stays float32): half the bundle, "
            "measured identical to torch on ChessReD test, and what docs/models "
            "ships. uint8 weight quantisation measured 8 points worse — don't.",
        ),
    ] = False,
) -> None:
    """Write both models to ONNX, with the metadata needed to run them."""
    from chesssight.inference.onnx import export

    export(detector, corners, out, fp16=fp16)
    for file in sorted(out.iterdir()):
        typer.echo(f"  {file.name:16} {file.stat().st_size / 1e6:8.1f} MB")


@onnx_app.command("parity")
def onnx_parity(
    bundle: Annotated[Path, typer.Argument(help="Exported ONNX bundle.")],
    detector: Annotated[
        Path, typer.Option("--detector", help="RT-DETR checkpoint directory.")
    ],
    corners: Annotated[
        Path, typer.Option("--corners", help="Corner heatmap checkpoint directory.")
    ],
    data: Annotated[Path, typer.Option("--data", help="Dataset root.")],
    split: Annotated[str, typer.Option("--split")] = "test",
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Stop after this many.")
    ] = None,
    tolerance: Annotated[
        float, typer.Option("--tolerance", help="Allowed max abs tensor difference.")
    ] = 5e-3,
    min_agreement: Annotated[
        float, typer.Option("--min-agreement", help="Required fraction of equal FENs.")
    ] = 0.9,
) -> None:
    """Check the ONNX backend against the torch one, at two levels.

    The ONNX path reimplements the pre- and post-processing that the torch path
    gets from torchvision and transformers. A second copy of a rule is exactly
    what has drifted silently in this repository before, so it is checked.

    **Tensors** are the real guard: a wrong normalisation, label order or box
    format moves the model outputs by far more than `--tolerance`, and that
    fails loudly.

    **FENs** are reported but only loosely bounded. They cannot be expected to
    match exactly: the operating threshold sits in a dense part of the score
    distribution, and a detection within a thousandth of it flips on
    floating-point noise alone. A low agreement rate means something real is
    wrong; a handful of one-square differences does not.
    """
    import numpy as np
    import torch
    from PIL import Image

    from chesssight.data.dataset import DatasetReader
    from chesssight.inference.onnx import (
        corner_input,
        detector_input,
    )
    from chesssight.inference.onnx import (
        load_reader as load_onnx,
    )
    from chesssight.train.predict_position import load_reader as load_torch

    reader = DatasetReader(data)
    entries = reader.entries(split)[: limit or None]
    torch_pipeline = load_torch(detector, corners, "cpu")
    onnx_pipeline = load_onnx(bundle)
    meta = onnx_pipeline.meta

    worst = {"logits": 0.0, "boxes": 0.0, "heatmap": 0.0}
    agree = 0
    differing: list[str] = []

    for index, entry in enumerate(entries, 1):
        sample = reader.load(entry.id)
        image = Image.open(reader.image_path(sample)).convert("RGB")

        # Tensor level: the same pixels through both graphs.
        with torch.no_grad():
            reference = torch_pipeline.detector(
                pixel_values=torch.from_numpy(
                    detector_input(image, meta["detector_size"], meta["rescale"])
                )
            )
            reference_heat = torch_pipeline.corner_model(
                torch.from_numpy(corner_input(image, meta["corner_size"]))
            )
        got_logits, got_boxes = onnx_pipeline.detector.run(
            ["logits", "pred_boxes"],
            {
                "pixel_values": detector_input(
                    image, meta["detector_size"], meta["rescale"]
                )
            },
        )
        got_heat = onnx_pipeline.corners.run(
            ["heatmap"], {"image": corner_input(image, meta["corner_size"])}
        )[0]

        # The detector selects 300 queries out of 8400 proposals, so two
        # near-tied proposals can come back in a different order for a
        # difference of 1e-5 in their scores. Comparing element by element
        # then reports a huge difference for two rows that were merely
        # swapped. Sorting first asks the question actually meant: are these
        # the same numbers? The heatmap is dense with no selection in it, so
        # it is compared where it lies.
        for name, ref, got in (
            ("logits", reference.logits, got_logits),
            ("boxes", reference.pred_boxes, got_boxes),
        ):
            a = np.sort(ref.numpy().reshape(-1))
            b = np.sort(np.asarray(got).reshape(-1))
            worst[name] = max(worst[name], float(np.abs(a - b).max()))
        worst["heatmap"] = max(
            worst["heatmap"],
            float(np.abs(reference_heat.numpy() - got_heat).max()),
        )

        expected = torch_pipeline.read(image)["fen"]
        actual = onnx_pipeline.read(image)["fen"]
        if expected == actual:
            agree += 1
        else:
            differing.append(entry.id)
        if index % 25 == 0:
            typer.echo(f"  {index}/{len(entries)}")

    total = len(entries)
    typer.echo("max abs difference vs torch:")
    for name, value in worst.items():
        flag = "ok" if value <= tolerance else "FAIL"
        typer.echo(f"  {name:8} {value:.3e}  {flag}")
    typer.echo(f"identical FENs: {agree}/{total} ({agree / total:.1%})")
    if differing:
        typer.echo(f"  differing: {', '.join(differing[:10])}")

    failed = any(v > tolerance for v in worst.values()) or agree / total < min_agreement
    if failed:
        raise typer.Exit(1)


@app.command()
def demo(
    detector: Annotated[
        Path | None, typer.Option("--detector", help="RT-DETR checkpoint directory.")
    ] = None,
    corners: Annotated[
        Path | None,
        typer.Option("--corners", help="Corner heatmap checkpoint directory."),
    ] = None,
    host: Annotated[str, typer.Option("--host", help="Address to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind.")] = 7860,
    device: Annotated[str | None, typer.Option("--device")] = None,
    onnx: Annotated[
        Path | None,
        typer.Option("--onnx", help="Serve an exported ONNX bundle instead."),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Detection score floor (see `predict`)."),
    ] = POSITION_THRESHOLD,
) -> None:
    """Serve a page that reads the position off a photograph you drop on it.

    The same pipeline as `predict`, with a browser in front of it. Binds to
    localhost by default: the models and the photographs stay on this machine,
    and making it reachable from elsewhere should be a deliberate act.

    With `--onnx` the checkpoints are not needed and neither is torch.
    """
    from chesssight.demo.server import serve

    if onnx is not None:
        from chesssight.inference.onnx import load_reader as load_onnx

        reader: Any = load_onnx(onnx, threshold=threshold)
        typer.echo(f"  onnx bundle {onnx}")
    else:
        if detector is None or corners is None:
            typer.echo("pass --detector and --corners, or --onnx <bundle>.")
            raise typer.Exit(1)
        from chesssight.train.predict_position import load_reader

        reader = load_reader(detector, corners, device, threshold)
        typer.echo(f"  models loaded on {reader.device}")
    serve(reader, host=host, port=port)


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
        # Name the commonest reason. Every failure is already on disk in
        # raw_labels/*.error.txt, but a run that loses most of its samples to one
        # repeated cause should say so here rather than leave it to whoever
        # thinks to read a log -- the whole point is that a short dataset is easy
        # to mistake for a finished one.
        from collections import Counter

        reasons = Counter(reason for _, reason in result.failures)
        typer.echo(
            f"[chesssight] {result.failed} sample(s) failed; "
            f"most common: {reasons.most_common(1)[0][0][:160]}",
            err=True,
        )
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
    test_fraction: Annotated[
        float,
        typer.Option(
            "--test-fraction",
            help="Held out of training and never scored during the run. The "
            "validation split selects the checkpoint and fits the calibration, so "
            "it is not an unbiased estimate by the end.",
        ),
    ] = 0.1,
    corners: Annotated[
        bool,
        typer.Option(
            "--corners/--no-corners",
            help="Also learn the four board corners, carried as a class of small "
            "boxes. Their centres give the homography, which is what turns piece "
            "boxes into a position rather than a pile of detections.",
        ),
    ] = False,
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
    head_prior: Annotated[
        float | None,
        typer.Option(
            "--head-prior",
            help="Initialise a fresh classification head to predict this prior "
            "probability per class (the focal-loss bias trick) instead of the "
            "reinit's p=0.5, which is what compresses a short fine-tune's "
            "scores. 0 disables; ignored when warm-starting a matching head.",
        ),
    ] = 0.01,
    focal_alpha: Annotated[
        float | None,
        typer.Option(
            "--focal-alpha",
            help="Override RT-DETR's varifocal alpha. Stock value when omitted.",
        ),
    ] = None,
    focal_gamma: Annotated[
        float | None,
        typer.Option(
            "--focal-gamma",
            help="Override RT-DETR's varifocal gamma. Stock value when omitted.",
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
    ema: Annotated[
        bool,
        typer.Option(
            "--ema/--no-ema",
            help="Evaluate and save an exponential moving average of the weights "
            "(decay 0.9999), as the reference RT-DETR recipe does. On by default.",
        ),
    ] = True,
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
        head_prior=None if head_prior in (None, 0) else head_prior,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        augment=augment,
        corners=corners,
        ema=ema,
        model_name=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        backbone_learning_rate=backbone_lr,
        image_size=image_size,
        num_workers=workers,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        limit=limit,
    )
    run_training(config, device=device)


@train_app.command("corners")
def train_corners(
    data: Annotated[
        list[Path],
        typer.Argument(help="One or more run directories to train on."),
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Where to save checkpoints.")
    ],
    backbone: Annotated[
        str,
        typer.Option(
            "--backbone",
            help="Any torchvision ResNet. The task is geometric, not semantic, "
            "so a bigger backbone mostly buys cost.",
        ),
    ] = "resnet18",
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 20,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b")] = 16,
    lr: Annotated[float, typer.Option("--lr")] = 3e-4,
    image_size: Annotated[int, typer.Option("--image-size")] = 448,
    sigma: Annotated[
        float,
        typer.Option(
            "--sigma",
            help="Target peak width in heatmap cells. Too wide and two corners "
            "of a steeply-angled board merge into one blob.",
        ),
    ] = 2.0,
    workers: Annotated[int, typer.Option("--workers", "-w")] = 8,
    val_fraction: Annotated[float, typer.Option("--val-fraction")] = 0.1,
    test_fraction: Annotated[float, typer.Option("--test-fraction")] = 0.1,
    augment: Annotated[bool, typer.Option("--augment/--no-augment")] = True,
    val_data: Annotated[
        Path | None,
        typer.Option(
            "--val-data",
            help="A dataset to select the checkpoint on that is never trained "
            "on. Point it at the real set: synthetic corner error plateaus "
            "while real error is still moving, so selecting on renders past "
            "that point picks on noise in the wrong domain.",
        ),
    ] = None,
    eval_dataset: Annotated[int, typer.Option("--eval-dataset")] = 0,
    eval_split: Annotated[str, typer.Option("--eval-split")] = "val",
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Train the corner heatmap model.

    Predicts a single corner-ness map with four peaks rather than four named
    channels: which corner is a8 cannot be read off a board's appearance, so
    naming them is left to the geometry downstream.
    """
    from chesssight.train.corner_run import train as run_training
    from chesssight.train.heatmap import HeatmapConfig

    config = HeatmapConfig(
        data_roots=[Path(root) for root in data],
        output_dir=Path(out),
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        image_size=image_size,
        sigma=sigma,
        num_workers=workers,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        augment=augment,
        val_root=Path(val_data) if val_data else None,
        eval_dataset=eval_dataset,
        eval_split=eval_split,
        limit=limit,
    )
    run_training(config, device=device)


@train_app.command("corners-evaluate")
def train_corners_evaluate(
    checkpoint: Annotated[Path, typer.Argument(help="Saved corner checkpoint.")],
    data: Annotated[Path, typer.Option("--data", help="Run directory to score on.")],
    split: Annotated[str, typer.Option("--split")] = "test",
    min_score: Annotated[
        float,
        typer.Option(
            "--min-score",
            help="Peak confidence below which a corner does not count. At 0 the "
            "decoder always returns its four best cells, so the found rate is "
            "100% whether or not there is a board in the picture.",
        ),
    ] = 0.0,
    limit: Annotated[int | None, typer.Option("--limit", "-n")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Corner error on a dataset, in pixels and in squares.

    Squares is the number worth quoting: pixel error alone says nothing without
    knowing how large the board appeared.
    """
    from chesssight.data.dataset import DatasetReader
    from chesssight.train.corner_run import evaluate_samples, format_report
    from chesssight.train.heatmap import load

    model, config = load(Path(checkpoint))
    from chesssight.train.engine import resolve_device

    resolved = resolve_device(device)
    model = model.to(resolved)
    metrics = evaluate_samples(
        model,
        DatasetReader(Path(data)),
        resolved,
        split=split,
        input_size=config.image_size,
        stride=config.stride,
        limit=limit,
        min_score=min_score,
        progress=typer.echo,
    )
    typer.echo(format_report(metrics))


@train_app.command("evaluate")
def train_evaluate(
    checkpoint: Annotated[Path, typer.Argument(help="Saved checkpoint directory.")],
    data: Annotated[Path, typer.Option("--data", help="Run directory to evaluate on.")],
    split: Annotated[str, typer.Option("--split")] = "val",
    batch_size: Annotated[int, typer.Option("--batch-size", "-b")] = 8,
    val_fraction: Annotated[float, typer.Option("--val-fraction")] = 0.1,
    test_fraction: Annotated[float, typer.Option("--test-fraction")] = 0.1,
    device: Annotated[str | None, typer.Option("--device")] = None,
) -> None:
    """Report mAP for a saved checkpoint, overall and per class.

    The fractions must match the ones the run was trained with, or the split this
    scores is not the one that was held out.
    """
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
        split_spec=SplitSpec(val_fraction=val_fraction, test_fraction=test_fraction),
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
    from chesssight.train.dataset import SplitSpec, select_entries
    from chesssight.train.run import load_trained
    from chesssight.train.visualize import (
        draw_predictions,
        on_board_only,
        predict,
        square_accuracy,
        take_top,
    )

    reader = DatasetReader(data)
    # Reproduce the training-time hold-out exactly, so what is drawn is data the
    # model has genuinely not seen. Shared with the loaders and the evaluator
    # rather than reimplemented here: the last copy of this rule drifted and drew
    # validation images whenever `test` was asked for.
    entries, source = select_entries(
        reader.entries(), split=split, spec=SplitSpec(), split_source=split_source
    )

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


@assets_app.command("textures")
def assets_textures(
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the texture maps.")
    ] = Path("~/assets/chesssight/textures"),
    resolution: Annotated[
        str, typer.Option("--resolution", help="Poly Haven resolution, e.g. 1k/2k/4k.")
    ] = "2k",
    only: Annotated[
        str | None,
        typer.Option(
            "--only", help="Comma-separated group names: wood, parquet, cloth, stone."
        ),
    ] = None,
) -> None:
    """Download CC0 PBR surface textures for the table the board stands on.

    The tabletop is the largest area in frame after the board, and it was a flat
    colour: procedural noise makes a surface non-uniform, but it cannot produce
    figure that runs, joins between boards, or varnish pooling unevenly over wear.
    Point `scene.texture_dir` at the result.
    """
    from chesssight.synth.textures import CURATED, download

    slugs = None
    if only:
        groups = [g.strip() for g in only.split(",")]
        unknown = [g for g in groups if g not in CURATED]
        if unknown:
            typer.echo(f"unknown groups {unknown}; have {sorted(CURATED)}", err=True)
            raise typer.Exit(1)
        slugs = [slug for g in groups for slug in CURATED[g]]

    result = download(
        Path(out), slugs=slugs, resolution=resolution, progress=typer.echo
    )
    typer.echo(
        f"{result['total']} textures in {result['dir']} "
        f"({result['downloaded']} fetched, {result['skipped']} already present)"
    )
    typer.echo("\nUse them with:")
    typer.echo("  scene:")
    typer.echo(f"    texture_dir: {out}")
    typer.echo("    texture_probability: 0.85")


@assets_app.command("hdri")
def assets_hdri(
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Where to write the .hdr files.")
    ] = Path("~/assets/chesssight/hdri"),
    resolution: Annotated[
        str, typer.Option("--resolution", help="Poly Haven resolution, e.g. 1k/2k/4k.")
    ] = "2k",
    only: Annotated[
        str | None,
        typer.Option(
            "--only", help="Comma-separated group names: halls, rooms, offices, social."
        ),
    ] = None,
) -> None:
    """Download CC0 indoor HDRI environment maps for image-based lighting.

    Real rooms throw coloured bounce, soft window gradients and mismatched
    fixtures that the procedural sun-plus-fills rig cannot produce -- and piece
    appearance under light is precisely what failed to transfer to real
    photographs. Point `lighting.hdri_dir` at the result.
    """
    from chesssight.synth.hdri import CURATED, download

    slugs = None
    if only:
        groups = [g.strip() for g in only.split(",")]
        unknown = [g for g in groups if g not in CURATED]
        if unknown:
            typer.echo(f"unknown groups {unknown}; have {sorted(CURATED)}", err=True)
            raise typer.Exit(1)
        slugs = [slug for g in groups for slug in CURATED[g]]

    result = download(
        Path(out), slugs=slugs, resolution=resolution, progress=typer.echo
    )
    typer.echo(
        f"{result['total']} maps in {result['dir']} "
        f"({result['downloaded']} fetched, {result['skipped']} already present)"
    )
    typer.echo("\nUse them with:")
    typer.echo("  lighting:")
    typer.echo(f"    hdri_dir: {out}")
    typer.echo("    hdri_probability: 0.7")
