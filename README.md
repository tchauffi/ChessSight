# ChessSight

ChessSight is a computer vision project designed to identify and classify chess pieces on a chessboard using image processing techniques. This repository contains the code and resources needed to detect chess pieces in real-time, providing a foundation for applications in automated chess analysis and gameplay.

## Synthetic dataset generator

Training a model to read a position off a photograph needs far more labelled boards
than anyone wants to annotate by hand. ChessSight renders them in Blender instead,
with labels that come out exact and free.

```bash
make doctor                   # check Blender, GPU and output directory
make synth-sheet N=16         # render 16 boards + an annotated contact sheet
uv run chesssight synth run -c configs/default.yaml -n 50000 -w 2
```

Each sample carries everything a board-reading model might train against:

| Label | Notes |
|---|---|
| FEN + 8×8 class grid | 13 classes: empty plus six piece types in two colours |
| Board corners + homography | Both directions, so the board can be rectified |
| 64 square centres and quads | Derived from the homography, so real photos get them too |
| Per-piece masks and boxes | Occlusion-correct (from the mask) *and* amodal (from geometry) |
| Captured pieces beside the board | Labelled, masked, and deliberately *not* on the grid |

### Segmentation masks

Masks are stored as RLE inside each sample record — compact, exact, and about ten
times smaller than a PNG per piece. To get images a training script can open:

```bash
uv run chesssight data masks <run> --previews   # semantic/ + instance/ PNGs
uv run chesssight data coco  <run>              # COCO instance segmentation
```

`semantic/` holds one class index per pixel (0 background, 1–12 pieces, 13 board) and
`instance/` holds per-piece ids. Both are *label* images, so they look almost black in
a viewer — that is correct, and `--previews` writes colourised copies to check by eye.
The QA overlay tints each instance too, which is the only way to see that a mask
covers the piece it claims to rather than merely having a plausible pixel count.

Everything decodes from the sample records alone, so `id_pass/` can be deleted after a
run is collected. Verified byte-exact: across 200 samples the decoded masks matched the
renderer's own ID pass on 52.4 M of 52.4 M pixels.

### Captured pieces

Real games accumulate taken pieces beside the board, and a model that has never seen
them will happily assign one to the nearest square. Scenes therefore show the captured
pile — drawn from the pieces genuinely *missing* from the position, so the pile always
agrees with the board rather than contradicting it.

They appear in `pieces[]` with `on_board: false` and a null `square`, carry masks and
boxes like anything else, and never enter the 8×8 grid. That distinction — a piece is
visible, but no square is occupied — is the whole point of including them.

### How it fits together

Blender ships its own Python, isolated from this project's virtualenv, so the code
splits in two and the halves talk only through JSON:

```
project side (uv venv, typed, tested)          blender side (stdlib + numpy)
  chesssight.synth.runner                        chesssight.blender.entry
    │ resolves every random choice into a          │ builds the scene, renders
    │ job spec, then shards it                     │ measures pixel coordinates
    ▼                                              ▼
  chesssight.synth.postprocess  ◄────────────  raw_labels/*.json + id_pass/*.png
    │ solves the homography, decodes masks, validates
    ▼
  samples/*.json
```

Two consequences worth knowing. All randomisation happens on the project side, so a
job spec is a complete, replayable record of a scene and every randomisation decision
is unit-testable without Blender. And the Blender side never computes a homography or
validates a schema, so a real photograph annotated with four clicked corners goes
through exactly the same code path as a render.

### Using your own chess set

The procedural pieces need no assets, but any external set can be dropped in. A
manifest says what the set is; the importer normalises it to the pipeline's
conventions (origin at the base centre, standing on `z = 0`, sized in board squares,
facing the opponent), so nothing in the renderer or the label pass changes.

```bash
uv run chesssight assets template ~/assets/my-set   # write a manifest skeleton
# put pawn.obj, knight.obj, ... next to it, then:
uv run chesssight assets check ~/assets/my-set/manifest.json
```

Then point the config at it:

```yaml
pieces:
  provider: my-set                                   # the manifest's own name
  asset_manifest: ~/assets/my-set/manifest.json
```

OBJ, glTF/GLB, FBX, Collada, STL, USD and `.blend` append are all supported. The two
fields worth getting right are `forward_axis` and `up_axis`: sets exported from Y-up
tools arrive lying on their side, and a knight facing the wrong way is the most
visible defect an imported set can have.

`chesssight assets export ~/assets/procedural` writes the built-in set as OBJ plus a
working manifest — a correctly-oriented reference to compare against when adapting a
download.

**Bake before rendering at scale.** Importing a print-ready set costs far more than
rendering it — a Staunton STL set runs to hundreds of megabytes and ~600k triangles
per piece — and the scene is reset between jobs, so that cost is otherwise paid *per
image*. Baking does the import, decimation, orientation and scaling once:

```bash
uv run chesssight assets bake ~/assets/my-set/manifest.json -o ~/assets/my-set-baked
```

Measured on the uppalong Staunton set: **21 s → 0.5 s per image**, and 248 MB → 24 MB
on disk. Point `asset_manifest` at the baked manifest for real runs.

**On licensing.** Chess models are easy to find and hard to license. Most GitHub
repositories carrying one either have no licence at all — which means all rights
reserved — or committed a third-party asset with no provenance.

This matters more than it might seem: **a render is a derivative work of the model in
it**, so the set's terms follow the images, the dataset, and arguably anything trained
on them. `assets check` flags restrictive terms, and the set's licence and credit line
are written into every dataset's `meta.json`, so the terms are discoverable from the
data rather than from whoever happened to run the generator.

A `CC BY-NC` set is fine for research, learning and personal projects, and not fine
for anything commercial. If commercial use is a possibility, prefer CC0 or CC-BY, or
stay on the procedural set — which has no external terms at all and is why it remains
the default.

Sources with clear per-model licences: [Poly Pizza](https://poly.pizza),
[Sketchfab](https://sketchfab.com) (filter to CC0), and
[Printables](https://www.printables.com/model/76438-staunton-chess-set) for printable
Staunton sets (STL imports fine). None of this is legal advice.

### Reproducibility

One master seed; every sample and every randomised aspect derives its own seed by
hashing. Sample *i* is reproducible without generating the ones before it, so a single
odd frame can be re-rendered and inspected on its own, and an interrupted run resumes
as a plain set difference:

```bash
uv run chesssight synth run -c configs/default.yaml -n 50000 --resume
```

### Checking the output

Labels that are wrong in a self-consistent way are the real hazard: a mirrored or
transposed board produces a plausible image and perfectly well-formed JSON. Three
checks run on every sample at generation time.

- The square centres are computed twice — once through Blender's camera model, once
  through the fitted homography — and must agree to well under a pixel.
- The projected board corners must wind the correct way. A mirrored board keeps every
  other label consistent, so only the winding order reveals it.
- The reprojection error over all 64 squares must stay sub-pixel.

`chesssight qa overlay` draws the labels back onto the image, which catches the
remaining class of mistake that numbers do not.

```bash
uv run chesssight qa overlay <run> --id 000000 --names
uv run chesssight qa stats <run>        # class balance, occupancy, visibility
uv run chesssight synth verify <run>    # re-check a dataset end to end
```

### Performance

Measured on an RTX 5070 Ti at 512×512:

| Engine | Per image | Use |
|---|---|---|
| EEVEE, 16 samples | ~0.2 s | Iterating on a config |
| Cycles + OPTIX, 96 samples | ~0.4 s | The final dataset |

Cycles workers share one GPU, so the runner caps them at two and says so.

## Prerequisites

Before installing, make sure you have:
- Python 3.12 or higher
- uv (Python package manager)
- Make
- Blender 5.2 or newer, on `PATH` (only needed to render; the library and its tests
  do not require it)

On macOS, you can install these using Homebrew:
```bash
brew install python@3.12 uv make
```

## Installation

### For Users

To install the package for use:

```bash
pip install .
```

### For Developers

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ChessSight.git
cd ChessSight
```

2. Install dependencies and set up pre-commit hooks:
```bash
make install
```



## Development Commands

The project uses a Makefile to simplify common development tasks:

- `make help` - Show available commands
- `make install` - Install dependencies and pre-commit hooks
- `make format` - Format code using black and ruff
- `make lint` - Run all linters (ruff, mypy)
- `make test` - Run tests
- `make pre-commit` - Run pre-commit hooks manually
- `make clean` - Remove cache and build files


## Code Quality

This project uses several tools to maintain code quality:
- Black for code formatting
- Ruff for linting
- MyPy for type checking
- Pre-commit hooks for automated checks

These are automatically run on commit, but you can run them manually:
```bash
make pre-commit
```
