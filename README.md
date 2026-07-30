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

**Use more than one.** Scaling a set up and down does not change its outline, so a
dataset rendered from a single set lets the detector learn *that set's silhouette*
instead of learning what a bishop is — and that is precisely the cue that fails on a
photograph of somebody else's board. `pieces.sets` is a weighted list, drawn per image:

```yaml
pieces:
  sets:
    - {provider: uppachess-staunton, asset_manifest: assets/baked/manifest.json, weight: 0.6}
    - {provider: procedural, weight: 0.4}
  taper: {min: -0.12, max: 0.12}
```

`taper` varies the shape *within* the procedural set: positive widens the base and
narrows the top, negative the reverse, so it spans squat through slender rather than
one fixed profile. It is bounded jointly with `radius_scale` — enlarging and tapering
multiply, and a flared base that overflowed its square would leave pieces touching,
which is a quietly wrong dataset rather than a crash. The config rejects a combination
that would, at load time.

**Bake before rendering at scale.** Importing a print-ready set costs far more than
rendering it — a Staunton STL set runs to hundreds of megabytes and ~600k triangles
per piece — and the scene is reset between jobs, so that cost is otherwise paid *per
image*. Baking does the import, decimation, orientation and scaling once:

```bash
uv run chesssight assets bake ~/assets/my-set/manifest.json -o ~/assets/my-set-baked
```

Measured on the uppalong Staunton set: **21 s → 0.5 s per image**, and 248 MB → 24 MB
on disk. Point `asset_manifest` at the baked manifest for real runs.

### Positions

Boards come from real games. The random sampler that produced earlier datasets places
legal-*looking* pieces but not a legal-looking game: measured over 5 000 draws it puts
15% of pieces on pawns where real games put 26%, and 12% on queens where real games put
6%. It also has no structure — no pawn chains, no castled kings, nothing on a home
square — which is most of what a board actually looks like.

```bash
curl -O https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
```

```yaml
positions:
  pgn_paths: [~/assets/chesssight/pgn/lichess_db_standard_rated_2013-01.pgn.zst]
  weight_pgn: 0.7
  weight_random: 0.3
```

Any month works; the 2013 files are ~17 MB and hold ~120 k games, which is far more
than a dataset needs. `.pgn` and `.pgn.zst` are both read directly.

The random share stays deliberately non-zero. Real games cluster near the starting
position, and a purely game-derived set would let a model recover the position from
chess priors rather than from pixels — which is the exact failure this generator exists
to avoid.

### Environment lighting (HDRI)

The procedural sun-plus-fills rig makes clean, directional, physically simple light. A
real room does not: it throws colour off painted walls, soft gradients through windows,
several mismatched fixtures, and reflections that land differently on every curved
piece. That difference is exactly the kind of cue a classifier latches onto, so the
generator lights scenes with real captured environments instead — which also puts a
real, varied room *behind* the table, instead of the flat-colour void that most loudly
says "render".

```bash
uv run chesssight assets hdri ~/assets/chesssight/hdri     # ~22 curated maps at 2k
```

The checked-in configs already point at that directory with `hdri_probability: 1.0`;
if the directory is missing, the generator silently falls back to the procedural rig,
so the pipeline still works with no external assets. Dial the probability down to mix
the procedural world back in:

```yaml
lighting:
  hdri_dir: ~/assets/chesssight/hdri
  hdri_probability: 1.0      # below 1.0, the rest uses the procedural rig
```

The maps come from [Poly Haven](https://polyhaven.com/hdris) and are **CC0** — the one
asset class here with no licence to propagate. They are curated by name rather than
pulled by category: a category filter returns abandoned factories and Christmas photo
studios, and lighting a chessboard by a derelict boiler room is domain *noise*, not
domain randomisation. Everything included is a room a game could plausibly be played
in, in a deliberate mix of daylight and artificial light.

Two behaviours are worth knowing, both learned the hard way. An HDRI **replaces** the
sun and fills rather than adding to them — lighting a scene with all three at once
washed it out (mean luminance 148-202 of 255, almost no contrast) and cast a second set
of shadows contradicting the first. And it gets its own, much narrower strength range,
because an environment map already encodes absolute radiance, so the wide multiplier
that a flat-colour world needs simply blows the image out.

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

### Training a detector

```bash
uv sync --extra train                       # torch is ~3 GB, so it is opt-in
uv run chesssight train detr <run> -o runs/rtdetr --epochs 20
uv run chesssight train evaluate runs/rtdetr/best --data <run>
uv run chesssight train calibrate runs/rtdetr/best --data <real-run>   # then predict uses real thresholds
```

Fine-tunes **RT-DETR** to find the board and all twelve piece types in one pass.
RT-DETR rather than plain DETR: the two are the same family, but DETR's slow
convergence comes from one-to-one Hungarian matching over dense attention, and it
bites hardest on small objects — which here is every piece on a board seen from
across the table. Pass `--model facebook/detr-resnet-50` to compare against the
original.

The board is the 13th class, boxed from the four corner labels and clipped to the
image. Piece targets are the *modal* boxes measured from the masks; training on the
amodal boxes would teach the detector to predict pieces it cannot see.

A short DETR-family fine-tune ranks boxes well while scoring everything under
0.1 — the classification loss is dwarfed by the box terms, and varifocal logits
grow over many more epochs than a fine-tune runs. `train calibrate` fits Platt
scaling on a real val split: monotone, so mAP is untouched, but scores become
honest probabilities (measured ECE 0.03) and a normal threshold works. The fit
is saved into the checkpoint and applied by `train predict` automatically. For
training-time mitigation, `--cls-weight 3` raises the classification loss weight.

The train/val split hashes the sample id rather than slicing the index, so it
survives the dataset being regenerated at a different size and cannot put a
correlated block of seeds on one side. mAP is reported per class as well as
overall — a single averaged figure hides the thing that matters, since the board is
one enormous easy box and the pieces are dozens of small ones.

On an RTX 50-series card, note that torch and torchvision come from the CUDA 12.8
index (configured in `pyproject.toml`): Blackwell is `sm_120` and the default PyPI
wheels carry no kernels for it.

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
