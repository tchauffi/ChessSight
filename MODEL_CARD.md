# ChessSight detector v0.2.0

An RT-DETR object detector that finds a chessboard and the pieces standing on it in a
photograph or video frame. Trained **entirely on synthetic renders** — it has never
seen a real photograph during training, and every real image reported below is a
held-out test measurement.

- **Checkpoint**: `rtdetr_v4/best` (epoch 7 of 16, EMA weights)
- **Base model**: `PekingU/rtdetr_r50vd_coco_o365`
- **Classes**: 14 — six piece types × two colours, plus `board` and `corner`
- **Input**: 640×640
- **Size**: 164 MB (`model.safetensors`, fp32)
- **Repository commit**: see the `v0.2.0` tag

## What it does and does not do

It emits **boxes**: where the board is, where each piece is, and what type each piece
is. On its own it does not emit board corners, so a position readout needs a second
model — `corner_swin_v2` — and the pair together do produce a FEN. See
[Reading a position](#reading-a-position-two-operating-points) below and the README.

## Reading a position: two operating points

The threshold in `calibration.json` is fitted to maximise **detection F1**. Reading a
board is a different objective, and the same checkpoint scores very differently under
each (ChessReD **val**, where the operating point is chosen):

| operating point | per-square | boards exact |
|---|---|---|
| `calibration.threshold` = 0.229 (detection F1) | 96.56% | 43.3% |
| `POSITION_THRESHOLD` = 0.05 (board reading) | **98.40%** | **70.9%** |

Why the gap is so large: `grid_from` keeps only the best-scoring detection per square
and the homography discards anything off the board, so an extra low-scoring candidate
is usually harmless while a missing one always costs a square. Recall is worth far
more than precision here. The effect compounds — a sparsely populated grid also makes
the "which corner is a8" decision unreliable, and a rotated board is wrong everywhere
at once.

0.05 was chosen by sweeping ChessReD **val** and every headline figure below is
reported on **test**, so those are held out rather than tuned. It is the default for
position reading in both the torch and the ONNX backends; `--threshold` overrides it.
The value belongs to this checkpoint's score distribution: the previous detector's
compressed scores wanted 0.10, and a detector swap should re-run the sweep rather
than inherit the constant.

**On test, the pipeline reads 99.32% of squares and 71.24% of boards exactly** (the
previous release: 99.25% / 68.30%), and misnaming dominates what remains:

| ChessReD test | missed | spurious | misnamed | total |
|---|---|---|---|---|
| v0.1.0 @ 0.10 | 33 | 38 | 76 | 147 |
| **v0.2.0 @ 0.05** | **16** | **37** | **81** | **134** |

Missed squares halved; the residual error is naming — above all the queen/king pair,
silhouettes a few dozen pixels tall standing shoulder to shoulder — plus one sparse
endgame whose orientation flips (see the site's worst-board example).

## Results

### Real photographs — ChessReD test split, 306 images

The number that matters. Nothing from this split was used for training, checkpoint
selection or calibration.

| Metric | Value |
|---|---|
| mAP | **0.621** |
| mAP@50 | 0.879 |
| mAP@75 | 0.785 |
| mAP small | 0.427 |
| mAP medium | 0.616 |
| mAP large | 0.977 |

Raw mAP is statistically flat against v0.1.0 (0.636, and the run-to-run noise floor
is ~0.02) — but its composition moved where it matters for reading a board: the
queens, the previous release's single worst pair, gained ~0.1 AP on val, and the
board class reached 0.985. Boards-exact is the number this release was built for.

### Synthetic — `train7` validation split, 2050 renders

mAP **0.783** (mAP@50 0.934, small 0.738). Not comparable to the previous card's
0.839: `train7` is deliberately harder — framing out to 2.6× leaves the board barely
a quarter of the frame, and opening-heavy positions crowd the back ranks.

### Score calibration

A fresh RT-DETR head used to start at p=0.5 for every class on all 300 queries, and a
short fine-tune never recovered — everything scored under ~0.1. v0.2.0 initialises
the classification heads at a 0.01 prior (`--head-prior`, the standard focal-loss
bias trick), which roughly halves the compression: the Platt fit on **ChessReD val**
moved from `scale 4.695, bias 20.239` (v0.1.0) to

```
scale 2.950   bias 10.596   operating threshold 0.229
precision 0.626   recall 0.938   F1 0.751
```

The transform is monotone, so it cannot change mAP — it only makes the numbers mean
something and supplies a usable threshold. It is fitted on ChessReD's capture
conditions; the further your footage is from those, the less the confidences mean.
Re-fit with `chesssight train calibrate` for a new domain.

## Training data

`train7`: 20 000 Blender/Cycles renders at 640×640, 48 samples per image
(`configs/train7.yaml`). Every change below was promoted by a matched-seed 4800-image
A/B against the previous recipe; each arm changed exactly one variable.

- Positions: 55% from real Lichess games (2013-01 and 2013-02 dumps), 30% uniform
  random placement, and **15% opening positions** (plies 4–24), because "queen behind
  its neighbours on a crowded back rank" was the dominant real-photo failure.
- Piece sets: baked Staunton OBJ (60%) and procedural lathe profiles (40%). The
  procedural **queen now carries a coronet** — previously she was a pure lathe whose
  profile was the king's minus the cross, and queen↔king was the worst confusion.
  Rook merlon counts vary 3–8, and each letter's height is jittered ±6% per scene so
  the king/queen height gap is not a memorisable constant. (Val effect: both queens
  +0.05–0.07 AP.)
- Board: the frame's tone is **pinned dark** (the pre-train6 convention). The
  matched-seed A/B measured sampled light borders costing the *piece* detector 0.05
  mAP while helping the corner model — so the two models now train on different
  border recipes.
- Camera: framing margin extended to 0.95–2.6 (was 0.95–1.9), producing genuinely
  distant boards. (Val effect: mAP-small +0.06.)
- Materials, lighting, clocks, distractors: as before — HDRI on every image, finite
  table, enforced piece-vs-square contrast floor.

Split 80/10/10 into train/val/test by hashing the sample id, so the division survives
regeneration at a different size.

## Training recipe

16 epochs, batch 12, AdamW, lr 1e-4 with backbone at 1e-5, cosine schedule with 5%
warmup, grad clip 0.1, AMP, classification loss weight 3.0, classification-head
prior-bias init 0.01, photometric and geometric augmentation, EMA (decay 0.9999,
2000-step warmup). 148 minutes on one RTX 5070 Ti.

Reproduce with:

```bash
uv run chesssight train detr ~/datasets/chesssight/train7 -o ~/runs/rtdetr_v4 \
  -e 16 -b 12 --lr 1e-4 --backbone-lr 1e-5 --image-size 640 -w 4 \
  --cls-weight 3 --augment --ema --corners
```

A negative result worth recording: training this same data at 896 px input looked
excellent in-domain (synth val 0.794) and **collapsed on real small objects**
(mAP-small 0.041 vs 0.323) — 640² renders upsampled to 896 teach a blur that real
photographs downscaled to 896 do not have. Higher-resolution training requires
rendering at that resolution natively.

## Known limitations

- **Naming, not finding.** With the board-reading operating point the pipeline sees
  almost every piece (98.46% of occupied squares named correctly) and its remaining
  errors are mostly wrong *names* — the queen/king pair above all, reduced by the
  coronet work but still the top confusion. Piece identity on a crowded back rank is
  the open problem, not detection.
- **Small pieces.** mAP 0.43 on small objects against 0.98 on large. Distant boards
  and low camera angles degrade badly; native high-resolution rendering is the
  untried lever (see the 896 px negative result above).
- **Out-of-domain footage.** On small, blurred, near-edge-on boards the piece scores
  saturate on people and background, and the board box can come back an order of
  magnitude too large. `--board-gate`, class-agnostic NMS and a 32-piece cap bound the
  damage to something readable; **none of them make it correct.**
- **Calibration is domain-specific.** Confidences are meaningful for ChessReD-like
  photographs and progressively less so as footage diverges.
- **Video needs `--smooth`.** Per-frame detection flickers: frame-to-frame churn was
  2.30 pieces without tracking against 0.61 with it, on the same clip.
- **Single seed.** Every number here comes from one training run. Differences smaller
  than about 0.02 mAP are not distinguishable from run-to-run noise.

## Intended use

Analysis and research on chess imagery. Not validated for officiating, rating, or any
setting where a misread board carries a cost.

## Licence

Code under the repository's LICENSE. Training data is generated from CC0 assets
(Poly Haven HDRIs and textures) and public Lichess game dumps.
