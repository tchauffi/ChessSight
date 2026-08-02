# ChessSight detector v0.1.0

An RT-DETR object detector that finds a chessboard and the pieces standing on it in a
photograph or video frame. Trained **entirely on synthetic renders** — it has never
seen a real photograph during training, and every real image reported below is a
held-out test measurement.

- **Checkpoint**: `rtdetr_synth5/best` (epoch 7 of 16, EMA weights)
- **Base model**: `PekingU/rtdetr_r50vd_coco_o365`
- **Classes**: 13 — six piece types × two colours, plus `board`
- **Input**: 640×640
- **Size**: 164 MB (`model.safetensors`, fp32)
- **Repository commit**: see the `v0.1.0` tag

## What it does and does not do

It emits **boxes**: where the board is, where each piece is, and what type each piece
is. It does **not** emit board corners, so there is no homography and therefore **no
per-square position readout and no FEN** from a real image. Reconstructing a position
needs corner regression, which this release does not have. The synthetic dataset
already carries corner labels, so this is a training target that exists, not a missing
capability of the data.

## Results

### Real photographs — ChessReD test split, 306 images

The number that matters. Nothing from this split was used for training, checkpoint
selection or calibration.

| Metric | Value |
|---|---|
| mAP | **0.636** |
| mAP@50 | 0.884 |
| mAP@75 | 0.817 |
| mAP small | 0.466 |
| mAP medium | 0.640 |
| mAP large | 0.922 |

The `last` checkpoint scores 0.634 — the two agree to 0.002, so the result does not
depend on which epoch is picked. That stability is new here; the previous model
spanned 0.605–0.628 across its own two checkpoints, and 0.385–0.485 on small objects.

### Synthetic — `train5` held-out test split, 1966 renders

| Checkpoint | mAP | mAP@50 | mAP small |
|---|---|---|---|
| `best` | 0.839 | 0.943 | 0.758 |
| `last` | 0.788 | 0.881 | 0.691 |

Test tracks validation to within 0.005, so validation was not overfitted by
checkpoint selection.

**The 0.839 synthetic vs 0.636 real gap is the sim-to-real gap, and most of it lives
in small objects** (0.758 vs 0.466). Distant or heavily foreshortened pieces are where
this model is weakest, and no intervention tried so far — 640px training, EMA,
classification loss weighting, procedural and photographed materials — has moved it.

### Score calibration

Raw RT-DETR scores after a short fine-tune are severely compressed; ranking is good
while nothing scores above ~0.07. `calibration.json` ships a Platt scaling fitted on
**ChessReD val** (real photographs, not renders):

```
scale 4.695   bias 20.239   operating threshold 0.395
precision 0.829   recall 0.947   F1 0.884
```

The transform is monotone, so it cannot change mAP — it only makes the numbers mean
something and supplies a usable threshold. It is fitted on ChessReD's capture
conditions; the further your footage is from those, the less the confidences mean.
Re-fit with `chesssight train calibrate` for a new domain.

## Training data

`train5`: 20 000 Blender/Cycles renders at 640×640, 48 samples per image.

- Positions: 70% from real Lichess games (2013-01 and 2013-02 dumps), 30% uniform
  random placement. Random boards are deliberate — a purely game-derived set lets the
  model lean on chess priors instead of reading pixels.
- Piece sets: baked Staunton OBJ (60%) and procedural lathe profiles (40%) with
  silhouette taper, so the model cannot key on one set's outline.
- Materials: plastic, procedural wood and marble with domain-warped noise, and
  photographed Poly Haven PBR veneers, all with randomised hue, saturation and
  brightness. A minimum piece-vs-square luminance contrast is enforced, so no piece is
  invisible against the square it stands on.
- Scene: HDRI lighting and backdrop on every image, finite table, chess clocks in 45%
  of scenes, and up to three distractor objects from six kinds.
- Camera: azimuth 0–360°, elevation 8–75° (including grazing near-edge-on views),
  24–85 mm, depth of field on 35%.

Split 80/10/10 into train/val/test by hashing the sample id, so the division survives
regeneration at a different size.

## Training recipe

16 epochs, batch 12, AdamW, lr 1e-4 with backbone at 1e-5, cosine schedule with 5%
warmup, grad clip 0.1, AMP, classification loss weight 3.0, photometric and geometric
augmentation, EMA (decay 0.9999, 2000-step warmup). 147 minutes on one RTX 5070 Ti.

Reproduce with:

```bash
uv run chesssight train detr ~/datasets/chesssight/train5 -o ~/runs/rtdetr_synth5 \
  -e 16 -b 12 --lr 1e-4 --backbone-lr 1e-5 --image-size 640 -w 4 \
  --cls-weight 3 --augment --ema
```

## Known limitations

- **No position readout.** Boxes only; see above.
- **Small pieces.** mAP 0.47 on small objects against 0.92 on large. Distant boards
  and low camera angles degrade badly.
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
