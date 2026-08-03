// The board-reading pipeline, in the browser.
//
// This is a third implementation of rules that already exist in Python, which
// in this repository is the shape of bug that has shipped before: a copy that
// drifts and produces a merely wrong number. Two things guard it. Every
// constant comes from `constants.js`, generated from the Python values at
// build time rather than typed here. And `scripts/check_browser_pipeline.mjs`
// replays fixtures dumped from the Python pipeline through these functions and
// fails on any disagreement.
//
// Only the plumbing lives here. The models do the seeing.

import { C } from "./constants.js";

// ---------------------------------------------------------------- preprocess

/** Draw an image into a square canvas and return planar CHW float data. */
function toPlanar(image, size, transform) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, size, size);
  const { data } = context.getImageData(0, 0, size, size);

  const out = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0; i < plane; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[c * plane + i] = transform(data[i * 4 + c] / 255, c);
    }
  }
  return out;
}

/** RT-DETR: rescale only. Its saved config has do_normalize false. */
export function detectorInput(image, size) {
  return toPlanar(image, size, (v) => v);
}

/** The corner model: ImageNet statistics. */
export function cornerInput(image, size) {
  return toPlanar(image, size, (v, c) => (v - C.cornerMean[c]) / C.cornerStd[c]);
}

// -------------------------------------------------------------- corner peaks

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/**
 * Peaks of the corner heatmap as [x, y, score] in input-image pixels.
 * Sigmoid, a 3x3 maximum filter so one blob yields one point, top-k, then an
 * intensity-weighted mean over a small window to place the point between cells.
 */
export function decodePeaks(heatmap, height, width, stride, count = 4, radius = 2) {
  const scores = new Float64Array(height * width);
  for (let i = 0; i < scores.length; i += 1) scores[i] = sigmoid(heatmap[i]);

  const peaks = new Float64Array(height * width);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let max = -Infinity;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
          max = Math.max(max, scores[ny * width + nx]);
        }
      }
      const here = scores[y * width + x];
      peaks[y * width + x] = here >= max ? here : 0;
    }
  }

  const order = Array.from(peaks.keys()).sort((a, b) => peaks[b] - peaks[a]);
  const found = [];
  for (const index of order.slice(0, count)) {
    const cy = Math.floor(index / width);
    const cx = index % width;
    const left = Math.max(0, cx - radius);
    const right = Math.min(width, cx + radius + 1);
    const top = Math.max(0, cy - radius);
    const bottom = Math.min(height, cy + radius + 1);

    let weight = 0;
    let sx = 0;
    let sy = 0;
    for (let y = top; y < bottom; y += 1) {
      for (let x = left; x < right; x += 1) {
        const value = scores[y * width + x];
        weight += value;
        sx += value * x;
        sy += value * y;
      }
    }
    const x = weight > 0 ? sx / weight : cx;
    const y = weight > 0 ? sy / weight : cy;
    found.push([(x + 0.5) * stride, (y + 0.5) * stride, peaks[index]]);
  }
  return found;
}

/**
 * Four points ordered clockwise from the top-left, by angle about the centroid.
 * Sorting coordinates instead would silently return a bow-tie for a board seen
 * from a low, rotated viewpoint.
 */
export function orderClockwise(points) {
  const cx = points.reduce((a, p) => a + p[0], 0) / points.length;
  const cy = points.reduce((a, p) => a + p[1], 0) / points.length;
  const sorted = [...points].sort(
    (a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx),
  );

  const minX = Math.min(...points.map((p) => p[0]));
  const minY = Math.min(...points.map((p) => p[1]));
  let start = 0;
  let best = Infinity;
  sorted.forEach((p, i) => {
    const d = Math.hypot(p[0] - minX, p[1] - minY);
    if (d < best) {
      best = d;
      start = i;
    }
  });
  return [...sorted.slice(start), ...sorted.slice(0, start)];
}

// --------------------------------------------------------------- homography

/** Solve the 8x8 system for the board-plane -> image homography (DLT). */
export function boardToImage(corners) {
  const src = C.boardCorners;
  const a = [];
  const b = [];
  for (let i = 0; i < 4; i += 1) {
    const [u, v] = src[i];
    const [x, y] = corners[i];
    a.push([u, v, 1, 0, 0, 0, -u * x, -v * x]);
    b.push(x);
    a.push([0, 0, 0, u, v, 1, -u * y, -v * y]);
    b.push(y);
  }
  const h = solve(a, b);
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ];
}

/** Gaussian elimination with partial pivoting. */
function solve(a, b) {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let r = col + 1; r < n; r += 1) {
      if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    }
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const d = m[col][col];
    if (Math.abs(d) < 1e-12) throw new Error("degenerate corner geometry");
    for (let r = 0; r < n; r += 1) {
      if (r === col) continue;
      const f = m[r][col] / d;
      for (let c = col; c <= n; c += 1) m[r][c] -= f * m[col][c];
    }
  }
  return m.map((row, i) => row[n] / m[i][i]);
}

export function invert3(matrix) {
  const [[a, b, c], [d, e, f], [g, h, i]] = matrix;
  const det =
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) throw new Error("singular homography");
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

export function applyHomography(matrix, x, y) {
  const w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
  return [
    (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / w,
    (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / w,
  ];
}

// ---------------------------------------------------------------- detections

/**
 * Raw heads -> detections in the original image's pixels.
 * Mirrors RT-DETR's post-processing with focal loss: sigmoid, then the top
 * `queries` entries of the *flattened* score matrix, so one query can appear
 * under more than one label.
 */
export function decodeDetections(logits, boxes, queries, classes, width, height, threshold) {
  const total = queries * classes;
  const scores = new Float64Array(total);
  for (let i = 0; i < total; i += 1) scores[i] = sigmoid(logits[i]);

  const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);
  const out = [];
  for (const index of order.slice(0, queries)) {
    const score = scores[index];
    if (!(score > threshold)) continue;
    const label = index % classes;
    const query = Math.floor(index / classes);
    const cx = boxes[query * 4];
    const cy = boxes[query * 4 + 1];
    const w = boxes[query * 4 + 2];
    const h = boxes[query * 4 + 3];
    out.push({
      label,
      name: C.labels[label],
      score,
      box: [
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
      ],
    });
  }
  return out;
}

/** Platt scaling: the transform that makes the raw scores mean something. */
export function calibrate(score, scale, bias) {
  const eps = 1e-6;
  const clipped = Math.min(Math.max(score, eps), 1 - eps);
  const logit = Math.log(clipped / (1 - clipped));
  const z = Math.min(Math.max(scale * logit + bias, -35), 35);
  return 1 / (1 + Math.exp(-z));
}

// --------------------------------------------------------------------- grid

/** Assign each detected piece to the square its base stands on, best score winning. */
export function gridFrom(detections, homography) {
  const inverse = invert3(homography);
  const grid = Array.from({ length: 8 }, () => new Array(8).fill(0));
  const best = Array.from({ length: 8 }, () => new Array(8).fill(0));

  for (const detection of detections) {
    if (detection.label === C.boardIndex || detection.label === C.cornerIndex) continue;
    const [x0, y0, x1, y1] = detection.box;
    const fx = x0 + (x1 - x0) * C.footX;
    const fy = y0 + (y1 - y0) * C.footY;
    const [u, v] = applyHomography(inverse, fx, fy);
    const file = Math.floor(u);
    const rank = Math.floor(v);
    if (file < 0 || file > 7 || rank < 0 || rank > 7) continue;
    if (detection.score > best[rank][file]) {
      best[rank][file] = detection.score;
      grid[rank][file] = detection.label + 1;
    }
  }
  return grid;
}

// -------------------------------------------------------------- orientation

/** Mean luminance of each square, sampled through the homography. */
export function squareLuminance(grey, width, height, homography) {
  const offsets = [];
  const half = C.sampleFraction / 2;
  for (let i = 0; i < 3; i += 1) offsets.push(0.5 - half + (i * C.sampleFraction) / 2);

  const out = Array.from({ length: 8 }, () => new Array(8).fill(0));
  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      let sum = 0;
      for (const dv of offsets) {
        for (const du of offsets) {
          const [px, py] = applyHomography(homography, file + du, rank + dv);
          const x = Math.min(Math.max(Math.round(px), 0), width - 1);
          const y = Math.min(Math.max(Math.round(py), 0), height - 1);
          sum += grey[y * width + x];
        }
      }
      out[rank][file] = sum / (offsets.length * offsets.length);
    }
  }
  return out;
}

const rotate = (a, turns) => {
  let out = a.map((row) => [...row]);
  for (let t = 0; t < (turns % 4 + 4) % 4; t += 1) {
    // numpy's rot90: counter-clockwise.
    const n = out.length;
    const next = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let r = 0; r < n; r += 1) {
      for (let c = 0; c < n; c += 1) next[n - 1 - c][r] = out[r][c];
    }
    out = next;
  }
  return out;
};

export function colourScore(luminance) {
  let even = 0;
  let odd = 0;
  for (let r = 0; r < 8; r += 1) {
    for (let f = 0; f < 8; f += 1) {
      if ((r + f) % 2 === 0) even += luminance[r][f];
      else odd += luminance[r][f];
    }
  }
  return even / 32 - odd / 32;
}

export function pieceScore(grid) {
  let near = 0;
  let far = 0;
  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const occupant = grid[rank][file];
      if (!occupant) continue;
      const bottom = rank >= 4;
      const white = occupant >= 1 && occupant <= 6;
      const black = occupant >= 7 && occupant <= 12;
      if (white) {
        near += bottom ? 1 : 0;
        far += bottom ? 0 : 1;
      } else if (black) {
        near += bottom ? 0 : 1;
        far += bottom ? 1 : 0;
      }
    }
  }
  const total = near + far;
  return total ? (near - far) / total : 0;
}

/**
 * Net fraction of pawns standing on their own half of the board, in [-1, 1].
 * No vote (0) unless both colours still have a pawn — a lone runner is exactly
 * the pawn whose position lies about the orientation.
 */
export function pawnHomeScore(grid) {
  let own = 0;
  let total = 0;
  let seenWhite = false;
  let seenBlack = false;
  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const occupant = grid[rank][file];
      if (occupant === 1) {
        seenWhite = true;
        total += 1;
        if (rank >= 4) own += 1;
      } else if (occupant === 7) {
        seenBlack = true;
        total += 1;
        if (rank < 4) own += 1;
      }
    }
  }
  if (!seenWhite || !seenBlack) return 0;
  return (2 * own - total) / total;
}

/**
 * Quarter-turns that put a8 at grid[0][0]. Colour acts as a filter rather than
 * a term in a sum: letting a confident piece vote outweigh it would allow an
 * answer that puts a dark square on a8, which is not a board. Among the
 * survivors, pawns standing on their own half break the 180-degree tie better
 * than the material split, which wrongly assumes White sits nearer the camera.
 */
export function orient(grid, luminance) {
  const candidates = [];
  for (let turns = 0; turns < 4; turns += 1) {
    const rotated = rotate(grid, turns);
    const pieces = pieceScore(rotated);
    const pawns = pawnHomeScore(rotated);
    candidates.push({
      turns,
      colour: colourScore(rotate(luminance, turns)),
      pieces,
      pawns,
      score: pieces + C.pawnHomeWeight * pawns,
    });
  }
  const bestColour = Math.max(...candidates.map((c) => c.colour));
  const surviving = candidates.filter((c) => c.colour >= bestColour - C.minColourMargin);
  return surviving.reduce((a, b) => (b.score > a.score ? b : a));
}

// ---------------------------------------------------------------------- FEN

export function gridToFen(grid) {
  const letters = ["", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"];
  const ranks = [];
  for (let rank = 0; rank < 8; rank += 1) {
    let text = "";
    let empty = 0;
    for (let file = 0; file < 8; file += 1) {
      const occupant = grid[rank][file];
      if (!occupant) {
        empty += 1;
        continue;
      }
      if (empty) {
        text += empty;
        empty = 0;
      }
      text += letters[occupant];
    }
    if (empty) text += empty;
    ranks.push(text);
  }
  return `${ranks.join("/")} w - - 0 1`;
}

// ------------------------------------------------------------------ the read

/** Everything above, in order: two model runs in, one position out. */
export function readPosition({ heatmap, heatSize, logits, boxes, queries, classes },
  { width, height, grey }) {
  const peaks = decodePeaks(heatmap, heatSize, heatSize, C.cornerStride);
  if (peaks.length < 4) return { found: false };

  const scale = (p) => [
    (p[0] * width) / C.cornerSize,
    (p[1] * height) / C.cornerSize,
  ];
  let quad = orderClockwise(peaks.slice(0, 4).map(scale));

  let detections = decodeDetections(
    logits, boxes, queries, classes, width, height, 0,
  );
  if (C.calibration) {
    for (const d of detections) {
      d.score = calibrate(d.score, C.calibration.scale, C.calibration.bias);
    }
    detections = detections.filter((d) => d.score >= C.threshold);
  }

  const homography = boardToImage(quad);
  let grid = gridFrom(detections, homography);
  const luminance = squareLuminance(grey, width, height, homography);
  const { turns, colour, pieces, pawns } = orient(grid, luminance);
  grid = rotate(grid, turns);
  quad = [...quad.slice(turns), ...quad.slice(0, turns)];

  return {
    found: true,
    grid,
    quad,
    detections,
    fen: gridToFen(grid),
    evidence: { colour, pieces, pawns },
  };
}
