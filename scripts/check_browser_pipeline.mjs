// Run the browser pipeline over fixtures dumped from Python and compare.
//
// The JavaScript in docs/demo/pipeline.js reimplements rules that live in
// Python. This is what stops the two drifting: identical inputs in, and the
// corner quad, the square luminances, the orientation vote and the resulting
// grid all have to match.
//
//   uv run python scripts/dump_browser_fixture.py ... --out fixtures.json
//   node scripts/check_browser_pipeline.mjs fixtures.json
//
// Exits non-zero on the first disagreement outside tolerance.
//
// Scope: this checks the logic, given identical model outputs. It does not
// check preprocessing — the browser resizes with canvas drawImage and Python
// with PIL's bilinear, and those do not agree to the last bit. The effect is
// the same one the ONNX backend already has: a detection within a thousandth
// of the threshold can flip, changing a square. Measured end to end, the
// browser and Python int8 read the same photograph to within one square.

import { readFileSync } from "node:fs";
import {
  boardToImage,
  colourScore,
  decodePeaks,
  gridFrom,
  orderClockwise,
  orient,
  pieceScore,
  squareLuminance,
} from "../docs/demo/pipeline.js";
import { C } from "../docs/demo/constants.js";

const QUAD_TOLERANCE = 0.05; // pixels
const LUMINANCE_TOLERANCE = 1e-6;
// Python samples luminance in float32 and JavaScript has only float64, so the
// two means differ around 1e-8. Loose enough to ignore that, far tighter than
// any logic error: a wrong parity or a wrong rotation moves these by tenths,
// and `turns` is compared exactly regardless.
const SCORE_TOLERANCE = 1e-6;

const path = process.argv[2];
if (!path) {
  console.error("usage: node scripts/check_browser_pipeline.mjs <fixtures.json>");
  process.exit(2);
}

const cases = JSON.parse(readFileSync(path, "utf8"));
let failures = 0;

const fail = (id, what, detail) => {
  failures += 1;
  console.error(`  FAIL ${id}: ${what}\n       ${detail}`);
};

for (const item of cases) {
  const { id, width, height, heatSize, expected } = item;
  const heatmap = Float32Array.from(item.heatmap);
  const grey = Float32Array.from(item.grey);

  // 1. corner peaks -> quad
  const peaks = decodePeaks(heatmap, heatSize, heatSize, C.cornerStride);
  const quad = orderClockwise(
    peaks.slice(0, 4).map((p) => [
      (p[0] * width) / C.cornerSize,
      (p[1] * height) / C.cornerSize,
    ]),
  );
  let worst = 0;
  for (let i = 0; i < 4; i += 1) {
    worst = Math.max(
      worst,
      Math.hypot(quad[i][0] - expected.quad[i][0], quad[i][1] - expected.quad[i][1]),
    );
  }
  if (worst > QUAD_TOLERANCE) {
    fail(id, "corner quad", `worst corner off by ${worst.toFixed(4)} px`);
  }

  // 2. homography -> square luminance
  const homography = boardToImage(expected.quad);
  const luminance = squareLuminance(grey, width, height, homography);
  let worstLuminance = 0;
  for (let r = 0; r < 8; r += 1) {
    for (let f = 0; f < 8; f += 1) {
      worstLuminance = Math.max(
        worstLuminance,
        Math.abs(luminance[r][f] - expected.luminance[r][f]),
      );
    }
  }
  if (worstLuminance > LUMINANCE_TOLERANCE) {
    fail(id, "square luminance", `worst square off by ${worstLuminance.toExponential(2)}`);
  }

  // 3. detections -> grid
  const grid = gridFrom(item.detections, homography);
  const wrong = [];
  for (let r = 0; r < 8; r += 1) {
    for (let f = 0; f < 8; f += 1) {
      if (grid[r][f] !== expected.grid[r][f]) {
        wrong.push(`[${r}][${f}] js=${grid[r][f]} py=${expected.grid[r][f]}`);
      }
    }
  }
  if (wrong.length) fail(id, "grid assignment", wrong.join(", "));

  // 4. orientation vote
  const vote = orient(expected.grid, expected.luminance);
  if (vote.turns !== expected.turns) {
    fail(id, "orientation", `js turns=${vote.turns} py turns=${expected.turns}`);
  }
  if (Math.abs(vote.colour - expected.colour) > SCORE_TOLERANCE) {
    fail(id, "colour score", `js=${vote.colour} py=${expected.colour}`);
  }
  if (Math.abs(vote.pieces - expected.pieces) > SCORE_TOLERANCE) {
    fail(id, "piece score", `js=${vote.pieces} py=${expected.pieces}`);
  }

  // 5. the scores on their own, so a failure above localises
  const cs = colourScore(expected.luminance);
  const ps = pieceScore(expected.grid);
  if (!Number.isFinite(cs) || !Number.isFinite(ps)) {
    fail(id, "scores", "non-finite");
  }

  if (!failures) console.log(`  ok   ${id}`);
}

if (failures) {
  console.error(`\n${failures} disagreement(s) between the browser and Python.`);
  process.exit(1);
}
console.log(`\n${cases.length} photographs: browser pipeline matches Python.`);
