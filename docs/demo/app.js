// The "try it" panel: two models in the browser, one position out.
//
// The weights are deliberately not fetched on page load. They are 76 MB, most
// visitors are here to read, and downloading that unasked would be rude.

import { C } from "./constants.js";
import { cornerInput, detectorInput, readPosition } from "./pipeline.js";

const QUAD_COLOUR = "#e0af45";
const WHITE_COLOUR = "#7dd3fc";
const BLACK_COLOUR = "#f472b6";

// Resolved against this module rather than against the page: a relative fetch
// resolves against the *document*, so "../models" would climb out of the site
// root whenever the page is served from it.
const ROOT = new URL("../", import.meta.url);
const asset = (path) => new URL(path, ROOT).href;

const el = (id) => document.getElementById(id);
const state = { detector: null, corners: null, busy: false };

function setStatus(text, kind = "idle", busy = false) {
  const status = el("try-status");
  status.dataset.kind = kind;
  status.textContent = "";
  if (busy) {
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    status.append(spinner);
  }
  status.append(text);
}

/** Fetch with a progress readout, because 76 MB in silence looks broken. */
async function fetchModel(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function loadModels() {
  if (state.detector) return;
  el("try-load").disabled = true;
  const megabytes = (n) => (n / 1e6).toFixed(0);

  try {
    ort.env.wasm.wasmPaths = asset("vendor/");
    ort.env.wasm.numThreads = 1;

    let done = 0;
    const files = [
      ["corners", asset("models/corners.onnx")],
      ["detector", asset("models/detector.onnx")],
    ];
    for (const [name, url] of files) {
      const bytes = await fetchModel(url, (received, total) => {
        const pct = total ? ` ${Math.round((100 * received) / total)}%` : "";
        setStatus(
          `downloading ${name} ${megabytes(received)} MB${pct}`,
          "idle",
          true,
        );
      });
      setStatus(`starting ${name}…`, "idle", true);
      state[name] = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      done += 1;
    }
    setStatus(`${done} models ready — drop a photograph`, "ok");
    el("try-drop").hidden = false;
    el("try-load").hidden = true;
  } catch (error) {
    setStatus(`could not load the models: ${error.message}`, "error");
    el("try-load").disabled = false;
  }
}

/** A greyscale copy at full size, for the orientation step. */
function greyOf(canvas) {
  const { width, height } = canvas;
  const { data } = canvas.getContext("2d").getImageData(0, 0, width, height);
  const grey = new Float32Array(width * height);
  for (let i = 0; i < grey.length; i += 1) {
    // Rec. 601, matching PIL's "L" conversion.
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    grey[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  }
  return grey;
}

async function run(file) {
  if (state.busy || !state.detector) return;
  state.busy = true;
  setStatus(`reading ${file.name}…`, "idle", true);

  try {
    const bitmap = await createImageBitmap(file);
    // Work at a bounded size: a 12-megapixel phone photo makes the greyscale
    // pass slow for no gain, since both models see a small square anyway.
    const longest = Math.max(bitmap.width, bitmap.height);
    const scale = Math.min(1, 1280 / longest);
    const width = Math.round(bitmap.width * scale);
    const height = Math.round(bitmap.height * scale);

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(bitmap, 0, 0, width, height);

    const cornerTensor = new ort.Tensor(
      "float32",
      cornerInput(canvas, C.cornerSize),
      [1, 3, C.cornerSize, C.cornerSize],
    );
    const heat = await state.corners.run({ image: cornerTensor });
    const heatmap = heat.heatmap.data;
    const heatSize = heat.heatmap.dims.at(-1);

    const detectorTensor = new ort.Tensor(
      "float32",
      detectorInput(canvas, C.detectorSize),
      [1, 3, C.detectorSize, C.detectorSize],
    );
    const out = await state.detector.run({ pixel_values: detectorTensor });
    const logits = out.logits.data;
    const boxes = out.pred_boxes.data;
    const [, queries, classes] = out.logits.dims;

    const result = readPosition(
      { heatmap, heatSize, logits, boxes, queries, classes },
      { width, height, grey: greyOf(canvas) },
    );

    if (!result.found) {
      setStatus("no board found — the corner model found fewer than four corners", "none");
      el("try-result").hidden = true;
      return;
    }

    drawOverlay(canvas, result);
    drawBoard(result.grid);
    el("try-fen").textContent = result.fen;
    const placed = result.grid.flat().filter(Boolean).length;
    el("try-counts").textContent =
      `${placed} pieces placed · ${result.detections.length} boxes · ${width}×${height}`;
    el("try-result").hidden = false;
    setStatus("position read", "ok");
  } catch (error) {
    setStatus(`that did not work: ${error.message}`, "error");
  } finally {
    state.busy = false;
  }
}

function drawOverlay(source, result) {
  const canvas = el("try-overlay");
  canvas.width = source.width;
  canvas.height = source.height;
  const context = canvas.getContext("2d");
  context.drawImage(source, 0, 0);
  context.lineWidth = Math.max(2, source.width / 400);

  for (const detection of result.detections) {
    if (detection.label === C.boardIndex || detection.label === C.cornerIndex) continue;
    const [x0, y0, x1, y1] = detection.box;
    context.strokeStyle = detection.name.startsWith("white") ? WHITE_COLOUR : BLACK_COLOUR;
    context.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }

  context.strokeStyle = QUAD_COLOUR;
  context.lineWidth = Math.max(3, source.width / 300);
  context.beginPath();
  result.quad.forEach(([x, y], i) => (i ? context.lineTo(x, y) : context.moveTo(x, y)));
  context.closePath();
  context.stroke();
}

/** A diagram in the same colours as the ones baked into the page. */
function drawBoard(grid) {
  const size = 48;
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${size * 8} ${size * 8}">`,
  ];
  for (let rank = 0; rank < 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const light = (rank + file) % 2 === 0;
      parts.push(
        `<rect x="${file * size}" y="${rank * size}" width="${size}" ` +
          `height="${size}" fill="${light ? "#e6e8df" : "#9aa392"}"/>`,
      );
      const occupant = grid[rank][file];
      if (!occupant) continue;
      parts.push(
        `<use href="#${C.labels[occupant - 1].replace("_", "-")}" ` +
          `transform="translate(${file * size + size * 0.05}, ` +
          `${rank * size + size * 0.05}) scale(${(size * 0.9) / 45})"/>`,
      );
    }
  }
  parts.push("</svg>");
  el("try-board").innerHTML = parts.join("");
}

el("try-load").addEventListener("click", loadModels);
el("try-file").addEventListener("change", (event) => run(event.target.files[0]));
el("try-pick").addEventListener("click", () => el("try-file").click());

const drop = el("try-drop");
["dragenter", "dragover"].forEach((name) =>
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.add("is-over");
  }),
);
["dragleave", "drop"].forEach((name) =>
  drop.addEventListener(name, (event) => {
    event.preventDefault();
    drop.classList.remove("is-over");
  }),
);
drop.addEventListener("drop", (event) => run(event.dataTransfer.files[0]));
