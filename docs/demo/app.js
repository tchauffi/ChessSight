// The "try it" panel: two models in the browser, one position out.
//
// The weights are deliberately not fetched on page load. They are 145 MB, most
// visitors are here to read, and downloading that unasked would be rude. Once
// fetched they are kept in Cache Storage, so a second visit starts at once.
//
// Inference runs on WebGPU where the browser can run these fp16 graphs on the
// GPU, and on the CPU (WebAssembly) everywhere else. `?backend=wasm` or
// `?backend=webgpu` in the URL forces one.

import { C } from "./constants.js";
import {
  boardCrop,
  cornerInput,
  detectorInput,
  findQuad,
  readBoard,
  viewDetections,
} from "./pipeline.js";

const QUAD_COLOUR = "#e0af45";
const WHITE_COLOUR = "#7dd3fc";
const BLACK_COLOUR = "#f472b6";

// Resolved against this module rather than against the page: a relative fetch
// resolves against the *document*, so "../models" would climb out of the site
// root whenever the page is served from it.
const ROOT = new URL("../", import.meta.url);
const asset = (path) => new URL(path, ROOT).href;

const el = (id) => document.getElementById(id);
const state = { detector: null, corners: null, backend: null, busy: false };
const LABEL = { webgpu: "WebGPU", wasm: "the CPU" };
const MODEL_FILES = [
  ["corners", "models/corners.onnx"],
  ["detector", "models/detector.onnx"],
];

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

/** Fetch with a progress readout, because 145 MB in silence looks broken. */
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

// ------------------------------------------------------------ model cache

const CACHE_PREFIX = "chesssight-models-";

/**
 * Cache Storage named after the bundle's content hash, so a new export can
 * never be served from an old visit. Null where caching is unavailable (an
 * insecure origin, some private modes): every path below then just downloads.
 */
async function modelCache() {
  try {
    if (!C.modelVersion || !("caches" in self)) return null;
    const name = CACHE_PREFIX + C.modelVersion;
    for (const key of await caches.keys()) {
      if (key.startsWith(CACHE_PREFIX) && key !== name) await caches.delete(key);
    }
    return await caches.open(name);
  } catch {
    return null;
  }
}

async function getModel(url, onProgress) {
  const cache = await modelCache();
  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) return new Uint8Array(await hit.arrayBuffer());
    } catch {
      // fall through to the network
    }
  }
  const bytes = await fetchModel(url, onProgress);
  if (cache) {
    try {
      await cache.put(url, new Response(bytes));
    } catch {
      // over quota: the next visit downloads again, which is all that is lost
    }
  }
  return bytes;
}

async function allCached() {
  const cache = await modelCache();
  if (!cache) return false;
  try {
    const hits = await Promise.all(MODEL_FILES.map(([, path]) => cache.match(asset(path))));
    return hits.every(Boolean);
  } catch {
    return false;
  }
}

// --------------------------------------------------------------- backends

/**
 * WebGPU only where it can keep these fp16 graphs on the GPU. Without the
 * adapter's `shader-f16` feature the half-precision kernels would fall back
 * to the CPU node by node, which is slower than running on the CPU outright.
 */
async function pickBackend() {
  const forced = new URLSearchParams(location.search).get("backend");
  if (forced === "wasm" || forced === "webgpu") return forced;
  try {
    const adapter = navigator.gpu && (await navigator.gpu.requestAdapter());
    if (adapter && adapter.features.has("shader-f16")) return "webgpu";
  } catch {
    // no WebGPU here
  }
  return "wasm";
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.append(script);
  });
}

const megabytes = (n) => (n / 1e6).toFixed(0);

function downloadProgress(name, received, total) {
  const pct = total ? ` ${Math.round((100 * received) / total)}%` : "";
  setStatus(`downloading ${name} ${megabytes(received)} MB${pct}`, "idle", true);
}

async function createSessions(backend) {
  const sessions = {};
  for (const [name, path] of MODEL_FILES) {
    const bytes = await getModel(asset(path), (received, total) =>
      downloadProgress(name, received, total),
    );
    setStatus(`starting ${name} on ${LABEL[backend]}…`, "idle", true);
    sessions[name] = await ort.InferenceSession.create(bytes, {
      executionProviders: [backend],
      graphOptimizationLevel: "all",
    });
  }
  return sessions;
}

/** Put the sessions on the CPU, after WebGPU failed to create or to run. */
async function fallBackToCpu() {
  Object.assign(state, await createSessions("wasm"), { backend: "wasm" });
}

async function loadModels() {
  if (state.detector) return;
  el("try-load").disabled = true;

  try {
    const backend = await pickBackend();
    // The WebGPU build of onnxruntime carries the wasm backend too, so either
    // script serves the CPU fallback; the plain build is half the download.
    await loadScript(
      asset(backend === "webgpu" ? "vendor/ort.webgpu.min.js" : "vendor/ort.min.js"),
    );
    ort.env.wasm.wasmPaths = asset("vendor/");
    // Threads need cross-origin isolation, which GitHub Pages cannot send.
    ort.env.wasm.numThreads = 1;

    try {
      Object.assign(state, await createSessions(backend), { backend });
    } catch (error) {
      if (backend !== "webgpu") throw error;
      await fallBackToCpu();
    }
    setStatus(`models ready on ${LABEL[state.backend]} — drop a photograph`, "ok");
    el("try-drop").hidden = false;
    el("try-load").hidden = true;
  } catch (error) {
    setStatus(`could not load the models: ${error.message}`, "error");
    el("try-load").disabled = false;
  }
}

/** The square board crop as its own canvas, optionally mirrored left to right. */
function cropCanvas(source, [left, top, side], mirrored) {
  const canvas = document.createElement("canvas");
  canvas.width = side;
  canvas.height = side;
  const context = canvas.getContext("2d");
  if (mirrored) {
    context.translate(side, 0);
    context.scale(-1, 1);
  }
  context.drawImage(source, left, top, side, side, 0, 0, side, side);
  return canvas;
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
    const started = performance.now();
    let result;
    try {
      result = await readPhoto(bitmap, file.name);
    } catch (error) {
      if (state.backend !== "webgpu") throw error;
      setStatus("WebGPU failed on that photograph — switching to the CPU…", "idle", true);
      await fallBackToCpu();
      result = await readPhoto(bitmap, file.name);
    }
    if (!result) {
      setStatus("no board found — the corner model found fewer than four corners", "none");
      el("try-result").hidden = true;
      return;
    }
    const seconds = ((performance.now() - started) / 1000).toFixed(1);

    drawOverlay(result.canvas, result);
    drawBoard(result.grid);
    el("try-fen").textContent = result.fen;
    const placed = result.grid.flat().filter(Boolean).length;
    el("try-counts").textContent =
      `${placed} pieces placed · ${result.detections.length} boxes · ` +
      `${result.canvas.width}×${result.canvas.height}`;
    el("try-result").hidden = false;
    setStatus(`position read in ${seconds} s on ${LABEL[state.backend]}`, "ok");
  } catch (error) {
    setStatus(`that did not work: ${error.message}`, "error");
  } finally {
    state.busy = false;
  }
}

/** One photograph through both models; null when no board is found. */
async function readPhoto(bitmap, name) {
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
  const quad = findQuad(heat.heatmap.data, heat.heatmap.dims.at(-1), width, height);
  if (!quad) return null;

  // Three looks at the board: the whole frame, a crop that gives small
  // pieces real pixels, and that crop mirrored. Their scores are averaged.
  const crop = boardCrop(quad, width, height);
  const views = {
    full: canvas,
    crop: cropCanvas(canvas, crop, false),
    mirrored: cropCanvas(canvas, crop, true),
  };
  const passes = {};
  for (const [pass, view] of Object.entries(views)) {
    setStatus(`reading ${name} — ${pass} pass…`, "idle", true);
    const tensor = new ort.Tensor(
      "float32",
      detectorInput(view, C.detectorSize),
      [1, 3, C.detectorSize, C.detectorSize],
    );
    const out = await state.detector.run({ pixel_values: tensor });
    const [, queries, classes] = out.logits.dims;
    passes[pass] = viewDetections(
      { logits: out.logits.data, boxes: out.pred_boxes.data, queries, classes },
      view.width,
      view.height,
    );
  }

  const result = readBoard(quad, passes, { width, height, grey: greyOf(canvas) }, crop);
  return { ...result, canvas };
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
allCached().then((cached) => {
  if (cached) el("try-load").textContent = "Load the models · cached";
});
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
