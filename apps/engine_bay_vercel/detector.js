// YOLO11-seg inference in the browser (onnxruntime-web, WASM). Mirrors Ultralytics predict():
// letterbox (114 padding, centred) -> model -> confidence filter -> class-aware NMS -> masks = coef x prototypes,
// cropped to the box and upsampled to the input size, thresholded at logit 0 (sigmoid 0.5).
import * as ort from "./vendor/ort/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

export class Detector {
  static async load(config, onProgress) {
    const res = await fetch(config.model);
    if (!res.ok) throw new Error(`model download failed (${res.status})`);
    const total = Number(res.headers.get("content-length")) || 0;
    const reader = res.body.getReader();
    const chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      onProgress?.(total ? got / total : 0);
    }
    const bytes = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { bytes.set(c, o); o += c.length; }
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    return new Detector(config, session);
  }

  static async loadFromBytes(config, bytes, onProgress) {
    onProgress?.(0.3);
    const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    onProgress?.(0.6);
    const session = await ort.InferenceSession.create(u8, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    onProgress?.(1.0);
    return new Detector(config, session);
  }

  constructor(config, session) {
    this.cfg = { ...config };
    this.session = session;
    this.nc = config.names ? config.names.length : 21;
    this.size = config.imgsz || 640;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = this.size;
    this.threads = ort.env.wasm.numThreads;
  }

  // Same preprocessing as the Python app: downscale to MAX_SIDE (area filter), then Ultralytics' LetterBox, whose
  // cv2.INTER_LINEAR resize is reproduced exactly (half-pixel centres, no antialiasing). The browser's own canvas
  // resampling filters differently and moved scores by up to 0.3, so it is only used for the first (area) step.
  letterbox(img) {
    const S = this.size, MAX_SIDE = 1600, w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    const k0 = Math.min(1, MAX_SIDE / Math.max(w, h)), sw = Math.round(w * k0), sh = Math.round(h * k0);
    const src = document.createElement("canvas");
    src.width = sw; src.height = sh;
    const sctx = src.getContext("2d", { willReadFrequently: true });
    sctx.imageSmoothingQuality = "high";
    sctx.drawImage(img, 0, 0, sw, sh);
    const px = sctx.getImageData(0, 0, sw, sh).data;

    const r = Math.min(S / sh, S / sw);
    const nw = Math.round(sw * r), nh = Math.round(sh * r);
    const left = Math.round((S - nw) / 2 - 0.1), top = Math.round((S - nh) / 2 - 0.1);
    const n = S * S, t = new Float32Array(3 * n).fill(114 / 255);
    const fx = sw / nw, fy = sh / nh;
    for (let y = 0; y < nh; y++) {
      let syf = (y + 0.5) * fy - 0.5;
      if (syf < 0) syf = 0;
      let y0 = Math.floor(syf), wy = syf - y0;
      if (y0 >= sh - 1) { y0 = sh - 1; wy = 0; }
      const y1 = Math.min(y0 + 1, sh - 1), row = (top + y) * S + left;
      for (let x = 0; x < nw; x++) {
        let sxf = (x + 0.5) * fx - 0.5;
        if (sxf < 0) sxf = 0;
        let x0 = Math.floor(sxf), wx = sxf - x0;
        if (x0 >= sw - 1) { x0 = sw - 1; wx = 0; }
        const x1 = Math.min(x0 + 1, sw - 1);
        const a = 4 * (y0 * sw + x0), b = 4 * (y0 * sw + x1), c = 4 * (y1 * sw + x0), d = 4 * (y1 * sw + x1);
        const w00 = (1 - wx) * (1 - wy), w01 = wx * (1 - wy), w10 = (1 - wx) * wy, w11 = wx * wy, o = row + x;
        for (let ch = 0; ch < 3; ch++) {
          const v = px[a + ch] * w00 + px[b + ch] * w01 + px[c + ch] * w10 + px[d + ch] * w11;
          t[ch * n + o] = Math.round(v) / 255;  // cv2 returns uint8
        }
      }
    }
    // r maps input (640) pixels to ORIGINAL image pixels: downscale k0 and letterbox r combined
    return { tensor: new ort.Tensor("float32", t, [1, 3, S, S]), r: r * k0, left, top, nw, nh, w, h };
  }

  /** conf: minimum score kept before NMS. Returns {dets, lb, proto, ms, inferMs, preprocessMs, postprocessMs}; masks are rendered on demand. */
  async detect(img, conf) {
    const t0 = performance.now();
    const lb = this.letterbox(img);
    const tInfer0 = performance.now();
    const out = await this.session.run({ [this.session.inputNames[0]]: lb.tensor });
    const tInfer1 = performance.now();
    const pred = out[this.session.outputNames[0]], proto = out[this.session.outputNames[1]];
    const A = pred.dims[2], d = pred.data;
    const nm = proto && proto.dims ? proto.dims[1] : 32;
    const detectedNc = pred.dims[1] - 4 - nm;
    const nc = detectedNc > 0 ? detectedNc : this.nc;
    const cand = [];
    for (let a = 0; a < A; a++) {
      let best = -1, bc = 0;
      for (let c = 0; c < nc; c++) { const s = d[(4 + c) * A + a]; if (s > best) { best = s; bc = c; } }
      if (best < conf) continue;
      const cx = d[a], cy = d[A + a], bw = d[2 * A + a], bh = d[3 * A + a];
      const coef = new Float32Array(nm);
      for (let k = 0; k < nm; k++) coef[k] = d[(4 + nc + k) * A + a];
      cand.push({ cls: bc, score: best, box: [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], coef });
    }
    cand.sort((p, q) => q.score - p.score);
    const keep = [];
    for (const c of cand) {  // class-aware greedy NMS
      if (keep.length >= this.cfg.max_det) break;
      if (!keep.some((k) => k.cls === c.cls && iou(k.box, c.box) > this.cfg.iou)) keep.push(c);
    }
    const tNms1 = performance.now();
    const dets = keep.map((k) => {
      const name = (this.cfg.names && this.cfg.names[k.cls]) || `class_${k.cls}`;
      const name_vi = (this.cfg.names_vi && this.cfg.names_vi[k.cls]) || name;
      const name_en = (this.cfg.names_en && this.cfg.names_en[k.cls]) || name;
      return {
        ...k,
        name,
        name_vi,
        name_en,
        // box in original image pixels
        xyxy: [(k.box[0] - lb.left) / lb.r, (k.box[1] - lb.top) / lb.r, (k.box[2] - lb.left) / lb.r, (k.box[3] - lb.top) / lb.r]
          .map((v, i) => Math.max(0, Math.min(v, i % 2 ? lb.h : lb.w))),
      };
    });
    return {
      dets,
      lb,
      proto,
      ms: performance.now() - t0,
      inferMs: tInfer1 - tInfer0,
      preprocessMs: tInfer0 - t0,
      postprocessMs: tNms1 - tInfer1,
    };
  }

  /** Run multi-iteration benchmark with warmup and collect statistical latency metrics. */
  async benchmark(img, iterations = 10, conf = 0.35, onProgress = null) {
    // 2 Warmup runs
    for (let w = 0; w < 2; w++) {
      await this.detect(img, conf);
    }
    const runs = [];
    for (let i = 0; i < iterations; i++) {
      onProgress?.(i / iterations);
      const res = await this.detect(img, conf);
      runs.push(res);
    }
    onProgress?.(1.0);
    const lats = runs.map((r) => r.ms).sort((a, b) => a - b);
    const infers = runs.map((r) => r.inferMs).sort((a, b) => a - b);
    const sum = lats.reduce((a, b) => a + b, 0);
    const avg = sum / lats.length;
    const median = lats[Math.floor(lats.length / 2)];
    const min = lats[0];
    const max = lats[lats.length - 1];
    const p95 = lats[Math.min(lats.length - 1, Math.floor(lats.length * 0.95))];
    const fps = 1000 / (avg || 1);
    const stdDev = Math.sqrt(lats.reduce((acc, v) => acc + (v - avg) ** 2, 0) / lats.length);
    const avgInfer = infers.reduce((a, b) => a + b, 0) / infers.length;

    return {
      iterations,
      avg: Math.round(avg * 10) / 10,
      avgInfer: Math.round(avgInfer * 10) / 10,
      median: Math.round(median * 10) / 10,
      min: Math.round(min * 10) / 10,
      max: Math.round(max * 10) / 10,
      p95: Math.round(p95 * 10) / 10,
      fps: Math.round(fps * 10) / 10,
      stdDev: Math.round(stdDev * 10) / 10,
      detsCount: runs[runs.length - 1].dets.length,
      lastResult: runs[runs.length - 1],
    };
  }

  /** Coloured mask overlay at input resolution (size x size RGBA canvas) for the given detections. */
  renderMasks(dets, proto, colorOf, alpha = 110) {
    const S = this.size, [, nm, mh, mw] = proto.dims, P = proto.data, n = mh * mw, scale = mw / S;
    const img = new ImageData(S, S), out = img.data;
    const logits = new Float32Array(n);
    for (const det of [...dets].reverse()) {  // highest score drawn last = on top
      logits.fill(0);
      for (let k = 0; k < nm; k++) {
        const c = det.coef[k], off = k * n;
        for (let i = 0; i < n; i++) logits[i] += c * P[off + i];
      }
      const [x1, y1, x2, y2] = det.box;
      const [r, g, b] = colorOf(det.cls);
      const X1 = Math.max(0, Math.floor(x1)), Y1 = Math.max(0, Math.floor(y1));
      const X2 = Math.min(S, Math.ceil(x2)), Y2 = Math.min(S, Math.ceil(y2));
      for (let y = Y1; y < Y2; y++) {
        if (y + 0.5 < y1 || y + 0.5 > y2) continue;
        const fy = Math.min(Math.max((y + 0.5) * scale - 0.5, 0), mh - 1), y0 = Math.floor(fy), yb = Math.min(y0 + 1, mh - 1), wy = fy - y0;
        for (let x = X1; x < X2; x++) {
          if (x + 0.5 < x1 || x + 0.5 > x2) continue;
          const fx = Math.min(Math.max((x + 0.5) * scale - 0.5, 0), mw - 1), x0 = Math.floor(fx), xb = Math.min(x0 + 1, mw - 1), wx = fx - x0;
          const v = (logits[y0 * mw + x0] * (1 - wx) + logits[y0 * mw + xb] * wx) * (1 - wy)
                  + (logits[yb * mw + x0] * (1 - wx) + logits[yb * mw + xb] * wx) * wy;
          if (v > 0) { const o = 4 * (y * S + x); out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = alpha; }
        }
      }
    }
    const c = document.createElement("canvas");
    c.width = c.height = S;
    c.getContext("2d").putImageData(img, 0, 0);
    return c;
  }
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = ix * iy;
  return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9);
}
