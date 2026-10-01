package com.enginebay.vision.core

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/** One component. [box] is in model-input pixels (x1, y1, x2, y2); [coef] are the mask coefficients. */
class Detection(val cls: Int, val score: Float, val box: FloatArray, val coef: FloatArray) {
    val area: Float get() = (box[2] - box[0]) * (box[3] - box[1])
}

/**
 * YOLO11-seg post-processing, as Ultralytics predict(): best class per anchor >= conf, class-aware NMS (IoU, max_det),
 * masks = coef . prototypes, zeroed outside the box at prototype resolution, upsampled bilinearly
 * (align_corners = false) to the input size and thresholded at logit 0 (sigmoid 0.5).
 */
object Postprocess {

    /** [out0]: (4 + nc + nm) x anchors, row-major. */
    fun decode(out0: FloatArray, anchors: Int, nc: Int, conf: Float, iou: Float, maxDet: Int): List<Detection> {
        val nm = out0.size / anchors - 4 - nc
        val cand = ArrayList<Detection>()
        for (a in 0 until anchors) {
            var best = -1f; var bc = 0
            for (c in 0 until nc) {
                val s = out0[(4 + c) * anchors + a]
                if (s > best) { best = s; bc = c }
            }
            if (best < conf) continue
            val cx = out0[a]; val cy = out0[anchors + a]; val w = out0[2 * anchors + a]; val h = out0[3 * anchors + a]
            val coef = FloatArray(nm) { k -> out0[(4 + nc + k) * anchors + a] }
            cand.add(Detection(bc, best, floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), coef))
        }
        cand.sortByDescending { it.score }
        val keep = ArrayList<Detection>()
        for (d in cand) {
            if (keep.size >= maxDet) break
            if (keep.none { it.cls == d.cls && iou(it.box, d.box) > iou }) keep.add(d)
        }
        return keep
    }

    fun iou(a: FloatArray, b: FloatArray): Float {
        val ix = max(0f, min(a[2], b[2]) - max(a[0], b[0]))
        val iy = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val i = ix * iy
        return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9f)
    }

    /**
     * Masks at input resolution. Returns an index map (size x size): -1 = background, else the index in [dets] of the
     * detection drawn on top (the highest score wins where masks overlap). [proto]: nm x mh x mw, row-major.
     */
    fun maskIndex(dets: List<Detection>, proto: FloatArray, nm: Int, mh: Int, mw: Int, size: Int): IntArray {
        val index = IntArray(size * size) { -1 }
        val n = mh * mw
        val logits = FloatArray(n)
        val scale = mw.toFloat() / size
        for (di in dets.indices.reversed()) {  // lowest score first, so higher scores overwrite
            val det = dets[di]
            logits.fill(0f)
            for (k in 0 until nm) {
                val c = det.coef[k]; val off = k * n
                for (i in 0 until n) logits[i] += c * proto[off + i]
            }
            // Ultralytics crop_mask: zero the logits outside the box at PROTOTYPE resolution (box * mw/size, clamped
            // at 0, rounded half-even), then upsample. Pixels more than one prototype cell outside stay 0.
            val px1 = round(max(0f, det.box[0]) * scale).toInt(); val py1 = round(max(0f, det.box[1]) * scale).toInt()
            val px2 = round(max(0f, det.box[2]) * scale).toInt(); val py2 = round(max(0f, det.box[3]) * scale).toInt()
            for (r in 0 until mh) for (c in 0 until mw) {
                if (r < py1 || r >= py2 || c < px1 || c >= px2) logits[r * mw + c] = 0f
            }
            val inv = 1f / scale
            val xa = max(0, floor((px1 - 1) * inv).toInt()); val xb = min(size, ceil((px2 + 1) * inv).toInt())
            val ya = max(0, floor((py1 - 1) * inv).toInt()); val yb = min(size, ceil((py2 + 1) * inv).toInt())
            for (y in ya until yb) {
                val fy = max(0f, (y + 0.5f) * scale - 0.5f)  // torch bilinear, align_corners = false
                val r0 = min(floor(fy).toInt(), mh - 1); val r1 = min(r0 + 1, mh - 1); val wy = fy - r0
                for (x in xa until xb) {
                    val fx = max(0f, (x + 0.5f) * scale - 0.5f)
                    val c0 = min(floor(fx).toInt(), mw - 1); val c1 = min(c0 + 1, mw - 1); val wx = fx - c0
                    val v = (logits[r0 * mw + c0] * (1 - wx) + logits[r0 * mw + c1] * wx) * (1 - wy) +
                        (logits[r1 * mw + c0] * (1 - wx) + logits[r1 * mw + c1] * wx) * wy
                    if (v > 0f) index[y * size + x] = di
                }
            }
        }
        return index
    }
}
