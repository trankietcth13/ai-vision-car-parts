package com.enginebay.vision.core

import kotlin.math.floor
import kotlin.math.min
import kotlin.math.round

/** Where the photo sits inside the square model input. [scale] maps photo pixels to input pixels. */
data class Letterbox(val scale: Float, val left: Int, val top: Int, val newW: Int, val newH: Int, val srcW: Int, val srcH: Int)

/**
 * Ultralytics LetterBox for a static square model input: resize with cv2.INTER_LINEAR semantics (half-pixel centres,
 * no antialiasing), centre on a 114-grey canvas. Rounding follows Python (round-half-even), so the geometry is
 * identical to Ultralytics; pixel values differ from cv2 by at most 1/255 (cv2 uses fixed-point weights).
 */
object Preprocess {
    const val PAD = 114

    fun geometry(srcW: Int, srcH: Int, size: Int): Letterbox {
        val r = min(size.toDouble() / srcH, size.toDouble() / srcW)
        val newW = round(srcW * r).toInt()
        val newH = round(srcH * r).toInt()
        val dw = (size - newW) / 2.0
        val dh = (size - newH) / 2.0
        return Letterbox(r.toFloat(), round(dw - 0.1).toInt(), round(dh - 0.1).toInt(), newW, newH, srcW, srcH)
    }

    /** [argb]: packed 0xAARRGGBB pixels (Bitmap.getPixels order). Returns the NCHW float tensor (RGB / 255). */
    fun letterbox(argb: IntArray, srcW: Int, srcH: Int, size: Int): Pair<FloatArray, Letterbox> {
        val lb = geometry(srcW, srcH, size)
        val n = size * size
        val t = FloatArray(3 * n) { PAD / 255f }
        val fx = srcW.toDouble() / lb.newW
        val fy = srcH.toDouble() / lb.newH
        // precompute horizontal taps (same for every row)
        val x0s = IntArray(lb.newW); val x1s = IntArray(lb.newW); val wxs = FloatArray(lb.newW)
        for (x in 0 until lb.newW) {
            var sx = (x + 0.5) * fx - 0.5
            if (sx < 0) sx = 0.0
            var x0 = floor(sx).toInt(); var wx = sx - x0
            if (x0 >= srcW - 1) { x0 = srcW - 1; wx = 0.0 }
            x0s[x] = x0; x1s[x] = min(x0 + 1, srcW - 1); wxs[x] = wx.toFloat()
        }
        for (y in 0 until lb.newH) {
            var sy = (y + 0.5) * fy - 0.5
            if (sy < 0) sy = 0.0
            var y0 = floor(sy).toInt(); var wy = (sy - y0).toFloat()
            if (y0 >= srcH - 1) { y0 = srcH - 1; wy = 0f }
            val y1 = min(y0 + 1, srcH - 1)
            val r0 = y0 * srcW; val r1 = y1 * srcW
            val row = (lb.top + y) * size + lb.left
            for (x in 0 until lb.newW) {
                val wx = wxs[x]
                val a = argb[r0 + x0s[x]]; val b = argb[r0 + x1s[x]]; val c = argb[r1 + x0s[x]]; val d = argb[r1 + x1s[x]]
                val w00 = (1 - wx) * (1 - wy); val w01 = wx * (1 - wy); val w10 = (1 - wx) * wy; val w11 = wx * wy
                val o = row + x
                for (ch in 0 until 3) {
                    val shift = 16 - 8 * ch  // R, G, B
                    val v = ((a shr shift) and 0xFF) * w00 + ((b shr shift) and 0xFF) * w01 +
                        ((c shr shift) and 0xFF) * w10 + ((d shr shift) and 0xFF) * w11
                    t[ch * n + o] = Math.round(v).toFloat() / 255f  // cv2 returns uint8
                }
            }
        }
        return t to lb
    }
}
