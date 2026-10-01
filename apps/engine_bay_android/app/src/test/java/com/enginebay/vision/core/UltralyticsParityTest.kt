package com.enginebay.vision.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The app's pre/post-processing against Ultralytics on a real photo. Fixtures come from
 * scripts/deployment/prepare_android_app.py (decoded pixels, Ultralytics LetterBox output, raw ONNX outputs and
 * Ultralytics NMS + process_mask); the tests are skipped when they have not been generated.
 */
class UltralyticsParityTest {
    private val dir = listOf("src/test/resources/fixtures", "app/src/test/resources/fixtures").map(::File).firstOrNull { it.isDirectory }
    private fun bytes(name: String) = File(dir, name).readBytes()
    private fun floats(name: String): FloatArray {
        val bb = ByteBuffer.wrap(bytes(name)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(bb.remaining()).also { bb.get(it) }
    }

    private val ref: Map<String, Any?> by lazy { MiniJson.parse(File(dir, "reference.json").readText()) as Map<String, Any?> }

    @Test
    fun letterboxMatchesCv2() {
        assumeTrue("fixtures not generated", dir != null)
        val w = (ref["width"] as Number).toInt(); val h = (ref["height"] as Number).toInt()
        val rgb = bytes("image_rgb.bin")
        val argb = IntArray(w * h) { i ->
            (0xFF shl 24) or ((rgb[3 * i].toInt() and 0xFF) shl 16) or ((rgb[3 * i + 1].toInt() and 0xFF) shl 8) or (rgb[3 * i + 2].toInt() and 0xFF)
        }
        val (t, lb) = Preprocess.letterbox(argb, w, h, 640)
        val expected = bytes("letterbox_rgb.bin")  // 640 x 640 x 3, HWC
        val n = 640 * 640
        var maxDiff = 0; var off = 0
        for (i in 0 until n) for (ch in 0 until 3) {
            val mine = Math.round(t[ch * n + i] * 255f)
            val d = kotlin.math.abs(mine - (expected[3 * i + ch].toInt() and 0xFF))
            if (d > maxDiff) maxDiff = d
            if (d > 1) off++
        }
        println("letterbox $lb: max |diff| $maxDiff, values off by > 1: $off of ${3 * n}")
        assertTrue("cv2 uses fixed-point weights: allow 1/255", maxDiff <= 1)
    }

    @Test
    fun postprocessMatchesUltralytics() {
        assumeTrue("fixtures not generated", dir != null)
        val out0 = floats("output0.bin"); val out1 = floats("output1.bin")
        val nc = (ref["nc"] as Number).toInt()
        val dets = Postprocess.decode(out0, 8400, nc, (ref["conf"] as Number).toFloat(), (ref["iou"] as Number).toFloat(), 300)
        @Suppress("UNCHECKED_CAST") val expected = ref["detections"] as List<Map<String, Any?>>
        assertEquals("number of detections", expected.size, dets.size)
        val index = Postprocess.maskIndex(dets, out1, 32, 160, 160, 640)
        val masks = bytes("masks.bin"); val per = 640 * 640 / 8
        for ((i, e) in expected.withIndex()) {
            val d = dets[i]
            assertEquals("class of #$i", (e["cls"] as Number).toInt(), d.cls)
            assertEquals("score of #$i", (e["score"] as Number).toFloat(), d.score, 1e-5f)
            @Suppress("UNCHECKED_CAST") val box = (e["box"] as List<Number>).map { it.toFloat() }
            for (k in 0 until 4) assertEquals("box[$k] of #$i", box[k], d.box[k], 1e-3f)
            // own mask (before overlap resolution) vs Ultralytics' mask
            val own = Postprocess.maskIndex(listOf(d), out1, 32, 160, 160, 640)
            var inter = 0; var union = 0
            for (p in 0 until 640 * 640) {
                val r = (masks[i * per + p / 8].toInt() shr (7 - p % 8)) and 1 == 1
                val m = own[p] == 0
                if (r && m) inter++
                if (r || m) union++
            }
            val iou = if (union == 0) 1.0 else inter.toDouble() / union
            println("#$i cls ${d.cls} score ${"%.3f".format(d.score)} mask IoU vs Ultralytics ${"%.4f".format(iou)}")
            assertTrue("mask IoU of #$i = $iou", iou >= 0.99)
        }
        assertTrue(index.any { it >= 0 })
    }
}

/** Minimal JSON reader for the fixture metadata (org.json is only a stub in JVM unit tests). */
object MiniJson {
    fun parse(s: String): Any? = Parser(s).value()
    private class Parser(val s: String) {
        var i = 0
        fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
        fun value(): Any? {
            ws()
            return when (s[i]) {
                '{' -> obj(); '[' -> arr(); '"' -> str()
                't' -> { i += 4; true }; 'f' -> { i += 5; false }; 'n' -> { i += 4; null }
                else -> num()
            }
        }
        fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>(); i++; ws()
            if (s[i] == '}') { i++; return m }
            while (true) { ws(); val k = str(); ws(); i++; m[k] = value(); ws(); if (s[i++] == '}') return m }
        }
        fun arr(): List<Any?> {
            val l = ArrayList<Any?>(); i++; ws()
            if (s[i] == ']') { i++; return l }
            while (true) { l.add(value()); ws(); if (s[i++] == ']') return l }
        }
        fun str(): String {
            val b = StringBuilder(); i++
            while (s[i] != '"') { if (s[i] == '\\') { i++; b.append(s[i]) } else b.append(s[i]); i++ }
            i++; return b.toString()
        }
        fun num(): Double {
            val st = i
            while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
            return s.substring(st, i).toDouble()
        }
    }
}
