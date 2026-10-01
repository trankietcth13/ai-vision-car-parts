package com.enginebay.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import com.enginebay.vision.core.Detection
import com.enginebay.vision.core.Letterbox
import com.enginebay.vision.core.Postprocess
import com.enginebay.vision.core.Preprocess
import org.json.JSONObject
import java.io.Closeable
import java.nio.FloatBuffer

/** assets/config.json, written by scripts/deployment/prepare_android_app.py (same as the browser demo). */
class ModelConfig(json: JSONObject) {
    val model: String = json.getString("model")
    val modelName: String = json.getString("model_name")
    val modelMd5: String = json.getString("model_md5")
    val release: String = json.getString("release")
    val imgsz: Int = json.getInt("imgsz")
    val iou: Float = json.getDouble("iou").toFloat()
    val maxDet: Int = json.getInt("max_det")
    val defaultConf: Float = json.getDouble("default_conf").toFloat()
    val maxSide: Int = json.optInt("max_side", 1600)
    val names: List<String> = json.getJSONArray("names").let { a -> List(a.length()) { a.getString(it) } }
    val namesVi: List<String> = json.getJSONArray("names_vi").let { a -> List(a.length()) { a.getString(it) } }
    val colors: List<Int> = json.getJSONArray("colors").let { a -> List(a.length()) { Color.parseColor(a.getString(it)) } }
    val thresholds: Map<String, Float> = json.getJSONObject("thresholds").let { o -> o.keys().asSequence().associateWith { o.getDouble(it).toFloat() } }

    companion object {
        fun load(context: Context) = ModelConfig(JSONObject(context.assets.open("config.json").bufferedReader().use { it.readText() }))
    }
}

/** A detection mapped back to the analysed photo. */
class Part(val det: Detection, val name: String, val nameVi: String, val color: Int, val rect: RectF) {
    val score get() = det.score
    val cls get() = det.cls
}

class DetectionResult(
    val parts: List<Part>,
    /** model-input sized (imgsz x imgsz) ARGB overlay of the masks, and the per-pixel index into [parts] */
    val overlay: Bitmap,
    val maskIndex: IntArray,
    val letterbox: Letterbox,
    val inferenceMs: Long,
    val totalMs: Long,
)

/** YOLO11-seg student on ONNX Runtime (CPU, 4 threads). Not thread-safe: call [detect] from one worker thread. */
class Detector(context: Context) : Closeable {
    val config = ModelConfig.load(context)
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val bytes = context.assets.open(config.model).use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
    }

    /**
     * [photo] must already be upright and at most [ModelConfig.maxSide] px (see [PhotoLoader]).
     * [conf]: slider threshold; with [perClass] the calibrated per-class thresholds replace it for listed classes.
     */
    fun detect(photo: Bitmap, conf: Float, perClass: Boolean): DetectionResult {
        val t0 = System.nanoTime()
        val size = config.imgsz
        val px = IntArray(photo.width * photo.height)
        photo.getPixels(px, 0, photo.width, 0, 0, photo.width, photo.height)
        val (tensor, lb) = Preprocess.letterbox(px, photo.width, photo.height, size)
        val useThr = perClass && config.thresholds.isNotEmpty()
        val predConf = if (useThr) minOf(conf, config.thresholds.values.min()) else conf

        val t1 = System.nanoTime()
        val out0: FloatArray
        val out1: FloatArray
        val protoShape: LongArray
        OnnxTensor.createTensor(env, FloatBuffer.wrap(tensor), longArrayOf(1, 3, size.toLong(), size.toLong())).use { input ->
            session.run(mapOf(session.inputNames.first() to input)).use { res ->
                val o0 = res.get(0) as OnnxTensor
                val o1 = res.get(1) as OnnxTensor
                out0 = o0.floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
                out1 = o1.floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
                protoShape = o1.info.shape
            }
        }
        val inferenceMs = (System.nanoTime() - t1) / 1_000_000

        val anchors = out0.size / (4 + config.names.size + protoShape[1].toInt())
        val dets = Postprocess.decode(out0, anchors, config.names.size, predConf, config.iou, config.maxDet)
            .filter { it.score >= if (useThr) config.thresholds[config.names[it.cls]] ?: conf else conf }
        val index = Postprocess.maskIndex(dets, out1, protoShape[1].toInt(), protoShape[2].toInt(), protoShape[3].toInt(), size)
        val colors = IntArray(size * size) { i ->
            val d = index[i]
            if (d < 0) 0 else (config.colors[dets[d].cls] and 0x00FFFFFF) or (0x70 shl 24)
        }
        val overlay = Bitmap.createBitmap(colors, size, size, Bitmap.Config.ARGB_8888)
        val parts = dets.map { d ->
            val r = RectF(
                ((d.box[0] - lb.left) / lb.scale).coerceIn(0f, photo.width.toFloat()),
                ((d.box[1] - lb.top) / lb.scale).coerceIn(0f, photo.height.toFloat()),
                ((d.box[2] - lb.left) / lb.scale).coerceIn(0f, photo.width.toFloat()),
                ((d.box[3] - lb.top) / lb.scale).coerceIn(0f, photo.height.toFloat()),
            )
            Part(d, config.names[d.cls], config.namesVi[d.cls], config.colors[d.cls], r)
        }
        return DetectionResult(parts, overlay, index, lb, inferenceMs, (System.nanoTime() - t0) / 1_000_000)
    }

    override fun close() {
        session.close()
    }
}
