package com.enginebay.vision

import android.content.ActivityNotFoundException
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.slider.Slider
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val worker = Executors.newSingleThreadExecutor()
    private var detector: Detector? = null
    private var components: Map<String, ComponentInfo> = emptyMap()
    private var photo: Bitmap? = null
    private var result: DetectionResult? = null
    private var pendingCapture: Uri? = null
    private var runId = 0

    private lateinit var resultView: ResultView
    private lateinit var status: TextView
    private lateinit var progress: LinearProgressIndicator
    private lateinit var perClass: MaterialSwitch
    private lateinit var conf: Slider
    private lateinit var parts: LinearLayout

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val uri = pendingCapture
        if (ok && uri != null) open { PhotoLoader.fromUri(this, uri, maxSide()) }
    }
    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) open { PhotoLoader.fromUri(this, uri, maxSide()) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val root = findViewById<View>(R.id.root)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->  // edge-to-edge (targetSdk 35+)
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        resultView = findViewById(R.id.result)
        status = findViewById(R.id.status)
        progress = findViewById(R.id.progress)
        perClass = findViewById(R.id.perClass)
        conf = findViewById(R.id.conf)
        parts = findViewById(R.id.parts)
        val confValue = findViewById<TextView>(R.id.confValue)
        val original = findViewById<MaterialButton>(R.id.original)

        resultView.onPartTapped = ::showInfo
        original.addOnCheckedChangeListener { _, checked -> resultView.showOriginal = checked }
        conf.addOnChangeListener { _, v, _ -> confValue.text = String.format(Locale.US, "%.2f", v) }
        conf.addOnSliderTouchListener(object : Slider.OnSliderTouchListener {
            override fun onStartTrackingTouch(slider: Slider) = Unit
            override fun onStopTrackingTouch(slider: Slider) = analyze()
        })
        perClass.setOnCheckedChangeListener { _, _ -> analyze() }
        findViewById<MaterialButton>(R.id.capture).setOnClickListener { capture() }
        findViewById<MaterialButton>(R.id.gallery).setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        findViewById<MaterialButton>(R.id.samples).setOnClickListener { chooseSample() }

        worker.execute {  // model load takes a moment; keep the UI responsive
            try {
                val d = Detector(this)
                val info = ComponentInfo.loadAll(this)
                runOnUiThread {
                    detector = d
                    components = info
                    conf.value = d.config.defaultConf
                    perClass.isEnabled = d.config.thresholds.isNotEmpty()
                    perClass.isChecked = d.config.thresholds.isNotEmpty()
                    findViewById<TextView>(R.id.footer).text =
                        "Model ${d.config.modelName} (${d.config.release}, yolo11n-seg, ${d.config.names.size} loại) · md5 ${d.config.modelMd5.take(8)} · ONNX Runtime"
                    listOf(R.id.capture, R.id.gallery, R.id.samples).forEach { findViewById<View>(it).isEnabled = true }
                    progress.visibility = View.GONE
                    status.setText(R.string.ready)
                    intent.getStringExtra("sample")?.let { s -> open { PhotoLoader.fromAsset(this, "examples/$s", maxSide()) } }
                }
            } catch (e: Exception) {
                Log.e(TAG, "model load failed", e)
                runOnUiThread { progress.visibility = View.GONE; status.text = "Không tải được model: ${e.message}" }
            }
        }
    }

    private fun maxSide() = detector?.config?.maxSide ?: 1600

    private fun capture() {
        val dir = File(cacheDir, "captures").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }  // keep only the latest capture
        val file = File(dir, "photo_${System.currentTimeMillis()}.jpg")
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        pendingCapture = uri
        try {
            takePicture.launch(uri)
        } catch (e: ActivityNotFoundException) {
            status.setText(R.string.no_camera)
        }
    }

    private fun chooseSample() {
        val names = assets.list("examples")?.filter { it.endsWith(".jpg") }?.sorted() ?: return
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.samples)
            .setItems(names.map { it.removeSuffix(".jpg").replace('_', ' ') }.toTypedArray()) { _, i ->
                open { PhotoLoader.fromAsset(this, "examples/${names[i]}", maxSide()) }
            }.show()
    }

    /** Decode on the worker thread, then analyse. */
    private fun open(load: () -> Bitmap) {
        progress.visibility = View.VISIBLE
        status.setText(R.string.analyzing)
        worker.execute {
            val bmp = try { load() } catch (e: Exception) {
                Log.e(TAG, "decode failed", e)
                runOnUiThread { progress.visibility = View.GONE; status.setText(R.string.read_error) }
                return@execute
            }
            runOnUiThread {
                photo?.takeIf { it !== bmp }?.recycle()
                photo = bmp
                result = null
                findViewById<View>(R.id.empty).visibility = View.GONE
                resultView.visibility = View.VISIBLE
                resultView.show(bmp, null)
                analyze()
            }
        }
    }

    private fun analyze() {
        val d = detector ?: return
        val bmp = photo ?: return
        val id = ++runId
        val c = conf.value
        val pc = perClass.isChecked
        progress.visibility = View.VISIBLE
        status.setText(R.string.analyzing)
        worker.execute {
            val r = try { d.detect(bmp, c, pc) } catch (e: Exception) {
                Log.e(TAG, "inference failed", e)
                runOnUiThread { progress.visibility = View.GONE; status.text = "Lỗi khi phân tích: ${e.message}" }
                return@execute
            }
            Log.i(TAG, "detections ${r.parts.size} inference ${r.inferenceMs} ms total ${r.totalMs} ms: " +
                r.parts.joinToString { "${it.name}:${"%.2f".format(Locale.US, it.score)}" })
            runOnUiThread {
                if (id != runId) return@runOnUiThread  // a newer request is pending
                result = r
                progress.visibility = View.GONE
                resultView.show(bmp, r)
                findViewById<MaterialButton>(R.id.original).isEnabled = true
                status.text = if (r.parts.isEmpty()) "Không tìm thấy linh kiện ở ngưỡng này · ${r.totalMs} ms"
                else "${r.parts.size} linh kiện · ${r.totalMs} ms"
                renderParts(r)
            }
        }
    }

    private fun renderParts(r: DetectionResult) {
        parts.removeAllViews()
        val groups = r.parts.groupBy { it.cls }.values.sortedByDescending { g -> g.maxOf { it.score } }
        findViewById<View>(R.id.partsTitle).visibility = if (groups.isEmpty()) View.GONE else View.VISIBLE
        findViewById<View>(R.id.tapHint).visibility = if (groups.isEmpty()) View.GONE else View.VISIBLE
        val inflater = LayoutInflater.from(this)
        for (g in groups) {
            val best = g.maxBy { it.score }
            val row = inflater.inflate(R.layout.item_part, parts, false)
            row.findViewById<View>(R.id.dot).setBackgroundColor(best.color)
            row.findViewById<TextView>(R.id.name).text = best.nameVi + if (g.size > 1) " ×${g.size}" else ""
            val pct = (best.score * 100).toInt()
            row.findViewById<LinearProgressIndicator>(R.id.meter).progress = pct
            row.findViewById<TextView>(R.id.pct).text = "$pct%"
            row.setOnClickListener { showInfo(best) }
            parts.addView(row)
        }
    }

    private fun showInfo(part: Part) {
        val info = components[part.name]
        val view = layoutInflater.inflate(R.layout.sheet_component, null)
        view.findViewById<TextView>(R.id.title).text = info?.nameVi?.ifBlank { null } ?: part.nameVi
        view.findViewById<TextView>(R.id.subtitle).text = info?.nameEn?.ifBlank { null } ?: part.name
        val same = result?.parts?.count { it.cls == part.cls } ?: 1
        view.findViewById<TextView>(R.id.detection).text =
            "Độ tin cậy ${(part.score * 100).toInt()}%" + if (same > 1) " · $same vị trí trên ảnh" else ""
        view.findViewById<View>(R.id.draft).visibility = if (info?.draft == true) View.VISIBLE else View.GONE
        val sections = view.findViewById<LinearLayout>(R.id.sections)
        fun section(title: Int, body: String) {
            if (body.isBlank()) return
            sections.addView(TextView(this).apply {
                setText(title)
                setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_TitleSmall)
                setPadding(0, (16 * resources.displayMetrics.density).toInt(), 0, (4 * resources.displayMetrics.density).toInt())
            })
            sections.addView(TextView(this).apply {
                text = body
                setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium)
            })
        }
        fun bullets(items: List<String>) = items.joinToString("\n") { "• $it" }
        if (info == null || info.isEmpty) {
            section(R.string.sec_summary, getString(R.string.info_missing))
        } else {
            section(R.string.sec_summary, info.summary)
            section(R.string.sec_function, info.function)
            section(R.string.sec_location, info.locationHint)
            section(R.string.sec_checks, bullets(info.inspectionChecks))
            section(R.string.sec_symptoms, bullets(info.commonSymptoms))
            section(R.string.sec_dtcs, info.relatedDtcs.joinToString("\n") { (code, meaning) -> "• $code: $meaning" })
            section(R.string.sec_safety, bullets(info.safetyNotes))
        }
        BottomSheetDialog(this).apply {
            setContentView(view)
            behavior.skipCollapsed = true  // landscape tablets: the collapsed peek hides almost everything
            behavior.state = com.google.android.material.bottomsheet.BottomSheetBehavior.STATE_EXPANDED
        }.show()
    }

    override fun onDestroy() {
        worker.execute { detector?.close() }
        worker.shutdown()
        super.onDestroy()
    }

    companion object {
        const val TAG = "EngineBay"
    }
}
