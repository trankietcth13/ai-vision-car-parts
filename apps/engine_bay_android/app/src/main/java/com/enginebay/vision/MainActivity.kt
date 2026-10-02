package com.enginebay.vision

import android.content.ActivityNotFoundException
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.util.Log
import android.util.TypedValue
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.os.LocaleListCompat
import com.google.android.material.appbar.MaterialToolbar
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.materialswitch.MaterialSwitch
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import com.enginebay.vision.core.Diagnosis
import com.enginebay.vision.core.DtcLookup
import com.enginebay.vision.core.DtcTable
import com.enginebay.vision.core.SuspectStatus
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private val worker get() = AppState.worker
    // kept in AppState: switching the language recreates the activity, and nothing should be reloaded or lost
    private var detector: Detector? by AppState::detector
    private var components: Map<String, ComponentInfo> by AppState::components
    private var photo: Bitmap? by AppState::photo
    private var result: DetectionResult? by AppState::result
    private var dtcTable: DtcTable? by AppState::dtcTable
    private var diagnosis: Diagnosis? by AppState::diagnosis
    private var dtcText: String by AppState::dtcText
    private var pendingCapture: Uri? = null
    /** the 3D view of the open component sheet, paused and resumed with the activity */
    private var liveModelView: ModelView? = null
    private var runId = 0
    private val inspection = InspectionWorkflow(this, ::locateComponent, ::resetInspection)

    private lateinit var resultView: ResultView
    private lateinit var status: TextView
    private lateinit var progress: LinearProgressIndicator
    private lateinit var perClass: MaterialSwitch
    private lateinit var conf: Slider
    private lateinit var parts: LinearLayout
    private var selectedPartClass: String? = null
    private val partCardViews = mutableMapOf<String, com.google.android.material.card.MaterialCardView>()

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val uri = pendingCapture
        if (ok && uri != null) onPhotoCaptured(uri)
    }
    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) onPhotoPicked(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (AppCompatDelegate.getApplicationLocales().isEmpty) {  // first start: Vietnamese, switchable to English
            AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags("vi"))
            return  // this instance is recreated in Vietnamese right away; the new one does the work
        }
        setContentView(R.layout.activity_main)
        pendingCapture = savedInstanceState?.getString("engineCapture")?.let(Uri::parse)
        inspection.restoreExport(savedInstanceState)
        findViewById<MaterialToolbar>(R.id.toolbar).apply {
            inflateMenu(R.menu.main)
            menu.findItem(R.id.action_language).isEnabled = false  // until the model is ready (no half-loaded switch)
            setOnMenuItemClickListener { item ->
                if (item.itemId == R.id.action_models) {
                    showModelLibrary()
                    true
                } else if (item.itemId == R.id.action_language) {
                    AppState.conf = conf.value
                    AppState.perClass = perClass.isChecked
                    AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(if (isVi()) "en" else "vi"))
                    true
                } else false
            }
        }
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
        val rotate = findViewById<MaterialButton>(R.id.rotate)
        val resultZoomBadgeContainer = findViewById<View>(R.id.resultZoomBadgeContainer)
        val resultZoomBadge = findViewById<TextView>(R.id.resultZoomBadge)
        val btnMainZoomIn = findViewById<MaterialButton>(R.id.btnMainZoomIn)
        val btnMainZoomOut = findViewById<MaterialButton>(R.id.btnMainZoomOut)
        val btnMainZoomReset = findViewById<MaterialButton>(R.id.btnMainZoomReset)

        resultView.onPartTapped = ::showInfo
        resultView.onScaleChanged = { scale ->
            resultZoomBadge.text = String.format(Locale.US, "%.1fx", scale)
            resultZoomBadgeContainer.visibility = if (scale > 1.05f) View.VISIBLE else View.GONE
            if (scale <= 1.05f && selectedPartClass != null) {
                selectedPartClass = null
                updateHighlight()
                updatePartsSelection()
            }
        }
        btnMainZoomIn.setOnClickListener { resultView.zoomIn() }
        btnMainZoomOut.setOnClickListener { resultView.zoomOut() }
        btnMainZoomReset.setOnClickListener {
            selectedPartClass = null
            resultView.resetZoom()
            updateHighlight()
            updatePartsSelection()
        }

        original.addOnCheckedChangeListener { _, checked -> resultView.showOriginal = checked }
        rotate.setOnClickListener { rotateCurrentPhoto(90f) }
        conf.addOnChangeListener { _, v, _ -> confValue.text = String.format(Locale.US, "%.2f", v); AppState.conf = v }
        conf.addOnSliderTouchListener(object : Slider.OnSliderTouchListener {
            override fun onStartTrackingTouch(slider: Slider) = Unit
            override fun onStopTrackingTouch(slider: Slider) { setConf(slider.value); analyze() }  // a tap can land off-step
        })
        perClass.setOnCheckedChangeListener { _, checked -> AppState.perClass = checked; analyze() }
        findViewById<MaterialButton>(R.id.capture).setOnClickListener { capture() }
        findViewById<MaterialButton>(R.id.gallery).setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        findViewById<MaterialButton>(R.id.samples).setOnClickListener { chooseSample() }
        findViewById<MaterialButton>(R.id.dtc).setOnClickListener { askCodes() }
        inspection.bind()

        val firstStart = !AppState.extrasConsumed  // adb test extras (sample, dtc) are applied once per process
        AppState.extrasConsumed = true
        val loaded = detector
        if (loaded != null) {  // recreated (language switch or orientation change): reuse everything
            onModelReady(loaded, firstStart)
            findViewById<View>(R.id.dtc).isEnabled = dtcTable != null
            inspection.refresh()
            photo?.let { bmp ->
                findViewById<View>(R.id.empty).visibility = View.GONE
                resultView.visibility = View.VISIBLE
                updateHighlight()
                resultView.show(bmp, result)
                findViewById<MaterialButton>(R.id.rotate).isEnabled = true
                setZoomButtonsEnabled(true)
                val r = result
                if (r != null) showResult(r) else analyze()  // the switch interrupted an analysis: redo it
            }
            renderDiagnosis()
            return
        }
        worker.execute {  // model load takes a moment; keep the UI responsive
            val restored = if (!AppState.inspectionLoaded) inspection.loadDraft() else null
            val table = DiagnosisTable.load(this)  // small; the error-code lookup works even if the model fails
            dtcTable = table
            runOnUiThread {
                if (isDestroyed) return@runOnUiThread
                restored?.first?.let { draft ->
                    AppState.inspection = draft
                    photo = restored.second
                    dtcText = draft.codes
                    AppState.conf = draft.threshold
                    AppState.perClass = draft.perClass
                }
                AppState.inspectionLoaded = true
                inspection.refresh()
                findViewById<View>(R.id.dtc).isEnabled = table != null
                if (dtcText.isNotBlank()) applyCodes(dtcText)
                if (firstStart) intent.getStringExtra("dtc")?.let(::applyCodes)  // adb: --es dtc "P0301 P0171"
                if (firstStart) intent.getStringExtra("model")?.let { show3D(it) }  // adb: --es model spark_plug
            }
            try {
                val d = Detector(this)
                val info = ComponentInfo.loadAll(this)
                detector = d
                components = info
                runOnUiThread {
                    if (!isDestroyed) {
                        onModelReady(d, firstStart)
                        // Also restore after Back -> relaunch in the same process, when test extras were already consumed.
                        if (!(firstStart && intent.getStringExtra("sample") != null)) photo?.let { bmp ->
                            findViewById<View>(R.id.empty).visibility = View.GONE
                            resultView.visibility = View.VISIBLE
                            updateHighlight()
                            resultView.show(bmp, null)
                            findViewById<MaterialButton>(R.id.rotate).isEnabled = true
                            setZoomButtonsEnabled(true)
                            analyze()
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "model load failed", e)
                runOnUiThread {
                    if (!isDestroyed) { progress.visibility = View.GONE; status.text = getString(R.string.model_load_failed, e.message) }
                }
            }
        }
    }

    private fun setZoomButtonsEnabled(enabled: Boolean) {
        findViewById<MaterialButton>(R.id.btnMainZoomIn)?.isEnabled = enabled
        findViewById<MaterialButton>(R.id.btnMainZoomOut)?.isEnabled = enabled
        findViewById<MaterialButton>(R.id.btnMainZoomReset)?.isEnabled = enabled
    }

    /**
     * Sets the threshold snapped onto the slider's 0.05 grid. A tap can report an off-step value (0.37), and saved
     * drafts may hold one (0.6875); Slider throws while drawing such a value after the activity is recreated.
     */
    private fun setConf(v: Float) {
        val steps = Math.round((v.coerceIn(conf.valueFrom, conf.valueTo) - conf.valueFrom) / conf.stepSize)
        conf.value = (conf.valueFrom + steps * conf.stepSize).coerceAtMost(conf.valueTo)
        AppState.conf = conf.value
    }

    /** Model loaded (or reused after a language switch): enable the controls, texts in the current language. */
    private fun onModelReady(d: Detector, firstStart: Boolean) {
        setConf(AppState.conf ?: d.config.defaultConf)
        perClass.isEnabled = d.config.thresholds.isNotEmpty()
        perClass.isChecked = AppState.perClass ?: d.config.thresholds.isNotEmpty()
        findViewById<TextView>(R.id.footer).text =
            getString(R.string.footer, d.config.modelName, d.config.release, d.config.names.size, d.config.modelMd5.take(8))
        listOf(R.id.capture, R.id.gallery, R.id.samples).forEach { findViewById<View>(it).isEnabled = true }
        findViewById<MaterialButton>(R.id.rotate).isEnabled = photo != null
        setZoomButtonsEnabled(photo != null)
        findViewById<MaterialToolbar>(R.id.toolbar).menu.findItem(R.id.action_language).isEnabled = true
        progress.visibility = View.GONE
        status.setText(R.string.ready)
        if (firstStart) intent.getStringExtra("sample")?.let { s -> open { PhotoLoader.fromAsset(this, "examples/$s", maxSide()) } }
    }

    /** Vietnamese UI and Vietnamese data texts, or English. */
    private fun isVi() = lang() == "vi"

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
            .setItems(names.indices.map { getString(R.string.sample_n, it + 1) }.toTypedArray()) { _, i ->
                open { PhotoLoader.fromAsset(this, "examples/${names[i]}", maxSide()) }
            }.show()
    }

    /** Decode on the worker thread, then analyse. */
    private fun open(load: () -> Bitmap) {
        if (AppState.inspection.checks.any { it.outcome != com.enginebay.vision.core.CheckOutcome.PENDING || it.note.isNotBlank() }) {
            MaterialAlertDialogBuilder(this).setMessage(R.string.inspection_replace_photo)
                .setPositiveButton(R.string.inspection_replace) { _, _ -> loadPhoto(load) }
                .setNegativeButton(R.string.cancel, null).show()
        } else loadPhoto(load)
    }

    private fun commitPhoto(bmp: Bitmap) {
        photo = bmp
        result = null
        inspection.photoChanged(bmp)
        findViewById<View>(R.id.empty).visibility = View.GONE
        resultView.visibility = View.VISIBLE
        findViewById<View>(R.id.resultZoomBadgeContainer).visibility = View.GONE
        resultView.show(bmp, null)
        findViewById<MaterialButton>(R.id.rotate).isEnabled = true
        setZoomButtonsEnabled(true)
    }

    private fun loadPhoto(load: () -> Bitmap) {
        val request = ++runId
        progress.visibility = View.VISIBLE
        status.setText(R.string.analyzing)
        worker.execute {
            val bmp = try { load() } catch (e: Exception) {
                Log.e(TAG, "decode failed", e)
                runOnUiThread { if (!isDestroyed && request == runId) { progress.visibility = View.GONE; status.setText(R.string.read_error) } }
                return@execute
            }
            runOnUiThread {
                if (isDestroyed || request != runId) return@runOnUiThread
                commitPhoto(bmp)
                analyze()
            }
        }
    }

    private fun onPhotoCaptured(uri: Uri) {
        if (AppState.inspection.checks.any { it.outcome != com.enginebay.vision.core.CheckOutcome.PENDING || it.note.isNotBlank() }) {
            MaterialAlertDialogBuilder(this).setMessage(R.string.inspection_replace_photo)
                .setPositiveButton(R.string.inspection_replace) { _, _ -> prepareCapturedPhoto(uri) }
                .setNegativeButton(R.string.cancel, null).show()
        } else prepareCapturedPhoto(uri)
    }

    private fun prepareCapturedPhoto(uri: Uri) {
        val request = ++runId
        progress.visibility = View.VISIBLE
        status.setText(R.string.analyzing)
        worker.execute {
            val bmp = try { PhotoLoader.fromUri(this, uri, maxSide()) } catch (e: Exception) {
                Log.e(TAG, "decode failed", e)
                runOnUiThread {
                    if (!isDestroyed && request == runId) {
                        progress.visibility = View.GONE
                        status.setText(R.string.read_error)
                    }
                }
                return@execute
            }
            runOnUiThread {
                if (isDestroyed || request != runId) return@runOnUiThread
                progress.visibility = View.GONE
                status.setText(R.string.ready)
                showPhotoRotateDialog(bmp, isPreAnalysis = true, onRetake = { capture() })
            }
        }
    }

    private fun onPhotoPicked(uri: Uri) {
        if (AppState.inspection.checks.any { it.outcome != com.enginebay.vision.core.CheckOutcome.PENDING || it.note.isNotBlank() }) {
            MaterialAlertDialogBuilder(this).setMessage(R.string.inspection_replace_photo)
                .setPositiveButton(R.string.inspection_replace) { _, _ -> preparePickedPhoto(uri) }
                .setNegativeButton(R.string.cancel, null).show()
        } else preparePickedPhoto(uri)
    }

    private fun preparePickedPhoto(uri: Uri) {
        val request = ++runId
        progress.visibility = View.VISIBLE
        status.setText(R.string.analyzing)
        worker.execute {
            val bmp = try { PhotoLoader.fromUri(this, uri, maxSide()) } catch (e: Exception) {
                Log.e(TAG, "decode failed", e)
                runOnUiThread {
                    if (!isDestroyed && request == runId) {
                        progress.visibility = View.GONE
                        status.setText(R.string.read_error)
                    }
                }
                return@execute
            }
            runOnUiThread {
                if (isDestroyed || request != runId) return@runOnUiThread
                progress.visibility = View.GONE
                status.setText(R.string.ready)
                showPhotoRotateDialog(bmp, isPreAnalysis = true, onRetake = {
                    pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                })
            }
        }
    }

    private fun showPhotoRotateDialog(
        baseBitmap: Bitmap,
        isPreAnalysis: Boolean = true,
        onRetake: (() -> Unit)? = { capture() }
    ) {
        val dialog = BottomSheetDialog(this).apply {
            behavior.skipCollapsed = true
            behavior.state = com.google.android.material.bottomsheet.BottomSheetBehavior.STATE_EXPANDED
        }
        val view = layoutInflater.inflate(R.layout.sheet_photo_preview, null)
        val img = view.findViewById<ZoomableImageView>(R.id.previewImage)
        val angleBadge = view.findViewById<TextView>(R.id.previewAngleBadge)
        val zoomBadge = view.findViewById<TextView>(R.id.previewZoomBadge)
        val btnZoomIn = view.findViewById<MaterialButton>(R.id.btnZoomIn)
        val btnZoomOut = view.findViewById<MaterialButton>(R.id.btnZoomOut)
        val btnZoomReset = view.findViewById<MaterialButton>(R.id.btnZoomReset)
        val btnLeft = view.findViewById<MaterialButton>(R.id.btnRotateLeft)
        val btnRight = view.findViewById<MaterialButton>(R.id.btnRotateRight)
        val btnRetake = view.findViewById<MaterialButton>(R.id.btnRetake)
        val btnConfirm = view.findViewById<MaterialButton>(R.id.btnConfirmAnalyze)
        val title = view.findViewById<TextView>(R.id.previewTitle)
        val subtitle = view.findViewById<TextView>(R.id.previewSubtitle)

        var currentAngle = 0
        var currentPreviewBitmap: Bitmap = baseBitmap
        var isConfirmed = false

        img.setImageBitmap(baseBitmap)
        angleBadge.text = "0°"
        zoomBadge.text = "1.0x"

        img.onScaleChanged = { scale ->
            zoomBadge.text = String.format(Locale.US, "%.1fx", scale)
        }
        btnZoomIn.setOnClickListener { img.zoomIn() }
        btnZoomOut.setOnClickListener { img.zoomOut() }
        btnZoomReset.setOnClickListener { img.resetZoom() }

        if (!isPreAnalysis) {
            title.setText(R.string.photo_review_title)
            subtitle.setText(R.string.zoom_hint)
        }

        if (onRetake == null) {
            btnRetake.visibility = View.GONE
        } else {
            btnRetake.visibility = View.VISIBLE
            btnRetake.setOnClickListener {
                dialog.dismiss()
                onRetake()
            }
        }

        fun applyAngle(newAngle: Int) {
            currentAngle = ((newAngle % 360) + 360) % 360
            angleBadge.text = "$currentAngle°"
            if (currentAngle == 0) {
                if (currentPreviewBitmap !== baseBitmap) currentPreviewBitmap.recycle()
                currentPreviewBitmap = baseBitmap
                img.setImageBitmap(baseBitmap)
            } else {
                val m = Matrix().apply { postRotate(currentAngle.toFloat()) }
                val rotated = Bitmap.createBitmap(baseBitmap, 0, 0, baseBitmap.width, baseBitmap.height, m, true)
                if (currentPreviewBitmap !== baseBitmap) currentPreviewBitmap.recycle()
                currentPreviewBitmap = rotated
                img.setImageBitmap(rotated)
            }
            zoomBadge.text = "1.0x"
        }

        btnLeft.setOnClickListener { applyAngle(currentAngle - 90) }
        btnRight.setOnClickListener { applyAngle(currentAngle + 90) }
        btnConfirm.setOnClickListener {
            isConfirmed = true
            dialog.dismiss()
            val finalBmp = if (currentAngle == 0) {
                baseBitmap
            } else {
                currentPreviewBitmap
            }
            commitPhoto(finalBmp)
            analyze()
        }
        dialog.setOnDismissListener {
            if (!isConfirmed) {
                if (currentPreviewBitmap !== baseBitmap) currentPreviewBitmap.recycle()
            }
        }
        dialog.setContentView(view)
        dialog.show()
    }

    private fun rotateCurrentPhoto(degrees: Float) {
        val current = photo ?: return
        val id = ++runId
        progress.visibility = View.VISIBLE
        status.setText(R.string.analyzing)
        worker.execute {
            val rotated = PhotoLoader.rotate(current, degrees)
            runOnUiThread {
                if (isDestroyed || id != runId) return@runOnUiThread
                commitPhoto(rotated)
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
        result = null // do not save stale detections while a new inference is pending
        progress.visibility = View.VISIBLE
        status.setText(R.string.analyzing)
        worker.execute {
            val r = try { d.detect(bmp, c, pc) } catch (e: Exception) {
                Log.e(TAG, "inference failed", e)
                runOnUiThread { if (!isDestroyed && id == runId) { progress.visibility = View.GONE; status.text = getString(R.string.analysis_failed, e.message) } }
                return@execute
            }
            Log.i(TAG, "detections ${r.parts.size} inference ${r.inferenceMs} ms total ${r.totalMs} ms: " +
                r.parts.joinToString { "${it.name}:${"%.2f".format(Locale.US, it.score)}" })
            runOnUiThread {
                if (isDestroyed || id != runId || photo !== bmp) return@runOnUiThread
                result = r
                progress.visibility = View.GONE
                resultView.show(bmp, r)
                showResult(r)
                renderDiagnosis()
                inspection.detected(r)
            }
        }
    }

    private fun showResult(r: DetectionResult) {
        selectedPartClass = null
        findViewById<MaterialButton>(R.id.original).isEnabled = true
        findViewById<MaterialButton>(R.id.rotate).isEnabled = true
        setZoomButtonsEnabled(true)
        status.text = if (r.parts.isEmpty()) getString(R.string.status_none, r.totalMs.toInt())
        else getString(R.string.status_found, r.parts.size, r.totalMs.toInt())
        renderParts(r)
    }

    private fun renderParts(r: DetectionResult) {
        parts.removeAllViews()
        partCardViews.clear()
        val targets = diagnosis?.detectorTargets.orEmpty()
        val groups = r.parts.groupBy { it.cls }.values
            .sortedWith(compareByDescending<List<Part>> { it[0].name in targets }.thenByDescending { g -> g.maxOf { it.score } })
        findViewById<View>(R.id.partsTitle).visibility = if (groups.isEmpty()) View.GONE else View.VISIBLE
        findViewById<View>(R.id.tapHint).visibility = if (groups.isEmpty()) View.GONE else View.VISIBLE
        val inflater = LayoutInflater.from(this)
        for (g in groups) {
            val best = g.maxBy { it.score }
            val row = inflater.inflate(R.layout.item_part, parts, false) as com.google.android.material.card.MaterialCardView
            partCardViews[best.name] = row

            row.findViewById<View>(R.id.dot).setBackgroundColor(best.color)
            val name = best.label(isVi())
            val label = if (g.size > 1) getString(R.string.count_suffix, name, g.size) else name
            row.findViewById<TextView>(R.id.name).text = if (best.name in targets) getString(R.string.dtc_part_tagged, label) else label
            val pct = (best.score * 100).toInt()
            row.findViewById<LinearProgressIndicator>(R.id.meter).progress = pct
            row.findViewById<TextView>(R.id.pct).text = getString(R.string.percent, pct)

            // Clicking the row zooms in to that component on the engine bay photo
            row.setOnClickListener {
                onComponentClicked(best.name, g)
            }

            // Clicking the chevron button opens the 3D model and diagnostic sheet
            row.findViewById<View>(R.id.btnDetails)?.setOnClickListener {
                showInfo(best)
            }

            parts.addView(row)
        }
        updatePartsSelection()
    }

    private fun onComponentClicked(className: String, group: List<Part>) {
        if (selectedPartClass == className) {
            // Already selected: toggle back to full engine bay view
            selectedPartClass = null
            resultView.resetZoom()
            updateHighlight()
        } else {
            selectedPartClass = className
            findViewById<MaterialButton>(R.id.original).isChecked = false
            updateHighlight()
            resultView.zoomToParts(group)
        }
        updatePartsSelection()
    }

    private fun updatePartsSelection() {
        val dp = resources.displayMetrics.density
        val defaultOutline = ContextCompat.getColor(this, R.color.card_stroke_light)
        val defaultSurface = ContextCompat.getColor(this, R.color.card_surface_light)
        val selectedSurface = ContextCompat.getColor(this, R.color.accent_blue_light)
        val selectedOutline = ContextCompat.getColor(this, R.color.accent_blue)

        for ((name, card) in partCardViews) {
            val isSelected = (name == selectedPartClass)
            if (isSelected) {
                card.strokeColor = selectedOutline
                card.strokeWidth = (2f * dp).toInt()
                card.setCardBackgroundColor(selectedSurface)
            } else {
                card.strokeColor = defaultOutline
                card.strokeWidth = (1f * dp).toInt()
                card.setCardBackgroundColor(defaultSurface)
            }
        }
    }

    private fun showInfo(part: Part) {
        val info = components[part.name]
        val vi = isVi()
        val same = result?.parts?.count { it.cls == part.cls } ?: 1
        val pct = (part.score * 100).toInt()
        // title in the UI language, the other language underneath (technicians look parts up in both)
        openComponentSheet(
            modelKey = part.name,
            title = info?.name?.get(vi)?.ifBlank { null } ?: part.label(vi),
            subtitle = info?.name?.get(!vi)?.ifBlank { null } ?: part.label(!vi),
            status = if (same > 1) getString(R.string.detection_line_multi, pct, same) else getString(R.string.detection_line, pct),
            info = info,
        )
    }

    /**
     * Sheet for a component of the knowledge table (or a detector class) opened from the error-code panel or the 3D
     * library: works without a photo and for components the model cannot detect, where the 3D model is the main help.
     */
    private fun show3D(key: String, codes: List<String> = emptyList()) {
        val vi = isVi()
        val lang = lang()
        val ref = dtcTable?.components?.get(key)
        val info = components[ref?.detectorClass ?: key]
        val detectable = ref?.detectorClass != null || (ref == null && detector?.config?.names?.contains(key) == true)
        val status = getString(if (detectable) R.string.model_status_detectable else R.string.model_status_not_detectable) +
            if (codes.isEmpty()) "" else "\n" + getString(R.string.model_status_codes, codes.joinToString(", "))
        val tableCodes = dtcTable?.entries.orEmpty().filter { key in it.components }
            .map { it.codes.joinToString(", ") to Bilingual(it.meaning.en, it.meaning.vi) }
        openComponentSheet(
            modelKey = key,
            title = ref?.name?.get(lang) ?: info?.name?.get(vi)?.ifBlank { null } ?: key,
            subtitle = ref?.name?.get(if (vi) "en" else "vi") ?: info?.name?.get(!vi).orEmpty(),
            status = status,
            info = info,
            fallbackSummary = ref?.description?.get(lang),
            fallbackCodes = tableCodes,
        )
    }

    /** All components with a 3D model, in knowledge-table order; tap one to open its sheet. */
    private fun showModelLibrary() {
        val vi = isVi()
        val lang = lang()
        val table = dtcTable?.components.orEmpty()
        val keys = (table.keys + ModelStore.available(this).sorted()).distinct().filter { ModelStore.has(this, it) }
        val labels = keys.map { k ->
            val ref = table[k]
            val name = ref?.name?.get(lang) ?: components[k]?.name?.get(vi)?.ifBlank { null } ?: k
            val detectable = ref?.detectorClass != null || (ref == null && detector?.config?.names?.contains(k) == true)
            if (detectable) getString(R.string.models_detectable, name) else name
        }
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.models_title)
            .setItems(labels.toTypedArray()) { _, i -> show3D(keys[i]) }
            .show()
    }

    /** Bottom sheet: names, status line, 3D model (when there is one) and the component information sections. */
    private fun openComponentSheet(
        modelKey: String?,
        title: String,
        subtitle: String,
        status: String,
        info: ComponentInfo?,
        fallbackSummary: String? = null,
        fallbackCodes: List<Pair<String, Bilingual>> = emptyList(),
    ) {
        val vi = isVi()
        val view = layoutInflater.inflate(R.layout.sheet_component, null)
        view.findViewById<TextView>(R.id.title).text = title
        view.findViewById<TextView>(R.id.subtitle).text = subtitle
        view.findViewById<TextView>(R.id.detection).text = status
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
        fun bullets(items: List<Bilingual>) = items.joinToString("\n") { getString(R.string.bullet, it.get(vi)) }
        fun codes(list: List<Pair<String, Bilingual>>) = list.joinToString("\n") { (code, meaning) -> getString(R.string.dtc_code_line, code, meaning.get(vi)) }
        if (info == null || info.isEmpty) {
            section(R.string.sec_summary, fallbackSummary ?: getString(R.string.info_missing))
            section(R.string.sec_dtcs, codes(fallbackCodes))
        } else {
            section(R.string.sec_summary, info.summaryText.get(vi).ifBlank { fallbackSummary.orEmpty() })
            section(R.string.sec_function, info.functionText.get(vi))
            section(R.string.sec_location, info.locationText.get(vi))
            section(R.string.sec_checks, bullets(info.checks))
            section(R.string.sec_symptoms, bullets(info.symptoms))
            section(R.string.sec_dtcs, codes(info.dtcs.ifEmpty { fallbackCodes }))
            section(R.string.sec_safety, bullets(info.safety))
        }
        val dialog = BottomSheetDialog(this).apply {
            setContentView(view)
            behavior.skipCollapsed = true  // landscape tablets: the collapsed peek hides almost everything
            behavior.state = com.google.android.material.bottomsheet.BottomSheetBehavior.STATE_EXPANDED
            setOnDismissListener { liveModelView = null }
        }
        if (modelKey != null) {
            val dp = resources.displayMetrics.density
            sections.addView(MaterialButton(this).apply {
                setText(R.string.inspection_checks)
                cornerRadius = (24 * dp).toInt()
                minHeight = (48 * dp).toInt()
                insetTop = 0
                insetBottom = 0
                setOnClickListener { dialog.dismiss(); inspection.checks(modelKey) }
            }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                topMargin = (16 * dp).toInt()
            })
        }
        dialog.show()
        attachModel(view.findViewById(R.id.model3d), modelKey, dialog)
    }

    /** Load the component's 3D model (if there is one) into [box] of the open sheet. */
    private fun attachModel(box: ViewGroup, key: String?, dialog: BottomSheetDialog) {
        if (key == null || !ModelStore.has(this, key)) return
        val vi = isVi()
        ModelStore.load(this, key) { scene ->
            runOnUiThread {
                if (isDestroyed || !dialog.isShowing) return@runOnUiThread
                if (scene == null) {
                    box.addView(TextView(this).apply { setText(R.string.model3d_load_failed) })
                    return@runOnUiThread
                }
                val panel = ModelPanel(box, scene, vi)
                box.addView(panel.root)
                liveModelView = panel.modelView
            }
        }
    }

    private fun askCodes() {
        if (dtcTable == null) return
        val input = TextInputEditText(this).apply {
            setText(dtcText)
            setSingleLine()
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS
        }
        val pad = (20 * resources.displayMetrics.density).toInt()
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad / 2, pad, 0)
            addView(TextView(this@MainActivity).apply { setText(R.string.dtc_help) })
            addView(TextInputLayout(this@MainActivity).apply { hint = getString(R.string.dtc_hint); addView(input) })
        }
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.dtc_title)
            .setView(box)
            .setPositiveButton(R.string.dtc_lookup) { _, _ -> applyCodes(input.text?.toString().orEmpty()) }
            .setNeutralButton(R.string.dtc_clear) { _, _ -> applyCodes("") }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    /** UI language for the bilingual data files: "vi" or anything else (English). */
    private fun lang(): String = resources.configuration.locales[0].language

    /** Look the codes up (offline), emphasise the target components in the photo and list what to inspect. */
    private fun applyCodes(text: String) {
        val table = dtcTable ?: return
        dtcText = text.trim()
        diagnosis = if (dtcText.isEmpty()) null else DtcLookup.diagnose(dtcText, table)
        Log.i(TAG, "dtc '$dtcText': matched ${diagnosis?.matches?.map { it.code }} unknown ${diagnosis?.unknown} " +
            "targets ${diagnosis?.detectorTargets}")
        updateHighlight()
        result?.let(::renderParts)
        renderDiagnosis()
        inspection.persist()
    }

    private fun updateHighlight() {
        resultView.highlight = selectedPartClass?.let { setOf(it) } ?: AppState.findTarget?.let { setOf(it) } ?: diagnosis?.detectorTargets
    }

    private fun locateComponent(key: String?) {
        AppState.findTarget = key
        selectedPartClass = key
        findViewById<MaterialButton>(R.id.original).isChecked = false
        updateHighlight()
        if (key != null) {
            val matching = result?.parts?.filter { it.name == key }.orEmpty()
            if (matching.isNotEmpty()) {
                resultView.zoomToParts(matching)
            }
        }
        updatePartsSelection()
    }

    private fun resetInspection() {
        ++runId
        photo = null; result = null; dtcText = ""; diagnosis = null
        selectedPartClass = null
        parts.removeAllViews()
        partCardViews.clear()
        listOf(R.id.result, R.id.partsTitle, R.id.tapHint, R.id.diagnosis, R.id.progress).forEach {
            findViewById<View>(it).visibility = View.GONE
        }
        findViewById<View>(R.id.empty).visibility = View.VISIBLE
        findViewById<MaterialButton>(R.id.original).apply { isChecked = false; isEnabled = false }
        status.setText(R.string.ready)
        updateHighlight()
    }

    /** The error-code panel: urgency, code meanings, components to inspect (found in the photo or not), safety notes. */
    private fun renderDiagnosis() {
        val panel = findViewById<LinearLayout>(R.id.diagnosis)
        panel.removeAllViews()
        val d = diagnosis
        panel.visibility = if (d == null) View.GONE else View.VISIBLE
        if (d == null) return
        val dp = resources.displayMetrics.density
        val titleStyle = com.google.android.material.R.style.TextAppearance_Material3_TitleSmall
        val bodyStyle = com.google.android.material.R.style.TextAppearance_Material3_BodyMedium
        val smallStyle = com.google.android.material.R.style.TextAppearance_Material3_BodySmall
        fun text(body: CharSequence, style: Int, top: Int = 0) = TextView(this).apply {
            text = body
            setTextAppearance(style)
            setPadding(0, (top * dp).toInt(), 0, 0)
        }
        fun TextView.boxed(danger: Boolean) = apply {
            setBackgroundResource(if (danger) R.drawable.bg_danger else R.drawable.bg_note)
            setTextColor(ContextCompat.getColor(this@MainActivity, if (danger) R.color.danger_fg else R.color.warn_fg))
            val p = (10 * dp).toInt()
            setPadding(p, p, p, p)
        }

        val lang = lang()
        panel.addView(text(getString(R.string.dtc_title_codes, (d.matches.map { it.code } + d.unknown).joinToString(", ")), titleStyle, 16))
        if (d.matches.isNotEmpty()) {
            val urgency = when (d.urgency) {
                "stop" -> R.string.dtc_urgency_stop
                "soon" -> R.string.dtc_urgency_soon
                else -> R.string.dtc_urgency_check
            }
            panel.addView(text(getString(urgency), bodyStyle, 8).boxed(d.urgency == "stop"))
        }
        for (m in d.matches) panel.addView(text(getString(R.string.dtc_code_line, m.code, m.entry.meaning.get(lang)), bodyStyle, 4))
        if (d.unknown.isNotEmpty()) panel.addView(text(getString(R.string.dtc_unknown, d.unknown.joinToString(", ")), smallStyle, 4))
        if (d.matches.isEmpty()) panel.addView(text(getString(R.string.dtc_none), bodyStyle, 4))

        if (d.suspects.isNotEmpty()) panel.addView(text(getString(R.string.dtc_inspect), titleStyle, 12))
        val detected = result?.parts?.mapTo(HashSet()) { it.name }
        val ripple = TypedValue().also { theme.resolveAttribute(android.R.attr.selectableItemBackground, it, true) }.resourceId
        for (s in d.suspects) {
            val st = d.status(s, detected)
            val note = getString(when (st) {
                SuspectStatus.FOUND -> R.string.dtc_found
                SuspectStatus.NOT_FOUND -> R.string.dtc_not_found
                SuspectStatus.NOT_DETECTABLE -> R.string.dtc_not_detectable
                SuspectStatus.NO_PHOTO -> R.string.dtc_no_photo
            })
            val mark = getString(when (st) {
                SuspectStatus.FOUND -> R.string.mark_found
                SuspectStatus.NOT_FOUND -> R.string.mark_missing
                else -> R.string.mark_other
            })
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                minimumHeight = (48 * dp).toInt()
                setPadding(0, (6 * dp).toInt(), 0, (6 * dp).toInt())
                addView(text(getString(R.string.dtc_suspect_line, mark, s.component.name.get(lang)), bodyStyle))
                addView(text(getString(R.string.dtc_suspect_note, note, s.codes.joinToString(", ")), smallStyle))
            }
            val best = if (st != SuspectStatus.FOUND) null
            else result?.parts?.filter { it.name == s.component.detectorClass }?.maxByOrNull { it.score }
            val has3d = ModelStore.has(this, s.component.key)
            if (best != null) {
                row.setBackgroundResource(ripple)
                row.setOnClickListener { showInfo(best) }
            } else if (has3d) {  // not in the photo (or not detectable): the 3D model shows what to look for
                row.setBackgroundResource(ripple)
                row.setOnClickListener { show3D(s.component.key, s.codes) }
            }
            val line = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = android.view.Gravity.CENTER_VERTICAL
                addView(row, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            }
            if (has3d) line.addView(MaterialButton(this, null, com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
                setText(R.string.model_badge)
                contentDescription = getString(R.string.model_view_3d)
                setIconResource(R.drawable.ic_cube)
                cornerRadius = (18 * dp).toInt()
                minHeight = (36 * dp).toInt()
                insetTop = 0
                insetBottom = 0
                setOnClickListener { show3D(s.component.key, s.codes) }
            })
            panel.addView(line)
        }
        if (d.safety.isNotEmpty()) {
            panel.addView(text(getString(R.string.dtc_safety), titleStyle, 12))
            panel.addView(text(d.safety.joinToString("\n") { getString(R.string.bullet, it.get(lang)) }, bodyStyle, 4).boxed(true))
        }
        panel.addView(text(getString(R.string.dtc_note), smallStyle, 8))
    }

    override fun onPause() {
        inspection.persist()
        liveModelView?.onPause()
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        liveModelView?.onResume()
    }

    override fun onDestroy() {
        inspection.close()
        // AppState belongs to the process. A replacement activity can already be using it when an older,
        // finishing activity reaches onDestroy (Back -> quick relaunch). Keep the shared model and draft alive;
        // Android reclaims the process when memory is needed. Never close another activity's active ORT session.
        super.onDestroy()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString("engineCapture", pendingCapture?.toString())
        inspection.saveState(outState)
    }

    companion object {
        const val TAG = "EngineBay"
    }
}
