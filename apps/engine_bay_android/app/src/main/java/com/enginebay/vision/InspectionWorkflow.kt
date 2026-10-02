package com.enginebay.vision

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.widget.NestedScrollView
import com.enginebay.vision.core.*
import com.google.android.material.button.MaterialButton
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import java.io.ByteArrayOutputStream
import java.io.File
import java.text.DateFormat
import java.util.Date
import java.util.UUID

/** UI for the local inspection workflow; persistence and image encoding run on the existing serial worker. */
class InspectionWorkflow(private val activity: MainActivity, private val locate: (String?) -> Unit, private val reset: () -> Unit) {
    private val repository by lazy { InspectionRepository(File(activity.filesDir, "inspections")) }
    private lateinit var panel: LinearLayout
    private lateinit var summary: TextView
    private var saving = false
    private val handler = android.os.Handler(android.os.Looper.getMainLooper())
    private val pendingSave = Runnable { persist() }
    private var exportValue: Inspection? = null
    private val export = activity.registerForActivityResult(ActivityResultContracts.CreateDocument("text/html")) { uri ->
        val value = exportValue
        exportValue = null
        if (uri != null && value != null) {
            val labels = reportLabels()
            val language = lang()
            val date = date(value)
            background {
                val html = InspectionHtml.render(value, repository.photo(value.photoId), language, labels, date)
                activity.contentResolver.openOutputStream(uri, "wt")?.bufferedWriter(Charsets.UTF_8)?.use { it.write(html) }
                    ?: error("Cannot open destination")
                ui { message(R.string.report_saved) }
            }
        }
    }

    private fun lang() = activity.resources.configuration.locales[0].language
    private fun s(id: Int) = activity.getString(id)
    private fun date(v: Inspection) = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT,
        activity.resources.configuration.locales[0]).format(Date(v.createdAt))
    private fun ui(block: () -> Unit) = activity.runOnUiThread { if (!activity.isDestroyed) block() }
    private fun message(id: Int) { if (!activity.isDestroyed) Toast.makeText(activity, id, Toast.LENGTH_LONG).show() }
    private fun background(block: () -> Unit) {
        AppState.worker.execute {
            try { block() } catch (e: Exception) {
                android.util.Log.e(MainActivity.TAG, "Inspection operation failed", e)
                ui { saving = false; message(R.string.inspection_io_error) }
            }
        }
    }

    fun restoreExport(state: Bundle?) { exportValue = state?.getString("exportInspection")?.let { Inspection.decode(it) } }
    fun saveState(state: Bundle) { exportValue?.let { state.putString("exportInspection", it.encode()) } }

    /** Invoked on the serial worker before the first model load. */
    fun loadDraft(): Pair<Inspection?, Bitmap?> {
        return try {
            val value = repository.draft()
            val bytes = value?.let { repository.photo(it.photoId) }
            value to bytes?.let { BitmapFactory.decodeByteArray(it, 0, it.size) ?: error("Invalid saved photo") }
        } catch (e: Exception) {
            android.util.Log.e(MainActivity.TAG, "Draft could not be restored", e)
            ui { message(R.string.draft_restore_failed) }
            null to null
        }
    }

    fun bind() {
        val dp = activity.resources.displayMetrics.density
        panel = activity.findViewById(R.id.inspection_actions)
        panel.removeAllViews()
        summary = TextView(activity).apply {
            setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_TitleSmall)
            setPadding(0, 0, 0, (8 * dp).toInt())
        }
        panel.addView(summary)
        fun row(a: Int, first: () -> Unit, b: Int, second: () -> Unit) {
            panel.addView(LinearLayout(activity).apply {
                orientation = LinearLayout.HORIZONTAL
                listOf(a to first, b to second).forEachIndexed { index, (text, action) ->
                    val lp = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                        if (index == 0) rightMargin = (4 * dp).toInt() else leftMargin = (4 * dp).toInt()
                        topMargin = (4 * dp).toInt()
                    }
                    addView(button(text, action), lp)
                }
            })
        }
        row(R.string.inspection_details, ::details, R.string.find_component, ::find)
        row(R.string.inspection_checks, ::chooseChecks, R.string.inspection_save, ::save)
        row(R.string.inspection_history, ::history, R.string.inspection_new) {
            MaterialAlertDialogBuilder(activity).setMessage(R.string.inspection_new_confirm)
                .setPositiveButton(R.string.inspection_new) { _, _ ->
                    AppState.inspection = Inspection(); AppState.findTarget = null
                    reset(); persist(); refresh()
                }.setNegativeButton(R.string.cancel, null).show()
        }
        refresh()
    }

    private fun button(text: Int, action: () -> Unit): MaterialButton {
        val dp = activity.resources.displayMetrics.density
        return MaterialButton(activity, null, com.google.android.material.R.attr.materialButtonOutlinedStyle).apply {
            setText(text)
            cornerRadius = (20 * dp).toInt()
            minHeight = (44 * dp).toInt()
            insetTop = 0
            insetBottom = 0
            setOnClickListener { if (AppState.inspectionLoaded) action() }
        }
    }

    fun refresh() {
        if (!::summary.isInitialized) return
        val d = AppState.inspection
        val done = d.checks.count { it.outcome != CheckOutcome.PENDING }
        summary.text = activity.getString(R.string.inspection_summary, d.vehicle.ifBlank { s(R.string.inspection_vehicle_missing) }, done, d.checks.size)
        for (i in 0 until panel.childCount) {
            val child = panel.getChildAt(i)
            child.isEnabled = AppState.inspectionLoaded
            if (child is LinearLayout) for (j in 0 until child.childCount) child.getChildAt(j).isEnabled = AppState.inspectionLoaded
        }
    }

    private fun snapshot() = AppState.inspection.copy(codes = AppState.dtcText,
        threshold = AppState.conf ?: AppState.detector?.config?.defaultConf ?: 0.35f,
        perClass = AppState.perClass ?: true)

    fun persist() {
        handler.removeCallbacks(pendingSave)
        if (!AppState.inspectionLoaded) return
        val value = snapshot()
        val photo = AppState.photo
        AppState.inspection = value
        background { ensurePhoto(value, photo); repository.saveDraft(value) }
    }

    private fun ensurePhoto(value: Inspection, photo: Bitmap?) {
        if (value.photoId.isBlank() || repository.hasPhoto(value.photoId)) return
        require(photo != null)
        val bytes = ByteArrayOutputStream().use { out ->
            check(photo.compress(Bitmap.CompressFormat.JPEG, 90, out)); out.toByteArray()
        }
        repository.writePhoto(value.photoId, bytes)
    }

    fun close() { handler.removeCallbacks(pendingSave) }

    fun photoChanged(photo: Bitmap) {
        AppState.inspection = snapshot().copy(photoId = UUID.randomUUID().toString(), width = photo.width, height = photo.height,
            parts = emptyList(), checks = emptyList(), model = "")
        val value = AppState.inspection
        background {
            ensurePhoto(value, photo)
            repository.saveDraft(value)
        }
        refresh()
    }

    fun detected(result: DetectionResult) {
        val config = AppState.detector?.config
        AppState.inspection = snapshot().copy(model = config?.let { "${it.modelName} / ${it.release} / ${it.modelMd5}" }.orEmpty(),
            parts = result.parts.map { p -> InspectionPart(p.name, Text(p.nameEn, p.nameVi), p.score,
                listOf(p.rect.left, p.rect.top, p.rect.right, p.rect.bottom)) })
        persist(); refresh()
    }

    private fun column() = LinearLayout(activity).apply {
        orientation = LinearLayout.VERTICAL
        val p = (20 * resources.displayMetrics.density).toInt(); setPadding(p, p / 2, p, p / 2)
    }
    private fun scroll(box: LinearLayout) = ScrollView(activity).apply { addView(box) }
    private fun field(box: LinearLayout, hint: Int, value: String, multi: Boolean = false): TextInputEditText {
        val input = TextInputEditText(activity).apply {
            setText(value); if (!multi) setSingleLine() else { minLines = 2; maxLines = 5 }
            filters = arrayOf(android.text.InputFilter.LengthFilter(if (multi) 4000 else 160))
        }
        box.addView(TextInputLayout(activity).apply { this.hint = s(hint); addView(input) })
        return input
    }

    private fun details(after: (() -> Unit)? = null) {
        val box = column()
        val vehicle = field(box, R.string.inspection_vehicle, AppState.inspection.vehicle)
        val notes = field(box, R.string.inspection_notes, AppState.inspection.notes, true)
        box.addView(TextView(activity).apply { setText(R.string.inspection_photo_note) })
        val dialog = MaterialAlertDialogBuilder(activity).setTitle(R.string.inspection_details).setView(scroll(box))
            .setPositiveButton(R.string.inspection_apply, null).setNegativeButton(R.string.cancel, null).create()
        dialog.setOnShowListener {
            dialog.getButton(androidx.appcompat.app.AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                if (vehicle.text.toString().isBlank()) { vehicle.error = s(R.string.inspection_vehicle_required); return@setOnClickListener }
                AppState.inspection = snapshot().copy(vehicle = vehicle.text.toString().trim(), notes = notes.text.toString())
                persist(); refresh(); dialog.dismiss(); after?.invoke()
            }
        }
        dialog.show()
    }

    private fun names(): Map<String, Text> = linkedMapOf<String, Text>().apply {
        AppState.detector?.config?.let { c -> c.names.forEachIndexed { i, key -> put(key, Text(c.namesEn[i], c.namesVi[i])) } }
        AppState.dtcTable?.components?.forEach { (k, v) -> put(k, v.name) }
    }

    fun find() {
        val names = names()
        if (names.isEmpty()) { message(R.string.loading_model); return }
        val entries = names.entries.sortedBy { it.value.get(lang()) }
        MaterialAlertDialogBuilder(activity).setTitle(R.string.find_component)
            .setItems(entries.map { it.value.get(lang()) }.toTypedArray()) { _, i ->
                val key = entries[i].key
                val detectorKey = AppState.dtcTable?.components?.get(key)?.detectorClass
                    ?: key.takeIf { it in AppState.detector?.config?.names.orEmpty() }
                if (detectorKey == null) message(R.string.model_status_not_detectable)
                else {
                    locate(detectorKey)
                    if (AppState.photo == null) message(R.string.dtc_no_photo)
                    else if (AppState.result?.parts?.none { it.name == detectorKey } != false) message(R.string.find_not_seen)
                    activity.findViewById<NestedScrollView>(R.id.scroll).smoothScrollTo(0, 0)
                }
            }.setNeutralButton(R.string.find_clear) { _, _ -> locate(null) }.setNegativeButton(R.string.cancel, null).show()
    }

    private fun chooseChecks() {
        val names = names()
        if (names.isEmpty()) { message(R.string.loading_model); return }
        val suggested = AppState.diagnosis?.suspects?.map { it.component.key }.orEmpty().toSet() + AppState.result?.parts.orEmpty().map { it.name }
        val entries = names.entries.sortedWith(compareByDescending<Map.Entry<String, Text>> { it.key in suggested }.thenBy { it.value.get(lang()) })
        MaterialAlertDialogBuilder(activity).setTitle(R.string.inspection_checks)
            .setItems(entries.map { it.value.get(lang()) }.toTypedArray()) { _, i -> checks(entries[i].key) }.show()
    }

    private fun bilingual(id: Int): Text {
        fun localized(language: String): String {
            val config = android.content.res.Configuration(activity.resources.configuration)
            config.setLocale(java.util.Locale.forLanguageTag(language))
            return activity.createConfigurationContext(config).getString(id)
        }
        return Text(localized("en"), localized("vi"))
    }

    fun checks(key: String) {
        val inspectionId = AppState.inspection.id
        val photoId = AppState.inspection.photoId
        val name = names()[key] ?: Text(key, key)
        val detectorKey = AppState.dtcTable?.components?.get(key)?.detectorClass ?: key
        val info = AppState.components[detectorKey]
        fun open(extra: List<Pair<String, Text>>) {
            if (activity.isDestroyed || AppState.inspection.id != inspectionId || AppState.inspection.photoId != photoId) return
            val steps = listOf("identity" to bilingual(R.string.check_identity), "visual" to bilingual(R.string.check_visual)) + extra
            val existing = AppState.inspection.checks.filter { it.component == key }.map { it.step }.toSet()
            AppState.inspection = snapshot().copy(checks = AppState.inspection.checks + steps.filter { it.first !in existing }
                .map { (id, instruction) -> InspectionCheck(key, name, id, instruction) })
            persist(); refresh(); showChecks(key, name)
        }
        val specific = info?.checks.orEmpty().mapIndexed { i, t -> "info-$i" to Text(t.en, t.vi) }
        if (specific.isNotEmpty() || !ModelStore.has(activity, key)) open(specific)
        else ModelStore.load(activity, key) { scene -> ui {
            open(scene?.model?.parts.orEmpty().mapNotNull { p -> p.check?.let { "model-${p.name}" to it } })
        } }
    }

    private fun showChecks(key: String, name: Text) {
        val box = column()
        box.addView(TextView(activity).apply { setText(R.string.checklist_note) })
        val inspectionId = AppState.inspection.id
        fun update(step: String, change: (InspectionCheck) -> InspectionCheck) {
            if (AppState.inspection.id != inspectionId) return
            AppState.inspection = snapshot().copy(checks = AppState.inspection.checks.map {
                if (it.component == key && it.step == step) change(it) else it
            })
            handler.removeCallbacks(pendingSave)
            handler.postDelayed(pendingSave, 400)
            refresh()
        }
        AppState.inspection.checks.filter { it.component == key }.forEach { check ->
            box.addView(TextView(activity).apply { text = check.instruction.get(lang()); setPadding(0, 24, 0, 8) })
            box.addView(Spinner(activity).apply {
                adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_dropdown_item, outcomeLabels())
                setSelection(check.outcome.ordinal)
                onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                    override fun onNothingSelected(parent: AdapterView<*>?) = Unit
                    override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                        update(check.step) { it.copy(outcome = CheckOutcome.entries[position]) }
                    }
                }
            })
            field(box, R.string.inspection_observation, check.note, true).addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) { update(check.step) { it.copy(note = s.toString()) } }
                override fun afterTextChanged(s: Editable?) = Unit
            })
        }
        MaterialAlertDialogBuilder(activity).setTitle(name.get(lang())).setView(scroll(box))
            .setPositiveButton(R.string.inspection_done, null).show()
    }

    private fun save() {
        if (saving) return
        if (AppState.inspection.vehicle.isBlank()) { details { save() }; return }
        if (AppState.photo != null && AppState.result == null) { message(R.string.inspection_wait); return }
        val value = snapshot().copy(id = UUID.randomUUID().toString(), createdAt = System.currentTimeMillis())
        if (value.photoId.isBlank() && value.codes.isBlank() && value.checks.isEmpty() && value.notes.isBlank()) {
            message(R.string.inspection_empty); return
        }
        saving = true
        val photo = AppState.photo
        background {
            ensurePhoto(value, photo)
            repository.save(value)
            ui { saving = false; message(R.string.inspection_saved); preview(value) }
        }
    }

    private fun history() {
        background {
            val history = repository.list()
            ui {
                if (history.unreadable > 0) message(R.string.history_damaged)
                if (history.inspections.isEmpty()) { message(R.string.history_empty); return@ui }
                val box = column()
                val search = field(box, R.string.history_search, "")
                val list = ListView(activity)
                var visible = history.inspections
                fun render(query: String) {
                    visible = history.inspections.filter { it.vehicle.contains(query, true) }
                    list.adapter = ArrayAdapter(activity, android.R.layout.simple_list_item_1, visible.map { "${it.vehicle} · ${date(it)}" })
                }
                render("")
                box.addView(list, LinearLayout.LayoutParams(-1, (320 * activity.resources.displayMetrics.density).toInt()))
                val dialog = MaterialAlertDialogBuilder(activity).setTitle(R.string.inspection_history).setView(box)
                    .setNegativeButton(R.string.inspection_done, null).show()
                list.setOnItemClickListener { _, _, i, _ -> dialog.dismiss(); preview(visible[i]) }
                search.addTextChangedListener(object : TextWatcher {
                    override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
                    override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = render(s.toString())
                    override fun afterTextChanged(s: Editable?) = Unit
                })
            }
        }
    }

    private fun preview(value: Inspection) {
        background {
            val bytes = repository.photo(value.photoId)
            val image = bytes?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
            ui {
                val box = column()
                box.addView(TextView(activity).apply { text = date(value) + "\n" + s(R.string.report_disclaimer) })
                if (image != null) box.addView(ImageView(activity).apply { setImageBitmap(image); adjustViewBounds = true })
                val body = buildString {
                    append("${s(R.string.dtc_title)}: ${value.codes}\n\n${s(R.string.components)}\n")
                    value.parts.forEach { append("${it.name.get(lang())} · ${(it.score * 100).toInt()}%\n") }
                    append("\n${s(R.string.inspection_checks)}\n")
                    value.checks.forEach { append("${it.componentName.get(lang())}: ${it.instruction.get(lang())}\n${outcomeLabels()[it.outcome.ordinal]}\n${it.note}\n\n") }
                    append("${s(R.string.inspection_notes)}\n${value.notes}")
                }
                box.addView(TextView(activity).apply { text = body; setTextIsSelectable(true) })
                MaterialAlertDialogBuilder(activity).setTitle(value.vehicle).setView(scroll(box))
                    .setPositiveButton(R.string.report_export) { _, _ ->
                        exportValue = value
                        export.launch("inspection-${value.id}.html")
                    }.setNegativeButton(R.string.inspection_done, null).show()
            }
        }
    }

    private fun outcomeLabels() = listOf(R.string.check_pending, R.string.check_done, R.string.check_concern, R.string.check_skipped).map(::s)
    private fun reportLabels() = mapOf(
        "title" to s(R.string.report_title), "vehicle" to s(R.string.inspection_vehicle), "codes" to s(R.string.dtc_title),
        "model" to s(R.string.report_model), "disclaimer" to s(R.string.report_disclaimer), "photo" to s(R.string.report_photo),
        "parts" to s(R.string.components), "checks" to s(R.string.inspection_checks), "notes" to s(R.string.inspection_notes),
    ) + CheckOutcome.entries.mapIndexed { i, value -> value.name to outcomeLabels()[i] }
}
