package com.enginebay.vision

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Component information from assets/components.json (schema: docs/COMPONENT_INFO_SPEC.md in this project).
 * Every field is optional; empty fields are not shown.
 */
class ComponentInfo(
    val nameVi: String,
    val nameEn: String,
    val draft: Boolean,
    val summary: String,
    val function: String,
    val locationHint: String,
    val inspectionChecks: List<String>,
    val commonSymptoms: List<String>,
    val relatedDtcs: List<Pair<String, String>>,
    val safetyNotes: List<String>,
) {
    val isEmpty get() = summary.isBlank() && function.isBlank() && locationHint.isBlank() && inspectionChecks.isEmpty() &&
        commonSymptoms.isEmpty() && relatedDtcs.isEmpty() && safetyNotes.isEmpty()

    companion object {
        fun loadAll(context: Context): Map<String, ComponentInfo> = try {
            val root = JSONObject(context.assets.open("components.json").bufferedReader().use { it.readText() })
            val comps = root.getJSONObject("components")
            comps.keys().asSequence().associateWith { parse(comps.getJSONObject(it)) }
        } catch (e: Exception) {
            emptyMap()  // the app still works without component information
        }

        private fun strings(a: JSONArray?) = if (a == null) emptyList() else List(a.length()) { a.optString(it) }.filter { it.isNotBlank() }

        private fun parse(o: JSONObject) = ComponentInfo(
            nameVi = o.optString("name_vi"),
            nameEn = o.optString("name_en"),
            draft = o.optBoolean("draft", false),
            summary = o.optString("summary"),
            function = o.optString("function"),
            locationHint = o.optString("location_hint"),
            inspectionChecks = strings(o.optJSONArray("inspection_checks")),
            commonSymptoms = strings(o.optJSONArray("common_symptoms")),
            relatedDtcs = o.optJSONArray("related_dtcs")?.let { a ->
                List(a.length()) { a.getJSONObject(it).let { d -> d.optString("code") to d.optString("meaning") } }
            } ?: emptyList(),
            safetyNotes = strings(o.optJSONArray("safety_notes")),
        )
    }
}
