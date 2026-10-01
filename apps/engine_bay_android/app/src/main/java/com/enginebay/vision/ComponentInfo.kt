package com.enginebay.vision

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/** A text in both app languages. */
data class Bilingual(val en: String, val vi: String) {
    fun get(vietnamese: Boolean) = if (vietnamese) vi else en
    val isBlank get() = en.isBlank() && vi.isBlank()

    companion object {
        val EMPTY = Bilingual("", "")

        /** {"en": ..., "vi": ...}; a plain string (schema 1) is used for both languages. */
        fun parse(v: Any?): Bilingual = when (v) {
            is JSONObject -> Bilingual(v.optString("en"), v.optString("vi"))
            is String -> Bilingual(v, v)
            else -> EMPTY
        }
    }
}

/**
 * Component information from assets/components.json (schema 2, bilingual; docs/COMPONENT_INFO_SPEC.md).
 * Every field is optional; empty fields are not shown.
 */
class ComponentInfo(
    val name: Bilingual,
    val draft: Boolean,
    val summaryText: Bilingual,
    val functionText: Bilingual,
    val locationText: Bilingual,
    val checks: List<Bilingual>,
    val symptoms: List<Bilingual>,
    val dtcs: List<Pair<String, Bilingual>>,
    val safety: List<Bilingual>,
) {
    val isEmpty get() = summaryText.isBlank && functionText.isBlank && locationText.isBlank && checks.isEmpty() &&
        symptoms.isEmpty() && dtcs.isEmpty() && safety.isEmpty()

    companion object {
        fun loadAll(context: Context): Map<String, ComponentInfo> = try {
            parseAll(context.assets.open("components.json").bufferedReader().use { it.readText() })
        } catch (e: Exception) {
            emptyMap()  // the app still works without component information
        }

        fun parseAll(json: String): Map<String, ComponentInfo> {
            val comps = JSONObject(json).getJSONObject("components")
            return comps.keys().asSequence().associateWith { parse(comps.getJSONObject(it)) }
        }

        private fun texts(a: JSONArray?) =
            if (a == null) emptyList() else List(a.length()) { Bilingual.parse(a.opt(it)) }.filterNot { it.isBlank }

        private fun parse(o: JSONObject) = ComponentInfo(
            // schema 1 had name_vi / name_en
            name = if (o.has("name")) Bilingual.parse(o.opt("name")) else Bilingual(o.optString("name_en"), o.optString("name_vi")),
            draft = o.optBoolean("draft", false),
            summaryText = Bilingual.parse(o.opt("summary")),
            functionText = Bilingual.parse(o.opt("function")),
            locationText = Bilingual.parse(o.opt("location_hint")),
            checks = texts(o.optJSONArray("inspection_checks")),
            symptoms = texts(o.optJSONArray("common_symptoms")),
            dtcs = o.optJSONArray("related_dtcs")?.let { a ->
                List(a.length()) { a.getJSONObject(it).let { d -> d.optString("code") to Bilingual.parse(d.opt("meaning")) } }
            } ?: emptyList(),
            safety = texts(o.optJSONArray("safety_notes")),
        )
    }
}
