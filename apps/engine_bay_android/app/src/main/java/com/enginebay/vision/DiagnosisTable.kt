package com.enginebay.vision

import android.content.Context
import com.enginebay.vision.core.ComponentRef
import com.enginebay.vision.core.DtcEntry
import com.enginebay.vision.core.DtcTable
import com.enginebay.vision.core.Text
import org.json.JSONArray
import org.json.JSONObject

/** assets/diagnosis.json (every text is {"en", "vi"}), written by scripts/deployment/export_android_diagnosis.py from configs/diagnosis_knowledge.yaml. */
object DiagnosisTable {
    /** null when the asset is missing or unreadable: the app then works without the error-code feature. */
    fun load(context: Context): DtcTable? = try {
        parse(JSONObject(context.assets.open("diagnosis.json").bufferedReader().use { it.readText() }))
    } catch (e: Exception) {
        null
    }

    private fun strings(a: JSONArray?) = if (a == null) emptyList() else List(a.length()) { a.getString(it) }
    private fun text(o: JSONObject) = Text(o.getString("en"), o.getString("vi"))
    private fun texts(a: JSONArray?) = if (a == null) emptyList() else List(a.length()) { text(a.getJSONObject(it)) }

    fun parse(root: JSONObject): DtcTable {
        val comps = root.getJSONObject("components")
        val components = comps.keys().asSequence().associateWith { key ->
            val c = comps.getJSONObject(key)
            ComponentRef(key, text(c.getJSONObject("name")), if (c.isNull("detector_class")) null else c.getString("detector_class"),
                c.optJSONObject("description")?.let(::text))
        }
        val dtc = root.getJSONArray("dtc")
        val entries = List(dtc.length()) { i ->
            val d = dtc.getJSONObject(i)
            DtcEntry(strings(d.getJSONArray("codes")), text(d.getJSONObject("meaning")), strings(d.getJSONArray("components")),
                d.optString("urgency", "check"), texts(d.optJSONArray("safety")))
        }
        return DtcTable(entries, components)
    }
}
