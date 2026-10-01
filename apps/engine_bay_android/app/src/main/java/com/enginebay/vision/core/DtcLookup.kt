package com.enginebay.vision.core

/** A user-facing text in both app languages. */
data class Text(val en: String, val vi: String) {
    /** [lang]: ISO 639 language code of the UI; anything but "vi" gets English. */
    fun get(lang: String) = if (lang == "vi") vi else en
}

/** One row of the DTC table (assets/diagnosis.json). [codes] are single codes or same-letter ranges ("P0301-P0308"). */
class DtcEntry(
    val codes: List<String>,
    val meaning: Text,
    val components: List<String>,
    /** stop | soon | check */
    val urgency: String,
    val safety: List<Text>,
)

/** A component that can be suggested; [detectorClass] is the model class, or null when the model cannot see it. */
class ComponentRef(val key: String, val name: Text, val detectorClass: String?, val description: Text? = null)

class DtcTable(val entries: List<DtcEntry>, val components: Map<String, ComponentRef>)

class DtcMatch(val code: String, val entry: DtcEntry)

/** A component to inspect and the codes that point to it. */
class Suspect(val component: ComponentRef, val codes: List<String>)

enum class SuspectStatus { FOUND, NOT_FOUND, NOT_DETECTABLE, NO_PHOTO }

class Diagnosis(
    val matches: List<DtcMatch>,
    val unknown: List<String>,
    /** most codes first, then table order */
    val suspects: List<Suspect>,
    /** worst urgency of the matched codes: stop > soon > check; "check" when nothing matched */
    val urgency: String,
    val safety: List<Text>,
) {
    /** Model classes to highlight in the photo. */
    val detectorTargets: Set<String> get() = suspects.mapNotNullTo(LinkedHashSet()) { it.component.detectorClass }

    fun status(s: Suspect, detected: Set<String>?): SuspectStatus = when {
        s.component.detectorClass == null -> SuspectStatus.NOT_DETECTABLE
        detected == null -> SuspectStatus.NO_PHOTO
        s.component.detectorClass in detected -> SuspectStatus.FOUND
        else -> SuspectStatus.NOT_FOUND
    }
}

/**
 * Offline DTC lookup, the same rules as lookup_dtc() in src/jev/diagnosis_triage.py: the code letter must equal the
 * range letter and the rest is compared as a hexadecimal number; the first table entry that contains the code wins.
 * Checked against that function by DtcLookupTest (fixtures from scripts/deployment/export_android_diagnosis.py).
 */
object DtcLookup {
    private val URGENCY_RANK = mapOf("check" to 0, "soon" to 1, "stop" to 2)

    /** Codes typed or read from OBD, separated by anything that is not a letter or digit; upper-cased, duplicates dropped. */
    fun parseCodes(text: String): List<String> =
        text.split(Regex("[^A-Za-z0-9]+")).filter { it.isNotEmpty() }.map { it.uppercase() }.distinct()

    private fun value(code: String): Pair<Char, Int>? {
        if (code.length < 2) return null
        val v = code.substring(1).toIntOrNull(16) ?: return null
        return code[0] to v
    }

    fun find(code: String, table: DtcTable): DtcEntry? {
        val (p, v) = value(code) ?: return null
        for (entry in table.entries) for (spec in entry.codes) {
            val lo = value(spec.substringBefore('-')) ?: continue
            val hi = if ('-' in spec) value(spec.substringAfter('-')) ?: continue else lo
            if (p == lo.first && p == hi.first && v in lo.second..hi.second) return entry
        }
        return null
    }

    fun diagnose(codes: List<String>, table: DtcTable): Diagnosis {
        val matches = ArrayList<DtcMatch>()
        val unknown = ArrayList<String>()
        for (code in codes) {
            val e = find(code, table)
            if (e == null) unknown += code else matches += DtcMatch(code, e)
        }
        val byComponent = LinkedHashMap<String, MutableList<String>>()
        for (m in matches) for (c in m.entry.components) byComponent.getOrPut(c) { ArrayList() }.let { if (m.code !in it) it += m.code }
        val suspects = byComponent.entries.mapIndexedNotNull { i, (key, cs) -> table.components[key]?.let { Triple(i, Suspect(it, cs), cs.size) } }
            .sortedWith(compareByDescending<Triple<Int, Suspect, Int>> { it.third }.thenBy { it.first })
            .map { it.second }
        val urgency = matches.maxByOrNull { URGENCY_RANK[it.entry.urgency] ?: 0 }?.entry?.urgency ?: "check"
        val safety = matches.flatMap { it.entry.safety }.distinct()
        return Diagnosis(matches, unknown, suspects, urgency, safety)
    }

    fun diagnose(text: String, table: DtcTable): Diagnosis = diagnose(parseCodes(text), table)
}
