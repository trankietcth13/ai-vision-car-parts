package com.enginebay.vision.core

import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * Every user-facing text the app ships must exist in English AND Vietnamese. Run after replacing
 * assets/components.json (e.g. with Gemini's content): it lists every missing translation.
 */
class BilingualDataTest {
    private val assets = listOf("src/main/assets", "app/src/main/assets").map(::File).firstOrNull { it.isDirectory }

    @Suppress("UNCHECKED_CAST")
    private fun json(name: String) = MiniJson.parse(File(assets, name).readText()) as Map<String, Any?>

    /** Collect problems for one {"en","vi"} value at [path]. */
    private fun checkText(v: Any?, path: String, problems: MutableList<String>) {
        when (v) {
            null -> Unit
            is String -> if (v.isNotBlank()) problems += "$path: plain string, expected {\"en\", \"vi\"}"
            is Map<*, *> -> {
                val en = (v["en"] as? String).orEmpty(); val vi = (v["vi"] as? String).orEmpty()
                if (en.isBlank() != vi.isBlank()) problems += "$path: missing ${if (en.isBlank()) "en" else "vi"}"
            }
            else -> problems += "$path: unexpected ${v::class.simpleName}"
        }
    }

    @Test
    fun componentInfoIsCompleteInBothLanguages() {
        assumeTrue(assets != null && File(assets, "components.json").isFile)
        val root = json("components.json")
        @Suppress("UNCHECKED_CAST") val comps = root["components"] as Map<String, Map<String, Any?>>
        val problems = mutableListOf<String>()
        for ((key, c) in comps) {
            checkText(c["name"], "$key.name", problems)
            if (c["name"] == null) problems += "$key.name: missing"
            for (f in listOf("summary", "function", "location_hint")) checkText(c[f], "$key.$f", problems)
            for (f in listOf("inspection_checks", "common_symptoms", "safety_notes")) {
                (c[f] as? List<*>)?.forEachIndexed { i, t -> checkText(t, "$key.$f[$i]", problems) }
            }
            (c["related_dtcs"] as? List<*>)?.forEachIndexed { i, d ->
                val m = d as Map<*, *>
                if ((m["code"] as? String).isNullOrBlank()) problems += "$key.related_dtcs[$i].code: missing"
                checkText(m["meaning"], "$key.related_dtcs[$i].meaning", problems)
                if (m["meaning"] == null) problems += "$key.related_dtcs[$i].meaning: missing"
            }
        }
        // every class the model can output has an entry
        val cfg = File(assets, "config.json")
        if (cfg.isFile) {
            @Suppress("UNCHECKED_CAST") val names = json("config.json")["names"] as List<String>
            for (n in names) if (n !in comps) problems += "$n: no entry in components.json"
        }
        assertTrue("translations missing:\n" + problems.joinToString("\n"), problems.isEmpty())
    }

    @Test
    fun classNamesExistInBothLanguages() {
        assumeTrue(assets != null && File(assets, "config.json").isFile)
        val cfg = json("config.json")
        @Suppress("UNCHECKED_CAST") val names = cfg["names"] as List<String>
        for (key in listOf("names_en", "names_vi")) {
            @Suppress("UNCHECKED_CAST") val l = cfg[key] as? List<String>
            assertTrue("$key missing", l != null && l.size == names.size)
            l!!.forEachIndexed { i, s -> assertTrue("$key[$i] (${names[i]}) is blank", s.isNotBlank()) }
        }
    }
}
