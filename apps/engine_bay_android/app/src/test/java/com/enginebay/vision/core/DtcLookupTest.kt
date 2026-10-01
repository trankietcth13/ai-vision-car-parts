package com.enginebay.vision.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The app's DTC lookup against lookup_dtc() in src/jev/diagnosis_triage.py. assets/diagnosis.json and
 * fixtures/dtc_lookup.json come from scripts/deployment/export_android_diagnosis.py; skipped when not generated.
 * The table is parsed with the test's MiniJson (org.json is a stub in local unit tests).
 */
@Suppress("UNCHECKED_CAST")
class DtcLookupTest {
    private fun file(rel: String) = listOf("src/$rel", "app/src/$rel").map(::File).firstOrNull { it.isFile }
    private val tableFile = file("main/assets/diagnosis.json")
    private val casesFile = file("test/resources/fixtures/dtc_lookup.json")

    private val table: DtcTable by lazy {
        val root = MiniJson.parse(tableFile!!.readText()) as Map<String, Any?>
        fun text(o: Any?) = (o as Map<String, Any?>).let { Text(it["en"] as String, it["vi"] as String) }
        val comps = (root["components"] as Map<String, Map<String, Any?>>).mapValues { (k, c) ->
            ComponentRef(k, text(c["name"]), c["detector_class"] as String?)
        }
        val entries = (root["dtc"] as List<Map<String, Any?>>).map { d ->
            DtcEntry(d["codes"] as List<String>, text(d["meaning"]), d["components"] as List<String>,
                d["urgency"] as String, (d["safety"] as List<Any?>).map(::text))
        }
        DtcTable(entries, comps)
    }

    @Test
    fun lookupMatchesPython() {
        assumeTrue("diagnosis.json / dtc_lookup.json not generated", tableFile != null && casesFile != null)
        val cases = (MiniJson.parse(casesFile!!.readText()) as Map<String, Any?>)["cases"] as List<Map<String, Any?>>
        assertTrue(cases.isNotEmpty())
        for (c in cases) {
            val input = c["input"] as String
            val d = DtcLookup.diagnose(input, table)
            val expected = c["matched"] as List<Map<String, Any?>>
            assertEquals("matched codes for $input", expected.map { it["code"] }, d.matches.map { it.code })
            assertEquals("components for $input", expected.map { it["components"] }, d.matches.map { it.entry.components })
            assertEquals("unknown for $input", c["unknown"], d.unknown)
        }
    }

    @Test
    fun parsesFreeTextAndRanksSharedComponents() {
        assumeTrue("diagnosis.json not generated", tableFile != null)
        val d = DtcLookup.diagnose(" p0171, P0300;p0171  U9999 ", table)
        assertEquals(listOf("P0171", "P0300"), d.matches.map { it.code })
        assertEquals(listOf("U9999"), d.unknown)
        // intake_manifold and fuel_injector are named by both codes, so they come first (in table order)
        assertEquals(listOf("intake_manifold", "fuel_injector"), d.suspects.take(2).map { it.component.key })
        assertEquals(listOf("P0171", "P0300"), d.suspects[0].codes)
        assertTrue("intake_manifold" in d.detectorTargets)
        assertTrue(d.suspects.none { it.component.detectorClass != null && it.component.detectorClass !in d.detectorTargets })
    }

    @Test
    fun everyTextIsBilingual() {
        assumeTrue("diagnosis.json not generated", tableFile != null)
        val texts = table.components.values.map { it.name } + table.entries.flatMap { listOf(it.meaning) + it.safety }
        for (t in texts) assertTrue("missing translation: $t", t.en.isNotBlank() && t.vi.isNotBlank() && t.en != t.vi)
        val e = DtcLookup.find("P0217", table)!!
        assertEquals("Engine overheat condition", e.meaning.get("en"))
        assertEquals(e.meaning.vi, e.meaning.get("vi"))
        assertEquals(e.meaning.en, e.meaning.get("de"))
    }

    @Test
    fun urgencyAndStatus() {
        assumeTrue("diagnosis.json not generated", tableFile != null)
        assertEquals("stop", DtcLookup.diagnose("P0128 P0217", table).urgency)
        assertEquals("check", DtcLookup.diagnose("P0420", table).urgency)
        assertEquals("check", DtcLookup.diagnose("B1234", table).urgency)
        val d = DtcLookup.diagnose("P0301", table)
        val coil = d.suspects.first { it.component.key == "ignition_coil" }
        val plug = d.suspects.first { it.component.key == "spark_plug" }
        assertEquals(SuspectStatus.NO_PHOTO, d.status(coil, null))
        assertEquals(SuspectStatus.FOUND, d.status(coil, setOf("ignition_coil")))
        assertEquals(SuspectStatus.NOT_FOUND, d.status(coil, setOf("battery")))
        assertEquals(SuspectStatus.NOT_DETECTABLE, d.status(plug, setOf("ignition_coil")))
        assertTrue(d.safety.isNotEmpty())
    }
}
