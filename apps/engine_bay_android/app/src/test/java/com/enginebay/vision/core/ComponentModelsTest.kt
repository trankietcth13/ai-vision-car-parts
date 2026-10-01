package com.enginebay.vision.core

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * The 3D component models (assets/models/<component>.json, scripts/deployment/build_component_models.py):
 *  - every component of the knowledge table has one,
 *  - the on-device builder (ModelBuilder) makes the SAME geometry as the Python builder: per part the triangle count,
 *    surface area, signed volume and bounds match fixtures/model_stats.json,
 *  - every part label and inspection note exists in English AND Vietnamese, sizes stay phone-friendly,
 *  - the GLB reader (for models shipped as files) reads the Python GLB of the battery into the same geometry.
 */
class ComponentModelsTest {
    private val assets = listOf("src/main/assets", "app/src/main/assets").map(::File).firstOrNull { it.isDirectory }
    private val dir get() = File(assets, "models")
    /** Test fixture written by scripts/deployment/build_component_models.py (not in git); the test is skipped without it. */
    private fun fixture(name: String): File {
        val url = javaClass.classLoader!!.getResource("fixtures/$name")
        assumeTrue("run scripts/deployment/build_component_models.py to create fixtures/$name", url != null)
        return File(url!!.toURI())
    }

    private fun specs() = dir.listFiles { f -> f.name.endsWith(".json") }.orEmpty().sortedBy { it.name }

    @Test
    fun everyKnowledgeComponentHasAModel() {
        assumeTrue(assets != null && File(assets, "diagnosis.json").isFile)
        @Suppress("UNCHECKED_CAST")
        val comps = (MiniJson.parse(File(assets, "diagnosis.json").readText()) as Map<String, Any?>)["components"] as Map<String, Any?>
        val have = specs().map { it.name.removeSuffix(".json") }.toSet()
        val missing = comps.keys - have
        assertTrue("no 3D model for: $missing", missing.isEmpty())
    }

    /** Area / volume / bounds of a built part, from its welded triangles (the same triangles as the soup). */
    private fun stats(p: ModelPart): DoubleArray {
        var tris = 0; var area = 0.0; var vol = 0.0
        val lo = DoubleArray(3) { Double.MAX_VALUE }; val hi = DoubleArray(3) { -Double.MAX_VALUE }
        for (pr in p.prims) {
            val v = pr.positions; val ix = pr.indices
            for (t in 0 until ix.size / 3) {
                val a = 3 * ix[3 * t]; val b = 3 * ix[3 * t + 1]; val c = 3 * ix[3 * t + 2]
                val ux = (v[b] - v[a]).toDouble(); val uy = (v[b + 1] - v[a + 1]).toDouble(); val uz = (v[b + 2] - v[a + 2]).toDouble()
                val wx = (v[c] - v[a]).toDouble(); val wy = (v[c + 1] - v[a + 1]).toDouble(); val wz = (v[c + 2] - v[a + 2]).toDouble()
                val cx = uy * wz - uz * wy; val cy = uz * wx - ux * wz; val cz = ux * wy - uy * wx
                area += 0.5 * sqrt(cx * cx + cy * cy + cz * cz)
                // signed volume: p0 . (p1 x p2) / 6
                val bx = v[b].toDouble(); val by = v[b + 1].toDouble(); val bz = v[b + 2].toDouble()
                val qx = v[c].toDouble(); val qy = v[c + 1].toDouble(); val qz = v[c + 2].toDouble()
                vol += (v[a] * (by * qz - bz * qy) + v[a + 1] * (bz * qx - bx * qz) + v[a + 2] * (bx * qy - by * qx)) / 6
                tris++
            }
            for (i in v.indices) { lo[i % 3] = minOf(lo[i % 3], v[i].toDouble()); hi[i % 3] = maxOf(hi[i % 3], v[i].toDouble()) }
        }
        return doubleArrayOf(tris.toDouble(), area, vol, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
    }

    /** Compare a built model with the Python fingerprints; returns the problems. */
    private fun compare(key: String, m: Model3D, ref: JSONObject): List<String> {
        val problems = mutableListOf<String>()
        val names = ref.keys().asSequence().toSet()
        if (m.parts.map { it.name }.toSet() != names) problems += "$key: parts ${m.parts.map { it.name }} != $names"
        for (p in m.parts) {
            val r = ref.optJSONObject(p.name) ?: continue
            val s = stats(p)
            val tris = r.getInt("triangles")
            if (s[0].toInt() != tris) problems += "$key.${p.name}: ${s[0].toInt()} triangles, Python $tris"
            val area = r.getDouble("area"); val vol = r.getDouble("volume")
            if (abs(s[1] - area) > 1e-4 * area + 1e-3) problems += "$key.${p.name}: area ${s[1]} vs $area"
            if (abs(s[2] - vol) > 1e-4 * max(abs(vol), area) + 1e-3) problems += "$key.${p.name}: volume ${s[2]} vs $vol"
            val lo = r.getJSONArray("min"); val hi = r.getJSONArray("max")
            for (k in 0..2) if (abs(s[3 + k] - lo.getDouble(k)) > 1e-3 || abs(s[6 + k] - hi.getDouble(k)) > 1e-3)
                problems += "$key.${p.name}: bounds differ on axis $k"
        }
        return problems
    }

    @Test
    fun deviceBuilderMatchesPython() {
        assumeTrue(assets != null && dir.isDirectory)
        val ref = JSONObject(fixture("model_stats.json").readText())
        val problems = mutableListOf<String>()
        var total = 0L
        for (f in specs()) {
            val key = f.name.removeSuffix(".json")
            val t0 = System.nanoTime()
            val m = ModelBuilder.build(f.readText())
            total += System.nanoTime() - t0
            assertEquals("$key: component", key, m.component)
            if (!ref.has(key)) { problems += "$key: not in model_stats.json (re-run the build script)"; continue }
            problems += compare(key, m, ref.getJSONObject(key))
        }
        println("built ${specs().size} models on the JVM in ${total / 1_000_000} ms")
        assertTrue(problems.joinToString("\n"), problems.isEmpty())
    }

    @Test
    fun modelsAreBilingualAndLight() {
        assumeTrue(assets != null && dir.isDirectory)
        val problems = mutableListOf<String>()
        for (f in specs()) {
            val key = f.name.removeSuffix(".json")
            val m = ModelBuilder.build(f.readText())
            if (m.parts.isEmpty()) problems += "$key: no parts"
            if (m.triangles > 40_000) problems += "$key: ${m.triangles} triangles (too heavy for low-end tablets)"
            for (p in m.parts) {
                if (p.label.en.isBlank() || p.label.vi.isBlank()) problems += "$key.${p.name}: label missing a language"
                p.check?.let { if (it.en.isBlank() || it.vi.isBlank()) problems += "$key.${p.name}: check missing a language" }
                for (pr in p.prims) {
                    val n = pr.positions.size / 3
                    if (pr.indices.isEmpty() || pr.indices.size % 3 != 0) problems += "$key.${p.name}: bad index count"
                    if (pr.indices.any { it !in 0 until n }) problems += "$key.${p.name}: index out of range"
                    if (pr.normals.any { it.isNaN() } || pr.positions.any { it.isNaN() }) problems += "$key.${p.name}: NaN"
                }
            }
            if (m.parts.none { it.check != null }) problems += "$key: no part says what to check"
        }
        assertTrue(problems.joinToString("\n"), problems.isEmpty())
    }

    @Test
    fun glbReaderMatchesPython() {
        val ref = JSONObject(fixture("model_stats.json").readText())
        val m = Glb.parse(fixture("battery.glb").readBytes(), "battery")
        val problems = compare("battery (GLB)", m, ref.getJSONObject("battery"))
        assertTrue(problems.joinToString("\n"), problems.isEmpty())
    }
}
