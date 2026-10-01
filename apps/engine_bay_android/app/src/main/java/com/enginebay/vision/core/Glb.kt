package com.enginebay.vision.core

import org.json.JSONArray
import org.json.JSONObject
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** One drawable piece of a part: triangles with one material. Positions and normals are in model space (node transforms applied). */
class Prim(
    val positions: FloatArray,
    val normals: FloatArray,
    val indices: IntArray,
    /** RGBA, alpha < 1 = see-through (tanks: the fluid level shows through) */
    val color: FloatArray,
    val metallic: Float,
    val roughness: Float,
) {
    val transparent get() = color[3] < 0.999f
}

/** A named part of a component model, e.g. the "positive_post" of the battery. */
class ModelPart(
    val name: String,
    val label: Text,
    /** what to inspect on this part (draft knowledge), or null */
    val check: Text?,
    /** exploded-view offset in model units (zero = the part stays in place) */
    val explode: FloatArray,
    val prims: List<Prim>,
) {
    val min = FloatArray(3) { Float.MAX_VALUE }
    val max = FloatArray(3) { -Float.MAX_VALUE }

    init {
        for (p in prims) for (i in p.positions.indices) {
            val a = i % 3
            min[a] = min(min[a], p.positions[i]); max[a] = max(max[a], p.positions[i])
        }
    }

    val center get() = FloatArray(3) { (min[it] + max[it]) / 2 }
    val explodes get() = explode.any { it != 0f }
}

/** A 3D component model from assets/models/<component>.glb (scripts/deployment/build_component_models.py). */
class Model3D(
    val component: String,
    /** hero camera angle: yaw, pitch in degrees */
    val view: FloatArray,
    val parts: List<ModelPart>,
) {
    val min = FloatArray(3) { i -> parts.minOfOrNull { it.min[i] } ?: 0f }
    val max = FloatArray(3) { i -> parts.maxOfOrNull { it.max[i] } ?: 0f }
    val center get() = FloatArray(3) { (min[it] + max[it]) / 2 }
    val triangles get() = parts.sumOf { p -> p.prims.sumOf { it.indices.size / 3 } }

    /** Radius of the bounding sphere around [center]. */
    fun radius(): Float {
        val c = center
        var r = 0f
        for (p in parts) for (pr in p.prims) {
            val v = pr.positions
            var i = 0
            while (i < v.size) {
                val dx = v[i] - c[0]; val dy = v[i + 1] - c[1]; val dz = v[i + 2] - c[2]
                r = max(r, dx * dx + dy * dy + dz * dz)
                i += 3
            }
        }
        return sqrt(r)
    }
}

/**
 * Minimal binary glTF 2.0 reader for the app's component models (and any similar, uncompressed GLB): triangle meshes
 * with POSITION / NORMAL, optional indices, base-colour / metallic / roughness factors, node hierarchy with
 * matrix or TRS. Node extras {"label", "check", "explode"} and scene extras {"component", "view"} are read when present.
 */
object Glb {
    private const val MAGIC = 0x46546C67
    private const val JSON_CHUNK = 0x4E4F534A
    private const val BIN_CHUNK = 0x004E4942

    fun parse(bytes: ByteArray, fallbackName: String = "model"): Model3D {
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bb.getInt(0) == MAGIC) { "not a GLB file" }
        val length = bb.getInt(8)
        var off = 12
        var json: JSONObject? = null
        var bin: ByteBuffer? = null
        while (off + 8 <= length) {
            val len = bb.getInt(off)
            val type = bb.getInt(off + 4)
            if (type == JSON_CHUNK) json = JSONObject(String(bytes, off + 8, len, Charsets.UTF_8))
            else if (type == BIN_CHUNK) bin = ByteBuffer.wrap(bytes, off + 8, len).slice().order(ByteOrder.LITTLE_ENDIAN)
            off += 8 + len
        }
        val js = requireNotNull(json) { "GLB without JSON chunk" }
        js.optJSONArray("extensionsRequired")?.let { require(it.length() == 0) { "compressed glTF is not supported: $it" } }
        return Reader(js, bin).read(fallbackName)
    }

    private class Reader(val js: JSONObject, val bin: ByteBuffer?) {
        val accessors: JSONArray = js.optJSONArray("accessors") ?: JSONArray()
        val views: JSONArray = js.optJSONArray("bufferViews") ?: JSONArray()
        val materials: JSONArray = js.optJSONArray("materials") ?: JSONArray()
        val meshes: JSONArray = js.optJSONArray("meshes") ?: JSONArray()
        val nodes: JSONArray = js.optJSONArray("nodes") ?: JSONArray()
        val parts = ArrayList<ModelPart>()

        fun read(fallbackName: String): Model3D {
            val scenes = js.optJSONArray("scenes")
            val scene = scenes?.optJSONObject(js.optInt("scene", 0))
            val roots = scene?.optJSONArray("nodes")?.let { a -> List(a.length()) { a.getInt(it) } } ?: (0 until nodes.length()).toList()
            for (r in roots) walk(r, identity())
            val extras = scene?.optJSONObject("extras")
            val view = extras?.optJSONArray("view")?.let { floatArrayOf(it.optDouble(0, 30.0).toFloat(), it.optDouble(1, 20.0).toFloat()) }
                ?: floatArrayOf(30f, 20f)
            return Model3D(extras?.optString("component")?.ifBlank { null } ?: fallbackName, view, parts)
        }

        private fun walk(index: Int, parent: FloatArray) {
            val node = nodes.getJSONObject(index)
            val m = mul(parent, local(node))
            if (node.has("mesh")) {
                val mesh = meshes.getJSONObject(node.getInt("mesh"))
                val name = node.optString("name").ifBlank { mesh.optString("name").ifBlank { "part${parts.size + 1}" } }
                val prims = ArrayList<Prim>()
                val pa = mesh.getJSONArray("primitives")
                for (i in 0 until pa.length()) prim(pa.getJSONObject(i), m)?.let(prims::add)
                if (prims.isNotEmpty()) {
                    val ex = node.optJSONObject("extras")
                    val label = text(ex?.opt("label")) ?: Text(name, name)
                    val e = ex?.optJSONArray("explode")
                    val explode = if (e == null) FloatArray(3) else {
                        val v = floatArrayOf(e.optDouble(0).toFloat(), e.optDouble(1).toFloat(), e.optDouble(2).toFloat())
                        FloatArray(3) { r -> m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] }
                    }
                    parts += ModelPart(name, label, text(ex?.opt("check")), explode, prims)
                }
            }
            node.optJSONArray("children")?.let { c -> for (i in 0 until c.length()) walk(c.getInt(i), m) }
        }

        private fun text(v: Any?): Text? = when (v) {
            is JSONObject -> Text(v.optString("en"), v.optString("vi")).takeIf { it.en.isNotBlank() || it.vi.isNotBlank() }
            is String -> if (v.isBlank()) null else Text(v, v)
            else -> null
        }

        private fun prim(p: JSONObject, m: FloatArray): Prim? {
            if (p.optInt("mode", 4) != 4) return null  // triangles only
            val attrs = p.getJSONObject("attributes")
            if (!attrs.has("POSITION")) return null
            val pos = floats(attrs.getInt("POSITION"), 3)
            val n = pos.size / 3
            val idx = if (p.has("indices")) ints(p.getInt("indices")) else IntArray(n) { it }
            val nrm = if (attrs.has("NORMAL")) floats(attrs.getInt("NORMAL"), 3) else faceNormals(pos, idx)
            // bake the node transform: positions by m, normals by the inverse-transpose of its 3x3 part
            val nm = normalMatrix(m)
            val outP = FloatArray(pos.size)
            val outN = FloatArray(pos.size)
            for (i in 0 until n) {
                val x = pos[3 * i]; val y = pos[3 * i + 1]; val z = pos[3 * i + 2]
                for (r in 0..2) outP[3 * i + r] = m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r]
                val a = nrm[3 * i]; val b = nrm[3 * i + 1]; val c = nrm[3 * i + 2]
                var nx = nm[0] * a + nm[3] * b + nm[6] * c
                var ny = nm[1] * a + nm[4] * b + nm[7] * c
                var nz = nm[2] * a + nm[5] * b + nm[8] * c
                val l = sqrt(nx * nx + ny * ny + nz * nz).takeIf { it > 1e-12f } ?: 1f
                nx /= l; ny /= l; nz /= l
                outN[3 * i] = nx; outN[3 * i + 1] = ny; outN[3 * i + 2] = nz
            }
            val (color, metal, rough) = material(p.optInt("material", -1))
            return Prim(outP, outN, idx, color, metal, rough)
        }

        private fun material(i: Int): Triple<FloatArray, Float, Float> {
            val mat = if (i >= 0) materials.optJSONObject(i) else null
            val pbr = mat?.optJSONObject("pbrMetallicRoughness")
            val c = pbr?.optJSONArray("baseColorFactor")
            val color = if (c == null) floatArrayOf(0.6f, 0.62f, 0.65f, 1f) else FloatArray(4) { c.optDouble(it, 1.0).toFloat() }
            if (mat?.optString("alphaMode") != "BLEND") color[3] = 1f
            return Triple(color, pbr?.optDouble("metallicFactor", 1.0)?.toFloat() ?: 0f, pbr?.optDouble("roughnessFactor", 1.0)?.toFloat() ?: 0.6f)
        }

        private fun data(acc: JSONObject): Pair<ByteBuffer, Int> {
            val buf = requireNotNull(bin) { "external buffers are not supported" }
            val view = views.getJSONObject(acc.getInt("bufferView"))
            val start = view.optInt("byteOffset", 0) + acc.optInt("byteOffset", 0)
            return buf to start
        }

        private fun floats(a: Int, comps: Int): FloatArray {
            val acc = accessors.getJSONObject(a)
            require(acc.getInt("componentType") == 5126) { "only float vertex data is supported" }
            val count = acc.getInt("count")
            val (buf, start) = data(acc)
            val stride = views.getJSONObject(acc.getInt("bufferView")).optInt("byteStride", 0).takeIf { it > 0 } ?: (4 * comps)
            return FloatArray(count * comps) { i -> buf.getFloat(start + (i / comps) * stride + (i % comps) * 4) }
        }

        private fun ints(a: Int): IntArray {
            val acc = accessors.getJSONObject(a)
            val count = acc.getInt("count")
            val (buf, start) = data(acc)
            return when (acc.getInt("componentType")) {
                5121 -> IntArray(count) { buf.get(start + it).toInt() and 0xFF }
                5123 -> IntArray(count) { buf.getShort(start + 2 * it).toInt() and 0xFFFF }
                5125 -> IntArray(count) { buf.getInt(start + 4 * it) }
                else -> error("unsupported index type")
            }
        }

        private fun faceNormals(pos: FloatArray, idx: IntArray): FloatArray {
            val n = FloatArray(pos.size)
            var t = 0
            while (t + 2 < idx.size) {
                val a = idx[t]; val b = idx[t + 1]; val c = idx[t + 2]
                val ux = pos[3 * b] - pos[3 * a]; val uy = pos[3 * b + 1] - pos[3 * a + 1]; val uz = pos[3 * b + 2] - pos[3 * a + 2]
                val vx = pos[3 * c] - pos[3 * a]; val vy = pos[3 * c + 1] - pos[3 * a + 1]; val vz = pos[3 * c + 2] - pos[3 * a + 2]
                val nx = uy * vz - uz * vy; val ny = uz * vx - ux * vz; val nz = ux * vy - uy * vx
                for (v in intArrayOf(a, b, c)) { n[3 * v] += nx; n[3 * v + 1] += ny; n[3 * v + 2] += nz }
                t += 3
            }
            return n
        }

        /** Column-major 4x4 of a node: "matrix", or translation * rotation * scale. */
        private fun local(node: JSONObject): FloatArray {
            node.optJSONArray("matrix")?.let { a -> return FloatArray(16) { a.optDouble(it).toFloat() } }
            val t = node.optJSONArray("translation")
            val r = node.optJSONArray("rotation")
            val s = node.optJSONArray("scale")
            val m = identity()
            if (r != null) {
                val x = r.optDouble(0).toFloat(); val y = r.optDouble(1).toFloat(); val z = r.optDouble(2).toFloat(); val w = r.optDouble(3, 1.0).toFloat()
                m[0] = 1 - 2 * (y * y + z * z); m[1] = 2 * (x * y + z * w); m[2] = 2 * (x * z - y * w)
                m[4] = 2 * (x * y - z * w); m[5] = 1 - 2 * (x * x + z * z); m[6] = 2 * (y * z + x * w)
                m[8] = 2 * (x * z + y * w); m[9] = 2 * (y * z - x * w); m[10] = 1 - 2 * (x * x + y * y)
            }
            if (s != null) for (c in 0..2) { val f = s.optDouble(c, 1.0).toFloat(); for (r2 in 0..2) m[4 * c + r2] *= f }
            if (t != null) for (r2 in 0..2) m[12 + r2] = t.optDouble(r2).toFloat()
            return m
        }

        private fun identity() = FloatArray(16).also { it[0] = 1f; it[5] = 1f; it[10] = 1f; it[15] = 1f }

        private fun mul(a: FloatArray, b: FloatArray) = FloatArray(16) { i ->
            val c = i / 4; val r = i % 4
            a[r] * b[4 * c] + a[4 + r] * b[4 * c + 1] + a[8 + r] * b[4 * c + 2] + a[12 + r] * b[4 * c + 3]
        }

        /** Inverse-transpose of the upper 3x3 (column-major 3x3 result). */
        private fun normalMatrix(m: FloatArray): FloatArray {
            val a = m[0]; val b = m[4]; val c = m[8]
            val d = m[1]; val e = m[5]; val f = m[9]
            val g = m[2]; val h = m[6]; val i = m[10]
            val co = floatArrayOf(e * i - f * h, -(d * i - f * g), d * h - e * g,
                -(b * i - c * h), a * i - c * g, -(a * h - b * g),
                b * f - c * e, -(a * f - c * d), a * e - b * d)
            // co = cofactor matrix (row-major) = inverse-transpose * det: only the sign of det matters for normals
            val sign = if (a * co[0] + b * co[1] + c * co[2] < 0) -1f else 1f
            return floatArrayOf(co[0], co[3], co[6], co[1], co[4], co[7], co[2], co[5], co[8]).also { for (k in it.indices) it[k] *= sign }
        }
    }
}
