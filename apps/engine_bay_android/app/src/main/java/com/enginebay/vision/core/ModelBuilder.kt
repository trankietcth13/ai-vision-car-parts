package com.enginebay.vision.core

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Builds a component's 3D model ON THE DEVICE from its primitive spec (assets/models/<component>.json, written by
 * scripts/deployment/build_component_models.py). Every primitive is a line-by-line port of tools/model3d/build_model.py
 * and partkit.py (same sampling, same triangle order, same orientation fixes); ComponentModelsTest checks the result
 * against the Python geometry (fixtures/model_stats.json).
 *
 * Geometry is computed in double precision as a triangle soup (9 doubles per triangle), then shaded like the Python
 * GLB writer (crease-angle corner normals, 35°) and welded into indexed float buffers.
 */
object ModelBuilder {
    /** Parse and build a whole model. */
    fun build(json: String): Model3D = build(JSONObject(json))

    fun build(spec: JSONObject): Model3D {
        val parts = ArrayList<ModelPart>()
        val pa = spec.getJSONArray("parts")
        for (i in 0 until pa.length()) parts += part(pa.getJSONObject(i))
        val v = spec.optJSONArray("view")
        val view = floatArrayOf(v?.optDouble(0, 30.0)?.toFloat() ?: 30f, v?.optDouble(1, 20.0)?.toFloat() ?: 20f)
        return Model3D(spec.optString("component", "model"), view, parts)
    }

    private fun text(v: Any?): Text? = (v as? JSONObject)?.let { Text(it.optString("en"), it.optString("vi")) }

    private fun part(p: JSONObject): ModelPart {
        // one primitive per material, in first-use order (like the GLB writer)
        val groups = LinkedHashMap<String, Pair<DoubleArrayList, DoubleArray>>()
        val shapes = p.getJSONArray("shapes")
        for (i in 0 until shapes.length()) {
            val s = shapes.getJSONObject(i)
            val color = s.optString("color", p.getString("color"))
            val mat = doubleArrayOf(s.optDouble("metallic", p.optDouble("metallic", 0.0)), s.optDouble("roughness", p.optDouble("roughness", 0.5)),
                s.optDouble("alpha", p.optDouble("alpha", 1.0)))
            val key = "$color/${mat.joinToString("/")}"
            val g = groups.getOrPut(key) { DoubleArrayList() to doubleArrayOf(*hex(color), *mat) }
            g.first.addAll(shapeSoup(s))
        }
        val prims = groups.values.map { (soup, m) -> shade(soup.toArray(), floatArrayOf(m[0].toFloat(), m[1].toFloat(), m[2].toFloat(), m[5].toFloat()), m[3].toFloat(), m[4].toFloat()) }
        val e = p.optJSONArray("explode")
        val explode = if (e == null) FloatArray(3) else FloatArray(3) { e.optDouble(it).toFloat() }
        val name = p.getString("name")
        return ModelPart(name, text(p.opt("label")) ?: Text(name, name), text(p.opt("check")), explode, prims)
    }

    private fun hex(h: String): DoubleArray {
        val s = h.trimStart('#')
        return DoubleArray(3) { Integer.parseInt(s.substring(2 * it, 2 * it + 2), 16) / 255.0 }
    }

    // ------------------------------------------------------------------------------------------------ soups
    /** Growable list of doubles (triangle soups: 9 per triangle). */
    class DoubleArrayList(cap: Int = 1024) {
        var data = DoubleArray(cap)
        var size = 0
        fun add(v: Double) { if (size == data.size) data = data.copyOf(size * 2); data[size++] = v }
        fun addAll(a: DoubleArray) { while (size + a.size > data.size) data = data.copyOf(data.size * 2); a.copyInto(data, size); size += a.size }
        fun tri(a: DoubleArray, b: DoubleArray, c: DoubleArray) { for (p in arrayOf(a, b, c)) { add(p[0]); add(p[1]); add(p[2]) } }
        fun toArray() = data.copyOf(size)
    }

    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    private fun sub(a: DoubleArray, b: DoubleArray) = doubleArrayOf(a[0] - b[0], a[1] - b[1], a[2] - b[2])
    private fun dot(a: DoubleArray, b: DoubleArray) = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    private fun len(a: DoubleArray) = sqrt(dot(a, a))

    /** Unnormalised normal of triangle t of a soup. */
    private fun triNormal(T: DoubleArray, t: Int): DoubleArray {
        val o = 9 * t
        val ux = T[o + 3] - T[o]; val uy = T[o + 4] - T[o + 1]; val uz = T[o + 5] - T[o + 2]
        val vx = T[o + 6] - T[o]; val vy = T[o + 7] - T[o + 1]; val vz = T[o + 8] - T[o + 2]
        return doubleArrayOf(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
    }

    private fun swap12(T: DoubleArray, t: Int) {
        val o = 9 * t
        for (k in 0..2) { val x = T[o + 3 + k]; T[o + 3 + k] = T[o + 6 + k]; T[o + 6 + k] = x }
    }

    /** build_model._fix_volume: flip every triangle when the signed volume is negative. */
    private fun fixVolume(T: DoubleArray): DoubleArray {
        val n = T.size / 9
        var vol = 0.0
        for (t in 0 until n) {
            val o = 9 * t
            val cx = T[o + 4] * T[o + 8] - T[o + 5] * T[o + 7]
            val cy = T[o + 5] * T[o + 6] - T[o + 3] * T[o + 8]
            val cz = T[o + 3] * T[o + 7] - T[o + 4] * T[o + 6]
            vol += T[o] * cx + T[o + 1] * cy + T[o + 2] * cz
        }
        if (vol < 0) for (t in 0 until n) swap12(T, t)
        return T
    }

    /** Keep triangles whose |cross| > eps. */
    private fun dropDegenerate(T: DoubleArray, eps: Double): DoubleArray {
        val out = DoubleArrayList(T.size)
        for (t in 0 until T.size / 9) if (len(triNormal(T, t)) > eps) for (k in 0 until 9) out.add(T[9 * t + k])
        return out.toArray()
    }

    // ------------------------------------------------------------------------------------------------ primitives
    fun revolve(profile: List<DoubleArray>, seg: Int, closed: Boolean): DoubleArray {
        val prof = profile.map { it.copyOf() }.toMutableList()
        if (!closed) {
            if (prof[0][0] > 1e-9) prof.add(0, doubleArrayOf(0.0, prof[0][1]))
            if (prof.last()[0] > 1e-9) prof.add(doubleArrayOf(0.0, prof.last()[1]))
        }
        val n = prof.size
        val step = 2 * Math.PI / seg
        val cs = DoubleArray(seg) { cos(it * step) }
        val sn = DoubleArray(seg) { sin(it * step) }
        fun v(i: Int, j: Int) = doubleArrayOf(prof[i][0] * cs[j], prof[i][1], prof[i][0] * sn[j])
        val out = DoubleArrayList()
        val rows = if (closed) n else n - 1
        for (i in 0 until rows) {
            val i2 = (i + 1) % n
            for (j in 0 until seg) {
                val j2 = (j + 1) % seg
                val a = v(i, j); val b = v(i, j2); val c = v(i2, j2); val d = v(i2, j)
                out.tri(a, c, b); out.tri(a, d, c)
            }
        }
        return fixVolume(dropDegenerate(out.toArray(), 1e-12))
    }

    fun rbox(size: DoubleArray, radius: Double, segments: Int?): DoubleArray {
        val h = DoubleArray(3) { size[it] / 2 }
        val r = if (radius != 0.0) minOf(radius, h[0], h[1], h[2]) else 0.0
        val k = segments ?: 7
        fun samples(hh: Double): DoubleArray {
            if (r <= 0) return doubleArrayOf(-hh, hh)
            val a = DoubleArray(2 * (k + 1)) { i ->
                val f = (i % (k + 1)).toDouble() / k
                if (i <= k) -hh + r * f else (hh - r) + r * f
            }
            return a.map { Math.rint(it * 1e12) / 1e12 }.distinct().sorted().toDoubleArray()
        }
        val S = Array(3) { samples(h[it]) }
        val out = DoubleArrayList()
        for (ax in 0..2) {
            val u = (ax + 1) % 3; val v = (ax + 2) % 3
            val gu = S[u]; val gv = S[v]
            for (sgn in intArrayOf(-1, 1)) {
                fun g(i: Int, j: Int) = DoubleArray(3).also { it[ax] = sgn * h[ax]; it[u] = gu[i]; it[v] = gv[j] }
                for (i in 0 until gu.size - 1) for (j in 0 until gv.size - 1) out.tri(g(i, j), g(i + 1, j), g(i + 1, j + 1))
                for (i in 0 until gu.size - 1) for (j in 0 until gv.size - 1) out.tri(g(i, j), g(i + 1, j + 1), g(i, j + 1))
            }
        }
        val T = out.toArray()
        if (r > 0) {
            val inner = DoubleArray(3) { h[it] - r }
            var p = 0
            while (p < T.size) {
                val q = DoubleArray(3) { T[p + it].coerceIn(-inner[it], inner[it]) }
                val d = DoubleArray(3) { T[p + it] - q[it] }
                val l = len(d)
                if (l > 1e-12) for (c in 0..2) T[p + c] = q[c] + d[c] / max(l, 1e-12) * r
                p += 3
            }
        }
        val keep = DoubleArrayList(T.size)
        for (t in 0 until T.size / 9) {
            val n = triNormal(T, t)
            val o = 9 * t
            val c = DoubleArray(3) { (T[o + it] + T[o + 3 + it] + T[o + 6 + it]) / 3 }
            if (dot(n, c) < 0) swap12(T, t)
            if (len(n) > 1e-12) for (m in 0 until 9) keep.add(T[o + m])
        }
        return keep.toArray()
    }

    fun cylinder(r: Double, h: Double, r2In: Double?, hole: Double?, chamfer: Double, seg: Int): DoubleArray {
        val r2 = r2In ?: r
        val y0 = -h / 2; val y1 = h / 2
        val c = chamfer
        val inner = if (hole != null && hole != 0.0) hole else 0.0
        val raw = listOf(doubleArrayOf(inner, y0), doubleArrayOf(r - c, y0), doubleArrayOf(r, y0 + c), doubleArrayOf(r2, y1 - c),
            doubleArrayOf(r2 - c, y1), doubleArrayOf(inner, y1))
        val prof = raw.filterIndexed { k, p -> k == 0 || !(p[0] == raw[k - 1][0] && p[1] == raw[k - 1][1]) }
        return revolve(prof, seg, closed = hole != null && hole != 0.0)
    }

    fun torus(R: Double, r: Double, seg: Int, seg2: Int): DoubleArray =
        revolve(List(seg2) { val t = 2 * Math.PI * it / seg2; doubleArrayOf(R + r * cos(t), r * sin(t)) }, seg, closed = true)

    fun sphere(r: Double, seg: Int): DoubleArray {
        val n = seg / 2 + 1
        return revolve(List(n) { val t = -Math.PI / 2 + Math.PI * it / (n - 1); doubleArrayOf(r * cos(t), r * sin(t)) }, seg, closed = false)
    }

    private fun hull(ptsIn: List<DoubleArray>): List<DoubleArray> {
        val pts = ptsIn.map { doubleArrayOf(Math.rint(it[0] * 1e9) / 1e9, Math.rint(it[1] * 1e9) / 1e9) }
            .distinctBy { it[0] to it[1] }.sortedWith(compareBy({ it[0] }, { it[1] }))
        if (pts.size < 3) return pts
        fun cr(o: DoubleArray, a: DoubleArray, b: DoubleArray) = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        val lo = ArrayList<DoubleArray>(); val up = ArrayList<DoubleArray>()
        for (p in pts) { while (lo.size >= 2 && cr(lo[lo.size - 2], lo.last(), p) <= 0) lo.removeAt(lo.size - 1); lo += p }
        for (p in pts.asReversed()) { while (up.size >= 2 && cr(up[up.size - 2], up.last(), p) <= 0) up.removeAt(up.size - 1); up += p }
        return lo.dropLast(1) + up.dropLast(1)
    }

    private fun signedArea(P: List<DoubleArray>): Double {
        var s = 0.0
        for (i in P.indices) { val q = P[(i + 1) % P.size]; s += P[i][0] * q[1] - q[0] * P[i][1] }
        return 0.5 * s
    }

    private fun close(a: DoubleArray, b: DoubleArray) =  // numpy.allclose(a, b) with its default tolerances
        abs(a[0] - b[0]) <= 1e-8 + 1e-5 * abs(b[0]) && abs(a[1] - b[1]) <= 1e-8 + 1e-5 * abs(b[1])

    private fun earclip(P: List<DoubleArray>): List<IntArray> {
        val idx = P.indices.toMutableList()
        val tris = ArrayList<IntArray>()
        fun inside(p: DoubleArray, a: DoubleArray, b: DoubleArray, c: DoubleArray): Boolean {
            val d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
            val d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
            val d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
            val e = 1e-12
            return (d1 < -e && d2 < -e && d3 < -e) || (d1 > e && d2 > e && d3 > e)
        }
        var guard = 0
        while (idx.size > 3 && guard < 100000) {
            guard++
            val m = idx.size
            var clipped = false
            for (k in 0 until m) {
                val i0 = idx[(k - 1 + m) % m]; val i1 = idx[k]; val i2 = idx[(k + 1) % m]
                val a = P[i0]; val b = P[i1]; val c = P[i2]
                if ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 1e-12) continue
                val blocked = idx.any { j ->
                    j != i0 && j != i1 && j != i2 && !(close(P[j], a) || close(P[j], b) || close(P[j], c)) && inside(P[j], a, b, c)
                }
                if (blocked) continue
                tris += intArrayOf(i0, i1, i2)
                idx.removeAt(k)
                clipped = true
                break
            }
            if (!clipped) { tris += intArrayOf(idx.last(), idx[0], idx[1]); idx.removeAt(0) }
        }
        if (idx.size == 3) tris += intArrayOf(idx[0], idx[1], idx[2])
        return tris
    }

    fun extrude(h: Double, points: List<DoubleArray>?, hullCircles: List<DoubleArray>?, holes: List<DoubleArray>, seg: Int): DoubleArray {
        var outer: List<DoubleArray> = if (hullCircles != null) {
            val cp = ArrayList<DoubleArray>()
            for (c in hullCircles) for (i in 0 until seg) {
                val a = 2 * Math.PI * i / seg
                cp += doubleArrayOf(c[0] + c[2] * cos(a), c[1] + c[2] * sin(a))
            }
            hull(cp)
        } else points!!
        if (signedArea(outer) < 0) outer = outer.asReversed()
        val hl = holes.map { c ->
            List(seg) { k -> val a = 2 * Math.PI * (seg - 1 - k) / seg; doubleArrayOf(c[0] + c[2] * cos(a), c[1] + c[2] * sin(a)) }
        }
        val loops = listOf(outer) + hl
        // bridge the holes into the outer loop (rightmost hole first), then ear-clip
        var poly: List<DoubleArray> = outer
        for (H in hl.sortedBy { q -> -q.maxOf { it[0] } }) {
            var j = 0
            for (k in H.indices) if (H[k][0] > H[j][0]) j = k
            var i = 0
            var best = Double.MAX_VALUE
            for (k in poly.indices) {
                val d = Math.hypot(poly[k][0] - H[j][0], poly[k][1] - H[j][1])
                if (d < best) { best = d; i = k }
            }
            val hr = H.subList(j, H.size) + H.subList(0, j + 1)
            poly = poly.subList(0, i + 1) + hr + poly.subList(i, poly.size)
        }
        val cap = earclip(poly)
        val y0 = -h / 2; val y1 = h / 2
        fun to3(p: DoubleArray, y: Double) = doubleArrayOf(p[0], y, p[1])
        val tris = DoubleArrayList()
        val targets = ArrayList<DoubleArray>()
        for (t in cap) {
            tris.tri(to3(poly[t[0]], y1), to3(poly[t[1]], y1), to3(poly[t[2]], y1)); targets += doubleArrayOf(0.0, 1.0, 0.0)
            tris.tri(to3(poly[t[0]], y0), to3(poly[t[1]], y0), to3(poly[t[2]], y0)); targets += doubleArrayOf(0.0, -1.0, 0.0)
        }
        for (L in loops) for (k in L.indices) {
            val p = L[k]; val q = L[(k + 1) % L.size]
            val out = doubleArrayOf(q[1] - p[1], 0.0, -(q[0] - p[0]))
            tris.tri(to3(p, y0), to3(q, y0), to3(q, y1)); targets += out
            tris.tri(to3(p, y0), to3(q, y1), to3(p, y1)); targets += out
        }
        val T = tris.toArray()
        val keep = DoubleArrayList(T.size)
        for (t in 0 until T.size / 9) {
            val n = triNormal(T, t)
            if (dot(n, targets[t]) < 0) swap12(T, t)
            if (len(n) > 1e-12) for (m in 0 until 9) keep.add(T[9 * t + m])
        }
        return keep.toArray()
    }

    // ------------------------------------------------------------------------------------------------ swept shapes (partkit)
    fun catmull(points: List<DoubleArray>, n: Int, closed: Boolean): List<DoubleArray> {
        val P = points
        if (P.size < 3) {
            val cnt = max(2, n)
            return List(cnt) { i -> val t = i.toDouble() / (cnt - 1); DoubleArray(3) { P[0][it] + (P.last()[it] - P[0][it]) * t } }
        }
        val Q = if (closed) listOf(P.last()) + P + listOf(P[0], P[1])
        else listOf(DoubleArray(3) { 2 * P[0][it] - P[1][it] }) + P + listOf(DoubleArray(3) { 2 * P.last()[it] - P[P.size - 2][it] })
        val out = ArrayList<DoubleArray>()
        val segs = if (closed) P.size else P.size - 1
        for (i in 0 until segs) {
            val p0 = Q[i]; val p1 = Q[i + 1]; val p2 = Q[i + 2]; val p3 = Q[i + 3]
            for (s in 0 until n) {
                val t = s.toDouble() / n; val t2 = t * t; val t3 = t2 * t
                out += DoubleArray(3) {
                    0.5 * ((2 * p1[it]) + (-p0[it] + p2[it]) * t + (2 * p0[it] - 5 * p1[it] + 4 * p2[it] - p3[it]) * t2 +
                        (-p0[it] + 3 * p1[it] - 3 * p2[it] + p3[it]) * t3)
                }
            }
        }
        if (!closed) out += P.last()
        return out
    }

    fun sweep(path: List<DoubleArray>, profile: List<DoubleArray>, closed: Boolean, up: DoubleArray?, scale: DoubleArray?): DoubleArray {
        val n = path.size
        val tg = Array(n) { i ->
            val d = when {
                closed -> sub(path[(i + 1) % n], path[(i - 1 + n) % n])
                i == 0 -> sub(path[1], path[0])
                i == n - 1 -> sub(path[n - 1], path[n - 2])
                else -> DoubleArray(3) { (path[i + 1][it] - path[i - 1][it]) / 2 }
            }
            val l = len(d) + 1e-12
            DoubleArray(3) { d[it] / l }
        }
        var ref = up ?: doubleArrayOf(0.0, 1.0, 0.0)
        if (abs(dot(ref, tg[0])) > 0.9) ref = doubleArrayOf(1.0, 0.0, 0.0)
        val nrm = arrayOfNulls<DoubleArray>(n)
        val n0 = cross(cross(tg[0], ref), tg[0])
        nrm[0] = DoubleArray(3) { n0[it] / len(n0) }
        for (i in 1 until n) {
            val prev = nrm[i - 1]!!
            val k = dot(prev, tg[i])
            val v = DoubleArray(3) { prev[it] - k * tg[i][it] }
            val l = len(v) + 1e-12
            nrm[i] = DoubleArray(3) { v[it] / l }
        }
        if (closed) {  // spread the transport twist over the loop so the seam closes
            val b0 = cross(tg[0], nrm[0]!!)
            val last = nrm[n - 1]!!
            val k = dot(last, tg[0])
            val v = DoubleArray(3) { last[it] - k * tg[0][it] }
            val twist = atan2(dot(v, b0), dot(v, nrm[0]!!))
            for (i in 0 until n) {
                val a = -twist * i / n
                val b = cross(tg[i], nrm[i]!!)
                val ni = nrm[i]!!
                nrm[i] = DoubleArray(3) { ni[it] * cos(a) + b[it] * sin(a) }
            }
        }
        val kp = profile.size
        val R = Array(n) { i ->
            val b = cross(tg[i], nrm[i]!!)
            val sc = scale?.get(i) ?: 1.0
            Array(kp) { j -> DoubleArray(3) { path[i][it] + sc * (profile[j][0] * nrm[i]!![it] + profile[j][1] * b[it]) } }
        }
        val out = DoubleArrayList()
        val rows = if (closed) n else n - 1
        for (i in 0 until rows) {
            val i2 = (i + 1) % n
            for (j in 0 until kp) {
                val j2 = (j + 1) % kp
                out.tri(R[i][j], R[i][j2], R[i2][j2]); out.tri(R[i][j], R[i2][j2], R[i2][j])
            }
        }
        if (!closed) for (i in intArrayOf(0, n - 1)) {
            val cen = DoubleArray(3) { c -> R[i].sumOf { it[c] } / kp }
            for (j in 0 until kp) out.tri(cen, R[i][j], R[i][(j + 1) % kp])
        }
        val T = out.toArray()
        val keep = DoubleArrayList(T.size)
        for (t in 0 until T.size / 9) if (len(triNormal(T, t)) > 1e-10) for (m in 0 until 9) keep.add(T[9 * t + m])
        return fixVolume(keep.toArray())
    }

    private fun points(a: JSONArray) = List(a.length()) { i -> a.getJSONArray(i).let { p -> DoubleArray(p.length()) { p.getDouble(it) } } }

    fun tube(s: JSONObject): DoubleArray {
        val pts = points(s.getJSONArray("points"))
        val closed = s.optBoolean("closed", false)
        val C = if (s.optBoolean("smooth", true)) catmull(pts, s.optInt("samples", 10), closed) else pts
        val sides = s.optInt("sides", 20)
        val circle = List(sides) { val a = 2 * Math.PI * it / sides; doubleArrayOf(cos(a), sin(a)) }
        val r = s.get("r")
        val scale = if (r is JSONArray) DoubleArray(C.size) { r.getDouble(it) } else DoubleArray(C.size) { (r as Number).toDouble() }
        return sweep(C, circle, closed, null, scale)
    }

    fun belt(s: JSONObject): DoubleArray {
        val th = s.getDouble("thickness"); val w = s.getDouble("width")
        val pul = points(s.getJSONArray("pulleys")).map { doubleArrayOf(it[0], it[1], it[2] + th / 2, it[3]) }
        val m = pul.size
        class Line(val p1: DoubleArray, val p2: DoubleArray, val n: DoubleArray)
        val lines = ArrayList<Line>()
        for (i in 0 until m) {
            val c1 = pul[i]; val c2 = pul[(i + 1) % m]
            val a = c1[3] * c1[2]; val b = c2[3] * c2[2]
            val dx = c2[0] - c1[0]; val dy = c2[1] - c1[1]
            val L = Math.hypot(dx, dy)
            val th2 = acos(max(-1.0, min(1.0, (a - b) / L)))
            val phi = atan2(dy, dx)
            var best = doubleArrayOf(0.0, 0.0)
            for (sg in intArrayOf(1, -1)) {
                val nv = doubleArrayOf(cos(phi + sg * th2), sin(phi + sg * th2))
                if (-nv[1] * dx + nv[0] * dy > 0) best = nv  // travel direction = n rotated +90 (counter-clockwise loop)
            }
            lines += Line(doubleArrayOf(c1[0] + a * best[0], c1[1] + a * best[1]), doubleArrayOf(c2[0] + b * best[0], c2[1] + b * best[1]), best)
        }
        val pts = ArrayList<DoubleArray>()
        for (i in 0 until m) {
            val c = pul[i]; val r = c[2]; val sd = c[3]
            val nIn = lines[(i - 1 + m) % m].n; val out = lines[i]
            val a0 = atan2(sd * nIn[1], sd * nIn[0])
            var a1 = atan2(sd * out.n[1], sd * out.n[0])
            if (sd > 0) while (a1 < a0) a1 += 2 * Math.PI else while (a1 > a0) a1 -= 2 * Math.PI
            val k = max(2, (abs(a1 - a0) / Math.toRadians(6.0)).toInt())
            for (q in 0 until k) {
                val t = a0 + (a1 - a0) * q / (k - 1)
                pts += doubleArrayOf(c[0] + r * cos(t), c[1] + r * sin(t))
            }
            for (q in 1..4) {
                val f = q / 5.0
                pts += doubleArrayOf(out.p1[0] + (out.p2[0] - out.p1[0]) * f, out.p1[1] + (out.p2[1] - out.p1[1]) * f)
            }
        }
        val path = ArrayList<DoubleArray>()
        for ((i, p) in pts.withIndex())  // drop a point equal to the one before it
            if (i == 0 || Math.hypot(p[0] - pts[i - 1][0], p[1] - pts[i - 1][1]) > 1e-6) path += doubleArrayOf(p[0], p[1], 0.0)
        val prof = listOf(doubleArrayOf(-th / 2, -w / 2), doubleArrayOf(th / 2, -w / 2), doubleArrayOf(th / 2, w / 2), doubleArrayOf(-th / 2, w / 2))
        return sweep(path, prof, closed = true, up = doubleArrayOf(0.0, 0.0, 1.0), scale = null)
    }

    // ------------------------------------------------------------------------------------------------ placement
    /** rot_xyz: Rz @ Ry @ Rx (degrees), row-major 3x3. */
    private fun rotXyz(rx: Double, ry: Double, rz: Double): DoubleArray {
        val x = Math.toRadians(rx); val y = Math.toRadians(ry); val z = Math.toRadians(rz)
        val cx = cos(x); val sx = sin(x); val cy = cos(y); val sy = sin(y); val cz = cos(z); val sz = sin(z)
        val Rx = doubleArrayOf(1.0, 0.0, 0.0, 0.0, cx, -sx, 0.0, sx, cx)
        val Ry = doubleArrayOf(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy)
        val Rz = doubleArrayOf(cz, -sz, 0.0, sz, cz, 0.0, 0.0, 0.0, 1.0)
        return mul3(mul3(Rz, Ry), Rx)
    }

    private fun mul3(a: DoubleArray, b: DoubleArray) = DoubleArray(9) { i -> val r = i / 3; val c = i % 3; a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c] }

    private fun apply(T: DoubleArray, M: DoubleArray) {
        var p = 0
        while (p < T.size) {
            val x = T[p]; val y = T[p + 1]; val z = T[p + 2]
            T[p] = M[0] * x + M[1] * y + M[2] * z
            T[p + 1] = M[3] * x + M[4] * y + M[5] * z
            T[p + 2] = M[6] * x + M[7] * y + M[8] * z
            p += 3
        }
    }

    private val AXIS = mapOf("y" to doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        "x" to rotXyz(0.0, 0.0, -90.0), "z" to rotXyz(90.0, 0.0, 0.0))

    private fun vec(s: JSONObject, key: String): DoubleArray? = s.optJSONArray(key)?.let { a -> DoubleArray(a.length()) { a.getDouble(it) } }

    /** One shape of the spec -> placed triangle soup (build_model.shape_soup / partkit.shape_soup). */
    fun shapeSoup(s: JSONObject): DoubleArray {
        val type = s.getString("type")
        val seg = s.optInt("segments", 64)
        val custom = type == "tube" || type == "belt"
        val T = when (type) {
            "rbox", "box" -> rbox(vec(s, "size")!!, if (type == "rbox") s.optDouble("radius", 0.0) else 0.0,
                if (s.has("segments") && !s.isNull("segments")) s.getInt("segments") else null)
            "cylinder" -> cylinder(s.getDouble("r"), s.getDouble("h"), if (s.has("r2") && !s.isNull("r2")) s.getDouble("r2") else null,
                if (s.has("hole") && !s.isNull("hole")) s.getDouble("hole") else null, s.optDouble("chamfer", 0.0), seg)
            "lathe" -> revolve(points(s.getJSONArray("profile")), seg, s.optBoolean("closed", false))
            "torus" -> torus(s.getDouble("R"), s.getDouble("r"), seg, s.optInt("segments2", 24))
            "sphere" -> sphere(s.getDouble("r"), seg)
            "extrude" -> extrude(s.getDouble("h"), s.optJSONArray("points")?.let(::points), s.optJSONArray("hull_circles")?.let(::points),
                s.optJSONArray("holes")?.let(::points).orEmpty(), seg)
            "tube" -> tube(s)
            "belt" -> belt(s)
            else -> error("unknown shape type $type")
        }
        if (!custom) vec(s, "scale")?.let { sc ->
            apply(T, doubleArrayOf(sc[0], 0.0, 0.0, 0.0, sc[1], 0.0, 0.0, 0.0, sc[2]))
            if (sc[0] * sc[1] * sc[2] < 0) for (t in 0 until T.size / 9) swap12(T, t)
        }
        apply(T, AXIS[s.optString("axis", "y")] ?: error("axis"))
        vec(s, "rot")?.let { apply(T, rotXyz(it[0], it[1], it[2])) }
        vec(s, "pos")?.let { pos -> var p = 0; while (p < T.size) { T[p] += pos[0]; T[p + 1] += pos[1]; T[p + 2] += pos[2]; p += 3 } }
        return T
    }

    // ------------------------------------------------------------------------------------------------ shading + welding
    /**
     * model3d.corner_normals (crease 35°): each corner averages the (area-weighted) normals of the faces that share its
     * position and are within the crease angle of its own face. Then corners with equal position and normal are welded.
     */
    fun shade(T: DoubleArray, color: FloatArray, metallic: Float, roughness: Float): Prim {
        val m = T.size / 9
        val fn = Array(m) { triNormal(T, it) }
        val fu = Array(m) { val l = len(fn[it]) + 1e-12; DoubleArray(3) { k -> fn[it][k] / l } }
        var amax = 0.0
        for (v in T) amax = max(amax, abs(v))
        val scale = 1e5 / (amax + 1e-9)
        // faces around each quantised vertex position
        val keys = LongArray(3 * m)
        val groups = HashMap<Long, IntArrayList>(3 * m)
        for (c in 0 until 3 * m) {
            val o = 3 * c
            val k = key(Math.rint(T[o] * scale).toLong(), Math.rint(T[o + 1] * scale).toLong(), Math.rint(T[o + 2] * scale).toLong())
            keys[c] = k
            groups.getOrPut(k) { IntArrayList() }.add(c / 3)
        }
        val crease = cos(Math.toRadians(35.0))
        val pos = FloatArray(9 * m)
        val nrm = FloatArray(9 * m)
        for (c in 0 until 3 * m) {
            val f = c / 3
            var ax = 0.0; var ay = 0.0; var az = 0.0
            val g = groups[keys[c]]!!
            for (q in 0 until g.size) {
                val nf = g.data[q]
                if (dot(fu[f], fu[nf]) > crease) { ax += fn[nf][0]; ay += fn[nf][1]; az += fn[nf][2] }
            }
            val l = sqrt(ax * ax + ay * ay + az * az) + 1e-12
            var nx = ax / l; var ny = ay / l; var nz = az / l
            if (sqrt(nx * nx + ny * ny + nz * nz) < 0.5) { nx = fu[f][0]; ny = fu[f][1]; nz = fu[f][2] }
            for (k in 0..2) pos[3 * c + k] = T[3 * c + k].toFloat()
            nrm[3 * c] = nx.toFloat(); nrm[3 * c + 1] = ny.toFloat(); nrm[3 * c + 2] = nz.toFloat()
        }
        // weld: same position (1e-3) and normal (1e-3) -> one vertex
        val index = HashMap<WeldKey, Int>(3 * m)
        val outP = FloatArrayList(); val outN = FloatArrayList()
        val idx = IntArray(3 * m)
        for (c in 0 until 3 * m) {
            // positions in mm (< 2000) at 1e-3 -> 21 bits per axis; normals at 1e-3 -> 11 bits per axis
            val pk = ((Math.round(pos[3 * c] * 1000.0) + (1L shl 20)) shl 42) or ((Math.round(pos[3 * c + 1] * 1000.0) + (1L shl 20)) shl 21) or
                (Math.round(pos[3 * c + 2] * 1000.0) + (1L shl 20))
            val nk = ((Math.round(nrm[3 * c] * 1000.0) + 1024) shl 22) or ((Math.round(nrm[3 * c + 1] * 1000.0) + 1024) shl 11) or
                (Math.round(nrm[3 * c + 2] * 1000.0) + 1024)
            idx[c] = index.getOrPut(WeldKey(pk, nk)) {
                for (k in 0..2) { outP.add(pos[3 * c + k]); outN.add(nrm[3 * c + k]) }
                outP.size / 3 - 1
            }
        }
        return Prim(outP.toArray(), outN.toArray(), idx, color, metallic, roughness)
    }

    /** Exact key of a quantised position (|v| <= 1e5 after scaling, 21 bits per axis). */
    private fun key(x: Long, y: Long, z: Long) = ((x + (1L shl 20)) shl 42) or ((y + (1L shl 20)) shl 21) or (z + (1L shl 20))

    private data class WeldKey(val pos: Long, val nrm: Long)

    class IntArrayList { var data = IntArray(8); var size = 0; fun add(v: Int) { if (size == data.size) data = data.copyOf(size * 2); data[size++] = v } }
    class FloatArrayList { var data = FloatArray(1024); var size = 0
        fun add(v: Float) { if (size == data.size) data = data.copyOf(size * 2); data[size++] = v }
        fun toArray() = data.copyOf(size) }
}
