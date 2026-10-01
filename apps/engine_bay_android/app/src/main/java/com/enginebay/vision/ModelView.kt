package com.enginebay.vision

import android.content.Context
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.SystemClock
import android.util.AttributeSet
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import com.enginebay.vision.core.Model3D
import com.enginebay.vision.core.ModelPart
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.ShortBuffer
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.egl.EGLDisplay
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.atan
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/** GPU-ready copy of a [Model3D]: centred, scaled to radius 1, vertex buffers built. Made off the UI thread. */
class ModelScene(val model: Model3D) {
    class Piece(val vertices: FloatBuffer, val shorts: ShortBuffer?, val ints: IntBuffer?, val count: Int,
                val positions: FloatArray, val indices: IntArray, val color: FloatArray, val metallic: Float, val roughness: Float)

    class Item(val part: ModelPart, val pieces: List<Piece>, val offset: FloatArray, val center: FloatArray) {
        val transparent = pieces.any { it.color[3] < 0.999f }
    }

    val items: List<Item>
    /** centre and radius of the bounding sphere with every part moved out (exploded view); model radius = 1 */
    val explodedCenter: FloatArray
    val explodedRadius: Float
    val minY: Float
    /** lowest point with the parts moved out */
    val explodedMinY: Float
    val canExplode = model.parts.any { it.explodes }

    init {
        val c = model.center
        val s = 1f / max(model.radius(), 1e-6f)
        items = model.parts.map { part ->
            val pieces = part.prims.map { p ->
                val n = p.positions.size / 3
                val pos = FloatArray(p.positions.size) { (p.positions[it] - c[it % 3]) * s }
                val inter = ByteBuffer.allocateDirect(n * 24).order(ByteOrder.nativeOrder()).asFloatBuffer()
                for (i in 0 until n) inter.put(pos, 3 * i, 3).put(p.normals, 3 * i, 3)
                inter.position(0)
                val small = n < 65536
                val shorts = if (small) ByteBuffer.allocateDirect(p.indices.size * 2).order(ByteOrder.nativeOrder()).asShortBuffer()
                    .also { b -> p.indices.forEach { b.put(it.toShort()) }; b.position(0) } else null
                val ints = if (small) null else ByteBuffer.allocateDirect(p.indices.size * 4).order(ByteOrder.nativeOrder()).asIntBuffer()
                    .also { b -> b.put(p.indices); b.position(0) }
                Piece(inter, shorts, ints, p.indices.size, pos, p.indices, p.color, p.metallic, p.roughness)
            }
            val pc = part.center
            Item(part, pieces, FloatArray(3) { part.explode[it] * s }, FloatArray(3) { (pc[it] - c[it]) * s })
        }
        // box around the exploded parts (and the assembled model), then the sphere around that box's centre
        val lo = FloatArray(3) { -1f }; val hi = FloatArray(3) { 1f }
        for (it in items) for (k in 0..2) {
            lo[k] = min(lo[k], (it.part.min[k] - c[k]) * s + it.offset[k])
            hi[k] = max(hi[k], (it.part.max[k] - c[k]) * s + it.offset[k])
        }
        explodedCenter = FloatArray(3) { (lo[it] + hi[it]) / 2 }
        explodedMinY = lo[1]
        var r = 1f
        for (it in items) {
            val half = FloatArray(3) { k -> (it.part.max[k] - it.part.min[k]) * s / 2 }
            val ext = sqrt(half[0] * half[0] + half[1] * half[1] + half[2] * half[2])
            val dx = it.center[0] + it.offset[0] - explodedCenter[0]
            val dy = it.center[1] + it.offset[1] - explodedCenter[1]
            val dz = it.center[2] + it.offset[2] - explodedCenter[2]
            r = max(r, sqrt(dx * dx + dy * dy + dz * dz) + ext * 0.8f)
        }
        explodedRadius = r
        minY = (model.min[1] - c[1]) * s
    }
}

/**
 * Interactive 3D view of one component model (OpenGL ES 2.0): drag to orbit, pinch to zoom, tap a part to select it,
 * double-tap to reset. The selected part pulses in the brand colour while the rest is dimmed; see-through tanks show
 * the fluid level; [exploded] slides the parts apart along their disassembly offsets.
 */
class ModelView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : GLSurfaceView(context, attrs) {
    /** UI thread: a part was tapped on the model (index into [ModelScene.items]) or -1 for empty space. */
    var onPartTapped: ((Int) -> Unit)? = null
    /** UI thread: screen position of the selected part's centre, or NaN when there is none. */
    var onAnchor: ((Float, Float) -> Unit)? = null

    @Volatile private var scene: ModelScene? = null
    @Volatile private var yaw = 30f
    @Volatile private var pitch = 20f
    @Volatile private var zoom = 1f
    @Volatile private var selected = -1
    @Volatile private var spinning = true
    @Volatile private var explodeTarget = 0f
    @Volatile private var explodeT = 0f
    @Volatile private var background = floatArrayOf(0.95f, 0.95f, 0.96f)
    @Volatile private var highlight = floatArrayOf(0.76f, 0.25f, 0.05f)

    var exploded: Boolean
        get() = explodeTarget > 0.5f
        set(v) { explodeTarget = if (v) 1f else 0f }

    init {
        setEGLContextClientVersion(2)
        setEGLConfigChooser(Chooser())
        preserveEGLContextOnPause = true
        setRenderer(ModelRenderer())
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    fun setScene(s: ModelScene, backgroundArgb: Int, highlightArgb: Int) {
        background = floatArrayOf(((backgroundArgb shr 16) and 255) / 255f, ((backgroundArgb shr 8) and 255) / 255f, (backgroundArgb and 255) / 255f)
        highlight = floatArrayOf(((highlightArgb shr 16) and 255) / 255f, ((highlightArgb shr 8) and 255) / 255f, (highlightArgb and 255) / 255f)
        scene = s
        resetView()
    }

    fun select(index: Int) {
        selected = index
        if (index < 0) post { onAnchor?.invoke(Float.NaN, Float.NaN) }
    }

    fun resetView() {
        val s = scene ?: return
        yaw = s.model.view[0]; pitch = s.model.view[1]; zoom = 1f
        spinning = true
    }

    // ------------------------------------------------------------------------------------------- camera
    private class Cam(val view: FloatArray, val proj: FloatArray, val eye: FloatArray)

    private fun camera(w: Int, h: Int, t: Float): Cam {
        val s = scene
        val aspect = w.toFloat() / max(h, 1)
        val half = Math.toRadians(15.0)
        val fit = atan(tan(half) * min(1f, aspect))  // the narrower side decides the distance
        val radius = 1f + ((s?.explodedRadius ?: 1f) - 1f) * t
        val target = FloatArray(3) { (s?.explodedCenter?.get(it) ?: 0f) * t }
        val dist = (radius / sin(fit) * 1.04 / zoom).toFloat()
        val y = Math.toRadians(yaw.toDouble()); val p = Math.toRadians(pitch.toDouble())
        val eye = floatArrayOf(target[0] + (dist * cos(p) * sin(y)).toFloat(), target[1] + (dist * sin(p)).toFloat(),
            target[2] + (dist * cos(p) * cos(y)).toFloat())
        val view = FloatArray(16)
        Matrix.setLookAtM(view, 0, eye[0], eye[1], eye[2], target[0], target[1], target[2], 0f, 1f, 0f)
        val proj = FloatArray(16)
        Matrix.perspectiveM(proj, 0, 30f, aspect, max(0.02f, dist - radius * 2.5f), dist + radius * 2.5f)
        return Cam(view, proj, eye)
    }

    // ------------------------------------------------------------------------------------------- touch
    private val gestures = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDown(e: MotionEvent) = true
        override fun onScroll(e1: MotionEvent?, e2: MotionEvent, dx: Float, dy: Float): Boolean {
            spinning = false
            val k = 180f / max(width, 1) * 1.6f
            yaw -= dx * k
            pitch = (pitch - dy * k).coerceIn(-80f, 85f)
            return true
        }
        override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
            val i = pick(e.x, e.y)
            spinning = false
            onPartTapped?.invoke(i)
            return true
        }
        override fun onDoubleTap(e: MotionEvent): Boolean { resetView(); return true }
    })
    private val scaler = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            spinning = false
            zoom = (zoom * d.scaleFactor).coerceIn(0.6f, 4f)
            return true
        }
    })

    override fun onTouchEvent(e: MotionEvent): Boolean {
        // the view sits in a scrolling bottom sheet: keep the gesture for the model
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> parent?.requestDisallowInterceptTouchEvent(true)
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> parent?.requestDisallowInterceptTouchEvent(false)
        }
        scaler.onTouchEvent(e)
        if (!scaler.isInProgress) gestures.onTouchEvent(e)
        return true
    }

    /** Index of the part under the screen point (ray / triangle test on the CPU copy), or -1. */
    private fun pick(x: Float, y: Float): Int {
        val s = scene ?: return -1
        if (width == 0 || height == 0) return -1
        val t = explodeT
        val cam = camera(width, height, t)
        val vp = FloatArray(16); val inv = FloatArray(16)
        Matrix.multiplyMM(vp, 0, cam.proj, 0, cam.view, 0)
        if (!Matrix.invertM(inv, 0, vp, 0)) return -1
        val nx = 2f * x / width - 1f; val ny = 1f - 2f * y / height
        fun unproject(z: Float): FloatArray {
            val o = FloatArray(4)
            Matrix.multiplyMV(o, 0, inv, 0, floatArrayOf(nx, ny, z, 1f), 0)
            return floatArrayOf(o[0] / o[3], o[1] / o[3], o[2] / o[3])
        }
        val a = unproject(-1f); val b = unproject(1f)
        val d = floatArrayOf(b[0] - a[0], b[1] - a[1], b[2] - a[2])
        var best = Float.MAX_VALUE
        var hit = -1
        for ((i, item) in s.items.withIndex()) {
            val ox = a[0] - item.offset[0] * t; val oy = a[1] - item.offset[1] * t; val oz = a[2] - item.offset[2] * t
            for (pc in item.pieces) {
                val p = pc.positions; val idx = pc.indices
                var k = 0
                while (k + 2 < idx.size) {
                    val d0 = rayTriangle(ox, oy, oz, d, p, idx[k], idx[k + 1], idx[k + 2])
                    if (d0 in 0f..best) { best = d0; hit = i }
                    k += 3
                }
            }
        }
        return hit
    }

    private fun rayTriangle(ox: Float, oy: Float, oz: Float, d: FloatArray, p: FloatArray, i0: Int, i1: Int, i2: Int): Float {
        val e1x = p[3 * i1] - p[3 * i0]; val e1y = p[3 * i1 + 1] - p[3 * i0 + 1]; val e1z = p[3 * i1 + 2] - p[3 * i0 + 2]
        val e2x = p[3 * i2] - p[3 * i0]; val e2y = p[3 * i2 + 1] - p[3 * i0 + 1]; val e2z = p[3 * i2 + 2] - p[3 * i0 + 2]
        val px = d[1] * e2z - d[2] * e2y; val py = d[2] * e2x - d[0] * e2z; val pz = d[0] * e2y - d[1] * e2x
        val det = e1x * px + e1y * py + e1z * pz
        if (abs(det) < 1e-9f) return -1f
        val inv = 1f / det
        val tx = ox - p[3 * i0]; val ty = oy - p[3 * i0 + 1]; val tz = oz - p[3 * i0 + 2]
        val u = (tx * px + ty * py + tz * pz) * inv
        if (u < 0f || u > 1f) return -1f
        val qx = ty * e1z - tz * e1y; val qy = tz * e1x - tx * e1z; val qz = tx * e1y - ty * e1x
        val v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
        if (v < 0f || u + v > 1f) return -1f
        return (e2x * qx + e2y * qy + e2z * qz) * inv
    }

    // ------------------------------------------------------------------------------------------- GL
    /** 4x MSAA when available (smooth edges on thin parts), otherwise a plain RGB888 + depth config. */
    private class Chooser : EGLConfigChooser {
        override fun chooseConfig(egl: EGL10, display: EGLDisplay): EGLConfig {
            for (samples in intArrayOf(4, 0)) {
                val attrs = intArrayOf(EGL10.EGL_RED_SIZE, 8, EGL10.EGL_GREEN_SIZE, 8, EGL10.EGL_BLUE_SIZE, 8, EGL10.EGL_DEPTH_SIZE, 16,
                    EGL10.EGL_RENDERABLE_TYPE, 4 /* EGL_OPENGL_ES2_BIT */, EGL10.EGL_SAMPLE_BUFFERS, if (samples > 0) 1 else 0,
                    EGL10.EGL_SAMPLES, samples, EGL10.EGL_NONE)
                val configs = arrayOfNulls<EGLConfig>(1)
                val n = IntArray(1)
                if (egl.eglChooseConfig(display, attrs, configs, 1, n) && n[0] > 0) return configs[0]!!
            }
            error("no OpenGL ES 2 config")
        }
    }

    private inner class ModelRenderer : Renderer {
        private var prog = 0
        private var shadowProg = 0
        private val buffers = HashMap<ModelScene.Piece, IntArray>()
        private var uploaded: ModelScene? = null
        private var w = 1
        private var h = 1
        private var last = 0L
        private var anchorX = Float.NaN
        private var anchorY = Float.NaN
        private val quad = ByteBuffer.allocateDirect(32).order(ByteOrder.nativeOrder()).asFloatBuffer()
            .apply { put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f)); position(0) }

        override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
            prog = program(VS, FS)
            shadowProg = program(SHADOW_VS, SHADOW_FS)
            buffers.clear()
            uploaded = null
        }

        override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
            w = width; h = height
            GLES20.glViewport(0, 0, width, height)
        }

        private fun upload(s: ModelScene) {
            for (ids in buffers.values) GLES20.glDeleteBuffers(2, ids, 0)
            buffers.clear()
            for (item in s.items) for (pc in item.pieces) {
                val ids = IntArray(2)
                GLES20.glGenBuffers(2, ids, 0)
                GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, ids[0])
                GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, pc.vertices.capacity() * 4, pc.vertices, GLES20.GL_STATIC_DRAW)
                GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, ids[1])
                if (pc.shorts != null) GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, pc.count * 2, pc.shorts, GLES20.GL_STATIC_DRAW)
                else GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, pc.count * 4, pc.ints, GLES20.GL_STATIC_DRAW)
                buffers[pc] = ids
            }
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
            GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0)
            uploaded = s
        }

        override fun onDrawFrame(gl: GL10?) {
            val now = SystemClock.uptimeMillis()
            val dt = if (last == 0L) 0f else min(0.1f, (now - last) / 1000f)
            last = now
            val bg = background
            GLES20.glClearColor(bg[0], bg[1], bg[2], 1f)
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
            val s = scene ?: return
            if (s !== uploaded) upload(s)
            if (spinning) yaw += 14f * dt
            val target = explodeTarget
            explodeT = if (explodeT < target) min(target, explodeT + dt * 1.6f) else max(target, explodeT - dt * 1.6f)
            val t = smooth(explodeT)
            val cam = camera(w, h, t)
            val vp = FloatArray(16)
            Matrix.multiplyMM(vp, 0, cam.proj, 0, cam.view, 0)
            drawShadow(s, vp, t)

            // lights follow the camera (key from upper left, fill from the right) so every side is readable
            val e = cam.eye
            val f = norm(FloatArray(3) { e[it] - s.explodedCenter[it] * t })
            val right = norm(floatArrayOf(f[2], 0f, -f[0]))
            val up = cross(f, right)
            val key = norm(FloatArray(3) { -0.45f * right[it] + 0.75f * up[it] + 0.6f * f[it] })
            val fill = norm(FloatArray(3) { 0.7f * right[it] + 0.1f * up[it] + 0.5f * f[it] })

            GLES20.glUseProgram(prog)
            GLES20.glEnable(GLES20.GL_DEPTH_TEST)
            GLES20.glUniformMatrix4fv(loc("uMVP"), 1, false, vp, 0)
            GLES20.glUniform3fv(loc("uEye"), 1, e, 0)
            GLES20.glUniform3fv(loc("uKey"), 1, key, 0)
            GLES20.glUniform3fv(loc("uFill"), 1, fill, 0)
            GLES20.glUniform3fv(loc("uHiColor"), 1, highlight, 0)
            val sel = selected
            val pulse = 0.55f + 0.45f * sin(now / 1000f * 4.5f)

            GLES20.glDisable(GLES20.GL_BLEND)
            GLES20.glDisable(GLES20.GL_CULL_FACE)
            GLES20.glDepthMask(true)
            for ((i, item) in s.items.withIndex()) for (pc in item.pieces) if (pc.color[3] >= 0.999f) draw(item, pc, i, sel, pulse, t)

            // see-through parts: back faces then front faces, far parts first, without writing depth
            GLES20.glEnable(GLES20.GL_BLEND)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glDepthMask(false)
            GLES20.glEnable(GLES20.GL_CULL_FACE)
            val glass = s.items.withIndex().filter { it.value.transparent }.sortedByDescending { (_, it) ->
                val dx = it.center[0] + it.offset[0] * t - e[0]; val dy = it.center[1] + it.offset[1] * t - e[1]; val dz = it.center[2] + it.offset[2] * t - e[2]
                dx * dx + dy * dy + dz * dz
            }
            for (face in intArrayOf(GLES20.GL_FRONT, GLES20.GL_BACK)) {
                GLES20.glCullFace(face)
                for ((i, item) in glass) for (pc in item.pieces) if (pc.color[3] < 0.999f) draw(item, pc, i, sel, pulse, t)
            }
            GLES20.glDisable(GLES20.GL_CULL_FACE)
            GLES20.glDepthMask(true)
            GLES20.glDisable(GLES20.GL_BLEND)

            reportAnchor(s, vp, sel, t)
        }

        private fun draw(item: ModelScene.Item, pc: ModelScene.Piece, i: Int, sel: Int, pulse: Float, t: Float) {
            val ids = buffers[pc] ?: return
            GLES20.glUniform3f(loc("uOffset"), item.offset[0] * t, item.offset[1] * t, item.offset[2] * t)
            GLES20.glUniform4fv(loc("uColor"), 1, pc.color, 0)
            GLES20.glUniform1f(loc("uMetal"), pc.metallic)
            GLES20.glUniform1f(loc("uRough"), pc.roughness)
            GLES20.glUniform1f(loc("uHi"), if (i == sel) pulse else 0f)
            GLES20.glUniform1f(loc("uDim"), if (sel >= 0 && i != sel) 1f else 0f)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, ids[0])
            val aPos = GLES20.glGetAttribLocation(prog, "aPos")
            val aNrm = GLES20.glGetAttribLocation(prog, "aNrm")
            GLES20.glEnableVertexAttribArray(aPos)
            GLES20.glEnableVertexAttribArray(aNrm)
            GLES20.glVertexAttribPointer(aPos, 3, GLES20.GL_FLOAT, false, 24, 0)
            GLES20.glVertexAttribPointer(aNrm, 3, GLES20.GL_FLOAT, false, 24, 12)
            GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, ids[1])
            GLES20.glDrawElements(GLES20.GL_TRIANGLES, pc.count, if (pc.shorts != null) GLES20.GL_UNSIGNED_SHORT else GLES20.GL_UNSIGNED_INT, 0)
        }

        /** Soft contact shadow under the model. */
        private fun drawShadow(s: ModelScene, vp: FloatArray, t: Float) {
            GLES20.glUseProgram(shadowProg)
            GLES20.glDisable(GLES20.GL_DEPTH_TEST)
            GLES20.glEnable(GLES20.GL_BLEND)
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)
            GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0)
            val a = GLES20.glGetAttribLocation(shadowProg, "aXZ")
            GLES20.glEnableVertexAttribArray(a)
            GLES20.glVertexAttribPointer(a, 2, GLES20.GL_FLOAT, false, 0, quad)
            GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(shadowProg, "uMVP"), 1, false, vp, 0)
            val dark = background[0] + background[1] + background[2] < 1.5f
            GLES20.glUniform4f(GLES20.glGetUniformLocation(shadowProg, "uC"), s.explodedCenter[0] * t,
                s.minY + (min(s.minY, s.explodedMinY) - s.minY) * t - 0.02f, s.explodedCenter[2] * t,
                if (dark) 0.5f else 0.22f)
            GLES20.glUniform1f(GLES20.glGetUniformLocation(shadowProg, "uS"), 1.15f + 0.4f * t)
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
            GLES20.glDisableVertexAttribArray(a)
            GLES20.glDisable(GLES20.GL_BLEND)
        }

        private fun reportAnchor(s: ModelScene, vp: FloatArray, sel: Int, t: Float) {
            var x = Float.NaN; var y = Float.NaN
            val item = s.items.getOrNull(sel)
            if (item != null) {
                val o = FloatArray(4)
                val c = item.center
                Matrix.multiplyMV(o, 0, vp, 0, floatArrayOf(c[0] + item.offset[0] * t, c[1] + item.offset[1] * t, c[2] + item.offset[2] * t, 1f), 0)
                if (o[3] > 0f) { x = (o[0] / o[3] + 1f) / 2f * w; y = (1f - o[1] / o[3]) / 2f * h }
            }
            val moved = if (x.isNaN() || anchorX.isNaN()) x.isNaN() != anchorX.isNaN() else abs(x - anchorX) > 0.5f || abs(y - anchorY) > 0.5f
            if (moved) {
                anchorX = x; anchorY = y
                post { onAnchor?.invoke(x, y) }
            }
        }

        private val locs = HashMap<String, Int>()
        private fun loc(name: String) = locs.getOrPut("$prog/$name") { GLES20.glGetUniformLocation(prog, name) }

        private fun program(vs: String, fs: String): Int {
            fun shader(type: Int, src: String) = GLES20.glCreateShader(type).also {
                GLES20.glShaderSource(it, src)
                GLES20.glCompileShader(it)
                val ok = IntArray(1)
                GLES20.glGetShaderiv(it, GLES20.GL_COMPILE_STATUS, ok, 0)
                if (ok[0] == 0) Log.e(MainActivity.TAG, "shader: " + GLES20.glGetShaderInfoLog(it))
            }
            return GLES20.glCreateProgram().also {
                GLES20.glAttachShader(it, shader(GLES20.GL_VERTEX_SHADER, vs))
                GLES20.glAttachShader(it, shader(GLES20.GL_FRAGMENT_SHADER, fs))
                GLES20.glLinkProgram(it)
            }
        }
    }

    private fun smooth(x: Float) = x * x * (3f - 2f * x)
    private fun norm(v: FloatArray): FloatArray { val l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); return floatArrayOf(v[0] / l, v[1] / l, v[2] / l) }
    private fun cross(a: FloatArray, b: FloatArray) = floatArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    private companion object {
        const val VS = """
            uniform mat4 uMVP;
            uniform vec3 uOffset;
            attribute vec3 aPos;
            attribute vec3 aNrm;
            varying vec3 vN;
            varying vec3 vW;
            void main() {
                vec3 p = aPos + uOffset;
                vW = p;
                vN = aNrm;
                gl_Position = uMVP * vec4(p, 1.0);
            }"""

        // Blinn-Phong with camera-following key/fill lights, a hemisphere ambient and a fake sky reflection for metals
        const val FS = """
            precision mediump float;
            uniform vec4 uColor;
            uniform float uMetal;
            uniform float uRough;
            uniform vec3 uEye;
            uniform vec3 uKey;
            uniform vec3 uFill;
            uniform vec3 uHiColor;
            uniform float uHi;
            uniform float uDim;
            varying vec3 vN;
            varying vec3 vW;
            void main() {
                vec3 n = normalize(vN);
                if (!gl_FrontFacing) n = -n;
                vec3 v = normalize(uEye - vW);
                vec3 base = uColor.rgb;
                vec3 amb = mix(vec3(0.20, 0.20, 0.22), vec3(0.42, 0.43, 0.46), 0.5 + 0.5 * n.y);
                float d1 = max(dot(n, uKey), 0.0);
                float d2 = max(dot(n, uFill), 0.0);
                vec3 col = base * (amb + 0.85 * d1 + 0.3 * d2) * (1.0 - 0.55 * uMetal);
                float shin = mix(96.0, 6.0, uRough);
                vec3 f0 = mix(vec3(0.05), base, uMetal);
                float spec = pow(max(dot(n, normalize(uKey + v)), 0.0), shin) * (1.0 - 0.6 * uRough);
                vec3 r = reflect(-v, n);
                vec3 env = mix(vec3(0.16, 0.16, 0.18), vec3(0.95, 0.96, 1.0), smoothstep(-0.1, 0.8, r.y));
                col += f0 * spec * 1.4 + f0 * env * uMetal * (1.0 - 0.5 * uRough) * 0.85;
                float rim = pow(1.0 - max(dot(n, v), 0.0), 2.5);
                col += uHiColor * uHi * (0.35 + 0.9 * rim);
                col = mix(col, col * 0.4 + vec3(0.06), uDim * 0.75);
                gl_FragColor = vec4(col, uColor.a * (1.0 - 0.4 * uDim));
            }"""

        const val SHADOW_VS = """
            uniform mat4 uMVP;
            uniform vec4 uC;
            uniform float uS;
            attribute vec2 aXZ;
            varying vec2 vUV;
            void main() {
                vUV = aXZ;
                gl_Position = uMVP * vec4(uC.x + aXZ.x * uS, uC.y, uC.z + aXZ.y * uS, 1.0);
            }"""

        const val SHADOW_FS = """
            precision mediump float;
            uniform vec4 uC;
            varying vec2 vUV;
            void main() {
                float d = length(vUV);
                gl_FragColor = vec4(0.0, 0.0, 0.0, uC.w * (1.0 - smoothstep(0.15, 1.0, d)));
            }"""
    }
}
