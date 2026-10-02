package com.enginebay.vision

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import kotlin.math.max

/** The analysed photo with masks, boxes and labels. Supports pinch-to-zoom, pan, double-tap zoom, and part tapping. */
class ResultView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    var onPartTapped: ((Part) -> Unit)? = null
    var onScaleChanged: ((scale: Float) -> Unit)? = null

    private var photo: Bitmap? = null
    private var result: DetectionResult? = null
    var showOriginal = false
        set(v) { field = v; invalidate() }

    private val vietnamese = context.resources.configuration.locales[0].language == "vi"
    var highlight: Set<String>? = null
        set(v) { field = v; focusOverlay = null; invalidate() }
    private var focusOverlay: Bitmap? = null

    private val dst = RectF()
    private val src = Rect()
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE }
    private val fillPaint = Paint()
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { isFakeBoldText = true }
    private val density = resources.displayMetrics.density

    // Transformation and Zoom state
    private val transformMatrix = Matrix()
    private val inverseMatrix = Matrix()
    var minZoom = 1.0f
    var maxZoom = 5.0f
    private var zoom = 1.0f
    private var zoomAnimator: ValueAnimator? = null

    val currentScale: Float
        get() = zoom

    private val scaleListener = object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            val factor = detector.scaleFactor
            if (factor.isNaN() || factor.isInfinite()) return false
            val targetZoom = (zoom * factor).coerceIn(minZoom, maxZoom)
            val actualFactor = targetZoom / zoom
            zoom = targetZoom
            transformMatrix.postScale(actualFactor, actualFactor, detector.focusX, detector.focusY)
            clampBounds()
            invalidate()
            onScaleChanged?.invoke(zoom)
            return true
        }
    }

    private val gestureListener = object : GestureDetector.SimpleOnGestureListener() {
        override fun onDown(e: MotionEvent): Boolean = true

        override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
            handleTap(e.x, e.y)
            return true
        }

        override fun onDoubleTap(e: MotionEvent): Boolean {
            val target = if (zoom > 1.25f) minZoom else 2.5f.coerceAtMost(maxZoom)
            animateZoomTo(target, e.x, e.y)
            return true
        }

        override fun onScroll(
            e1: MotionEvent?,
            e2: MotionEvent,
            distanceX: Float,
            distanceY: Float
        ): Boolean {
            if (zoom > 1.02f) {
                transformMatrix.postTranslate(-distanceX, -distanceY)
                clampBounds()
                invalidate()
                return true
            }
            return false
        }
    }

    private val scaleDetector = ScaleGestureDetector(context, scaleListener)
    private val gestureDetector = GestureDetector(context, gestureListener)

    fun show(photo: Bitmap, result: DetectionResult?) {
        this.photo = photo
        this.result = result
        focusOverlay = null
        zoomAnimator?.cancel()
        transformMatrix.reset()
        zoom = 1.0f
        onScaleChanged?.invoke(zoom)
        requestLayout()
        invalidate()
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val w = MeasureSpec.getSize(widthMeasureSpec)
        val p = photo
        // full width, but at most ~60% of the screen height (landscape tablets), so the controls stay visible
        val maxH = (resources.displayMetrics.heightPixels * 0.6f).toInt()
        val h = if (p == null) 0 else minOf((w.toFloat() * p.height / p.width).toInt(), maxH)
        setMeasuredDimension(w, h)
    }

    /** The photo's rectangle inside the view: aspect kept, centred horizontally. */
    private fun photoRect(p: Bitmap): RectF {
        val k = minOf(width.toFloat() / p.width, height.toFloat() / p.height)
        val w = p.width * k; val h = p.height * k
        val x = (width - w) / 2f
        val y = max(0f, (height - h) / 2f)
        return RectF(x, y, x + w, y + h)
    }

    override fun onDraw(canvas: Canvas) {
        val p = photo ?: return
        canvas.save()
        canvas.clipRect(0, 0, width, height)
        canvas.concat(transformMatrix)

        dst.set(photoRect(p))
        canvas.drawBitmap(p, null, dst, bitmapPaint)
        val r = result
        if (r == null || showOriginal) {
            canvas.restore()
            return
        }
        val k = dst.width() / p.width  // photo px -> view px
        val ox = dst.left
        val oy = dst.top
        val lb = r.letterbox
        // overlay covers the letterboxed photo region of the model input
        src.set(lb.left, lb.top, lb.left + lb.newW, lb.top + lb.newH)
        val focus = highlight?.takeIf { it.isNotEmpty() }
        canvas.drawBitmap(if (focus == null) r.overlay else focusOverlay(r, focus), src, dst, bitmapPaint)
        val lw = max(2f, 1.5f * density)
        val fs = 11f * density
        textPaint.textSize = fs
        // with a focus, faded components first (box only), then the emphasised ones on top
        val ordered = if (focus == null) r.parts.asReversed()
        else r.parts.filter { it.name !in focus } + r.parts.filter { it.name in focus }.asReversed()
        for (part in ordered) {
            val b = RectF(ox + part.rect.left * k, oy + part.rect.top * k, ox + part.rect.right * k, oy + part.rect.bottom * k)
            val dim = focus != null && part.name !in focus
            boxPaint.color = part.color
            boxPaint.alpha = if (dim) 70 else 255
            boxPaint.strokeWidth = if (focus != null && !dim) 2 * lw else lw
            canvas.drawRect(b, boxPaint)
            if (dim) continue
            val label = context.getString(R.string.photo_label, part.label(vietnamese), (part.score * 100).toInt())
            val tw = textPaint.measureText(label) + 8 * density
            val th = fs + 6 * density
            val top = if (b.top - th >= dst.top) b.top - th else b.top
            val left = b.left.coerceAtMost(dst.right - tw).coerceAtLeast(dst.left)
            fillPaint.color = part.color
            canvas.drawRect(left, top, left + tw, top + th, fillPaint)
            val lum = Color.red(part.color) * 0.299 + Color.green(part.color) * 0.587 + Color.blue(part.color) * 0.114
            textPaint.color = if (lum > 150) Color.BLACK else Color.WHITE
            canvas.drawText(label, left + 4 * density, top + fs + 1 * density, textPaint)
        }
        canvas.restore()
    }

    /** The mask overlay with the [focus] classes stronger and the rest faint; cached until the result or focus changes. */
    private fun focusOverlay(r: DetectionResult, focus: Set<String>): Bitmap {
        focusOverlay?.let { return it }
        val size = kotlin.math.sqrt(r.maskIndex.size.toDouble()).toInt()
        val colors = IntArray(r.maskIndex.size) { i ->
            val d = r.maskIndex[i]
            if (d < 0) 0 else r.parts[d].let { (it.color and 0x00FFFFFF) or ((if (it.name in focus) 0x90 else 0x1C) shl 24) }
        }
        return Bitmap.createBitmap(colors, size, size, Bitmap.Config.ARGB_8888).also { focusOverlay = it }
    }

    private fun clampBounds() {
        clampMatrix(transformMatrix)
    }

    private fun clampMatrix(m: Matrix) {
        val p = photo ?: return
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0f || viewH <= 0f) return

        val pr = photoRect(p)
        val mapped = RectF(pr)
        m.mapRect(mapped)

        var deltaX = 0f
        var deltaY = 0f

        if (mapped.width() <= viewW) {
            deltaX = (viewW - mapped.width()) / 2f - mapped.left
        } else {
            if (mapped.left > 0f) {
                deltaX = -mapped.left
            } else if (mapped.right < viewW) {
                deltaX = viewW - mapped.right
            }
        }

        if (mapped.height() <= viewH) {
            deltaY = (viewH - mapped.height()) / 2f - mapped.top
        } else {
            if (mapped.top > 0f) {
                deltaY = -mapped.top
            } else if (mapped.bottom < viewH) {
                deltaY = viewH - mapped.bottom
            }
        }

        if (deltaX != 0f || deltaY != 0f) {
            m.postTranslate(deltaX, deltaY)
        }
    }

    fun animateToMatrix(targetMatrix: Matrix, durationMs: Long = 260) {
        if (width <= 0 || height <= 0) return
        zoomAnimator?.cancel()

        val startMatrix = Matrix(transformMatrix)
        val startValues = FloatArray(9)
        val targetValues = FloatArray(9)
        val currentValues = FloatArray(9)

        startMatrix.getValues(startValues)
        targetMatrix.getValues(targetValues)

        val targetScale = targetValues[Matrix.MSCALE_X]

        zoomAnimator = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = durationMs
            interpolator = AccelerateDecelerateInterpolator()
            addUpdateListener { anim ->
                val fraction = anim.animatedFraction
                for (i in 0 until 9) {
                    currentValues[i] = startValues[i] + (targetValues[i] - startValues[i]) * fraction
                }
                transformMatrix.setValues(currentValues)
                zoom = currentValues[Matrix.MSCALE_X]
                if (fraction >= 1f && targetScale <= minZoom + 0.001f) {
                    transformMatrix.reset()
                    zoom = minZoom
                }
                invalidate()
                onScaleChanged?.invoke(zoom)
            }
            start()
        }
    }

    fun animateZoomTo(targetZoom: Float, focusX: Float, focusY: Float) {
        if (width <= 0 || height <= 0) return
        val targetScale = targetZoom.coerceIn(minZoom, maxZoom)
        val targetMatrix = Matrix(transformMatrix)
        val factor = targetScale / zoom
        targetMatrix.postScale(factor, factor, focusX, focusY)
        clampMatrix(targetMatrix)
        animateToMatrix(targetMatrix, 220)
    }

    fun zoomIn(stepFactor: Float = 1.35f) {
        val targetScale = (zoom * stepFactor).coerceIn(minZoom, maxZoom)
        val targetMatrix = Matrix(transformMatrix)
        val factor = targetScale / zoom
        targetMatrix.postScale(factor, factor, width / 2f, height / 2f)
        clampMatrix(targetMatrix)
        animateToMatrix(targetMatrix, 200)
    }

    fun zoomOut(stepFactor: Float = 1.35f) {
        val targetScale = (zoom / stepFactor).coerceIn(minZoom, maxZoom)
        val targetMatrix = Matrix(transformMatrix)
        val factor = targetScale / zoom
        targetMatrix.postScale(factor, factor, width / 2f, height / 2f)
        clampMatrix(targetMatrix)
        animateToMatrix(targetMatrix, 200)
    }

    fun resetZoom() {
        val targetMatrix = Matrix()
        animateToMatrix(targetMatrix, 240)
    }

    fun zoomToRect(targetRectInPhoto: RectF, targetScaleMultiplier: Float? = null) {
        val p = photo ?: return
        if (width <= 0 || height <= 0) return

        val pr = photoRect(p)
        val k = pr.width() / p.width.toFloat()
        val ox = pr.left
        val oy = pr.top

        val partViewLeft = ox + targetRectInPhoto.left * k
        val partViewTop = oy + targetRectInPhoto.top * k
        val partViewRight = ox + targetRectInPhoto.right * k
        val partViewBottom = oy + targetRectInPhoto.bottom * k
        val partCenterX = (partViewLeft + partViewRight) / 2f
        val partCenterY = (partViewTop + partViewBottom) / 2f
        val partW = max(1f, partViewRight - partViewLeft)
        val partH = max(1f, partViewBottom - partViewTop)

        val viewW = width.toFloat()
        val viewH = height.toFloat()

        val desiredScaleX = (viewW * 0.50f) / partW
        val desiredScaleY = (viewH * 0.50f) / partH
        val calculatedZoom = minOf(desiredScaleX, desiredScaleY).coerceIn(1.6f, 3.6f)
        val idealZoom = (targetScaleMultiplier ?: calculatedZoom).coerceIn(minZoom, maxZoom)

        val targetMatrix = Matrix()
        targetMatrix.setScale(idealZoom, idealZoom)
        val targetTransX = (viewW / 2f) - (idealZoom * partCenterX)
        val targetTransY = (viewH / 2f) - (idealZoom * partCenterY)
        targetMatrix.postTranslate(targetTransX, targetTransY)

        clampMatrix(targetMatrix)
        animateToMatrix(targetMatrix, 300)
    }

    fun zoomToPart(part: Part) {
        zoomToRect(part.rect)
    }

    fun zoomToParts(parts: List<Part>) {
        if (parts.isEmpty()) return
        val union = RectF(parts[0].rect)
        for (i in 1 until parts.size) {
            union.union(parts[i].rect)
        }
        zoomToRect(union)
    }

    private fun handleTap(screenX: Float, screenY: Float) {
        val p = photo ?: return
        val r = result ?: return
        if (showOriginal) return

        if (!transformMatrix.invert(inverseMatrix)) return
        val pts = floatArrayOf(screenX, screenY)
        inverseMatrix.mapPoints(pts)
        val unzoomedX = pts[0]
        val unzoomedY = pts[1]

        val pr = photoRect(p)
        if (!pr.contains(unzoomedX, unzoomedY)) return
        val px = (unzoomedX - pr.left) * p.width / pr.width()
        val py = (unzoomedY - pr.top) * p.height / pr.height()
        val lb = r.letterbox
        val mx = (px * lb.scale + lb.left).toInt()
        val my = (py * lb.scale + lb.top).toInt()
        val size = kotlin.math.sqrt(r.maskIndex.size.toDouble()).toInt()
        var hit = if (mx in 0 until size && my in 0 until size) r.maskIndex[my * size + mx] else -1
        if (hit < 0) {  // not on a mask: smallest box that contains the point
            hit = r.parts.indices.filter { r.parts[it].rect.contains(px, py) }
                .minByOrNull { r.parts[it].rect.width() * r.parts[it].rect.height() } ?: -1
        }
        if (hit >= 0) {
            performClick()
            onPartTapped?.invoke(r.parts[hit])
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (photo == null) return false

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (zoom <= 1.05f) {
                    parent?.requestDisallowInterceptTouchEvent(false)
                }
            }
        }

        var handled = scaleDetector.onTouchEvent(event)
        handled = gestureDetector.onTouchEvent(event) || handled

        if (zoom <= 1.05f && !scaleDetector.isInProgress && event.pointerCount == 1) {
            if (event.actionMasked == MotionEvent.ACTION_MOVE) {
                parent?.requestDisallowInterceptTouchEvent(false)
            }
        } else {
            parent?.requestDisallowInterceptTouchEvent(true)
        }

        return handled || super.onTouchEvent(event)
    }

    override fun performClick(): Boolean = super.performClick()
}
