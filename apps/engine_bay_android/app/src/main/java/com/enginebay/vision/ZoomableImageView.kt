package com.enginebay.vision

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.drawable.Drawable
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.animation.AccelerateDecelerateInterpolator
import androidx.appcompat.widget.AppCompatImageView

/**
 * An [AppCompatImageView] supporting pinch-to-zoom, smooth double-tap zoom,
 * dragging/panning when zoomed in, bounds clamping, and manual zoom controls.
 */
class ZoomableImageView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : AppCompatImageView(context, attrs, defStyleAttr) {

    var minScale = 1.0f
    var maxScale = 5.0f

    /** Callback invoked whenever the zoom scale factor changes. */
    var onScaleChanged: ((scale: Float) -> Unit)? = null

    private val baseMatrix = Matrix()
    private val suppMatrix = Matrix()
    private val displayMatrix = Matrix()
    private val matrixValues = FloatArray(9)

    private var currentZoom = 1.0f
    private var zoomAnimator: ValueAnimator? = null

    val currentScale: Float
        get() = currentZoom

    private val scaleListener = object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            val scaleFactor = detector.scaleFactor
            if (scaleFactor.isNaN() || scaleFactor.isInfinite()) return false

            val targetScale = (currentZoom * scaleFactor).coerceIn(minScale, maxScale)
            val actualFactor = targetScale / currentZoom
            currentZoom = targetScale

            suppMatrix.postScale(actualFactor, actualFactor, detector.focusX, detector.focusY)
            checkAndApplyMatrix()
            onScaleChanged?.invoke(currentZoom)
            return true
        }
    }

    private val gestureListener = object : GestureDetector.SimpleOnGestureListener() {
        override fun onDown(e: MotionEvent): Boolean = true

        override fun onDoubleTap(e: MotionEvent): Boolean {
            val target = if (currentZoom > 1.25f) minScale else 2.5f.coerceAtMost(maxScale)
            animateZoomTo(target, e.x, e.y)
            return true
        }

        override fun onScroll(
            e1: MotionEvent?,
            e2: MotionEvent,
            distanceX: Float,
            distanceY: Float
        ): Boolean {
            if (currentZoom > 1.02f) {
                suppMatrix.postTranslate(-distanceX, -distanceY)
                checkAndApplyMatrix()
                return true
            }
            return false
        }
    }

    private val scaleDetector = ScaleGestureDetector(context, scaleListener)
    private val gestureDetector = GestureDetector(context, gestureListener)

    init {
        scaleType = ScaleType.MATRIX
    }

    override fun setImageBitmap(bm: Bitmap?) {
        super.setImageBitmap(bm)
        resetZoomState()
    }

    override fun setImageDrawable(drawable: Drawable?) {
        super.setImageDrawable(drawable)
        resetZoomState()
    }

    override fun onLayout(changed: Boolean, left: Int, top: Int, right: Int, bottom: Int) {
        super.onLayout(changed, left, top, right, bottom)
        if (changed) {
            updateBaseMatrix()
        }
    }

    private fun resetZoomState() {
        zoomAnimator?.cancel()
        suppMatrix.reset()
        currentZoom = 1.0f
        updateBaseMatrix()
        onScaleChanged?.invoke(currentZoom)
    }

    private fun updateBaseMatrix() {
        val d = drawable ?: return
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0f || viewH <= 0f) return

        val dW = d.intrinsicWidth.toFloat()
        val dH = d.intrinsicHeight.toFloat()
        if (dW <= 0f || dH <= 0f) return

        baseMatrix.reset()
        val scale = minOf(viewW / dW, viewH / dH)
        val dx = (viewW - dW * scale) / 2f
        val dy = (viewH - dH * scale) / 2f

        baseMatrix.postScale(scale, scale)
        baseMatrix.postTranslate(dx, dy)

        checkAndApplyMatrix()
    }

    private fun checkAndApplyMatrix() {
        clampBounds()
        displayMatrix.set(baseMatrix)
        displayMatrix.postConcat(suppMatrix)
        imageMatrix = displayMatrix
    }

    private fun clampBounds() {
        val d = drawable ?: return
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0f || viewH <= 0f) return

        val temp = Matrix(baseMatrix).apply { postConcat(suppMatrix) }
        val rect = RectF(0f, 0f, d.intrinsicWidth.toFloat(), d.intrinsicHeight.toFloat())
        temp.mapRect(rect)

        var deltaX = 0f
        var deltaY = 0f

        if (rect.width() <= viewW) {
            deltaX = (viewW - rect.width()) / 2f - rect.left
        } else {
            if (rect.left > 0f) {
                deltaX = -rect.left
            } else if (rect.right < viewW) {
                deltaX = viewW - rect.right
            }
        }

        if (rect.height() <= viewH) {
            deltaY = (viewH - rect.height()) / 2f - rect.top
        } else {
            if (rect.top > 0f) {
                deltaY = -rect.top
            } else if (rect.bottom < viewH) {
                deltaY = viewH - rect.bottom
            }
        }

        if (deltaX != 0f || deltaY != 0f) {
            suppMatrix.postTranslate(deltaX, deltaY)
        }
    }

    fun zoomIn(stepFactor: Float = 1.35f) {
        val target = (currentZoom * stepFactor).coerceIn(minScale, maxScale)
        animateZoomTo(target, width / 2f, height / 2f)
    }

    fun zoomOut(stepFactor: Float = 1.35f) {
        val target = (currentZoom / stepFactor).coerceIn(minScale, maxScale)
        animateZoomTo(target, width / 2f, height / 2f)
    }

    fun resetZoom() {
        animateZoomTo(minScale, width / 2f, height / 2f)
    }

    private fun animateZoomTo(targetZoom: Float, focusX: Float, focusY: Float) {
        if (width <= 0 || height <= 0) return
        zoomAnimator?.cancel()
        val startZoom = currentZoom
        if (kotlin.math.abs(targetZoom - startZoom) < 0.01f) {
            if (targetZoom == minScale) {
                suppMatrix.reset()
                currentZoom = minScale
                checkAndApplyMatrix()
                onScaleChanged?.invoke(currentZoom)
            }
            return
        }

        zoomAnimator = ValueAnimator.ofFloat(startZoom, targetZoom).apply {
            duration = 240
            interpolator = AccelerateDecelerateInterpolator()
            var prevZoom = startZoom
            addUpdateListener { anim ->
                val zoomVal = anim.animatedValue as Float
                val factor = zoomVal / prevZoom
                prevZoom = zoomVal
                currentZoom = zoomVal
                suppMatrix.postScale(factor, factor, focusX, focusY)
                if (zoomVal <= minScale + 0.001f) {
                    suppMatrix.reset()
                }
                checkAndApplyMatrix()
                onScaleChanged?.invoke(currentZoom)
            }
            start()
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        // Prevent parent scroll container from intercepting when zooming or panning
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (currentZoom <= 1.05f) {
                    parent?.requestDisallowInterceptTouchEvent(false)
                }
            }
        }

        var handled = scaleDetector.onTouchEvent(event)
        handled = gestureDetector.onTouchEvent(event) || handled

        // If user is touching with 1 finger and not zoomed in, allow parent to scroll if moving vertically
        if (currentZoom <= 1.05f && !scaleDetector.isInProgress && event.pointerCount == 1) {
            if (event.actionMasked == MotionEvent.ACTION_MOVE) {
                parent?.requestDisallowInterceptTouchEvent(false)
            }
        } else {
            parent?.requestDisallowInterceptTouchEvent(true)
        }

        return handled || super.onTouchEvent(event)
    }
}
