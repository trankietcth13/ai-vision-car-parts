package com.enginebay.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.max

/** The analysed photo with masks, boxes and labels. Tapping a component reports it through [onPartTapped]. */
class ResultView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    var onPartTapped: ((Part) -> Unit)? = null
    private var photo: Bitmap? = null
    private var result: DetectionResult? = null
    var showOriginal = false
        set(v) { field = v; invalidate() }
    /** Model classes to emphasise (error-code diagnosis); the other components are drawn faded. null or empty: no emphasis. */
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

    fun show(photo: Bitmap, result: DetectionResult?) {
        this.photo = photo
        this.result = result
        focusOverlay = null
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
        return RectF(x, 0f, x + w, h)
    }

    override fun onDraw(canvas: Canvas) {
        val p = photo ?: return
        dst.set(photoRect(p))
        canvas.drawBitmap(p, null, dst, bitmapPaint)
        val r = result ?: return
        if (showOriginal) return
        val k = dst.width() / p.width  // photo px -> view px
        val ox = dst.left
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
            val b = RectF(ox + part.rect.left * k, part.rect.top * k, ox + part.rect.right * k, part.rect.bottom * k)
            val dim = focus != null && part.name !in focus
            boxPaint.color = part.color
            boxPaint.alpha = if (dim) 70 else 255
            boxPaint.strokeWidth = if (focus != null && !dim) 2 * lw else lw
            canvas.drawRect(b, boxPaint)
            if (dim) continue
            val label = "${part.nameVi} ${(part.score * 100).toInt()}%"
            val tw = textPaint.measureText(label) + 8 * density
            val th = fs + 6 * density
            val top = if (b.top - th >= 0) b.top - th else b.top
            val left = b.left.coerceAtMost(dst.right - tw).coerceAtLeast(dst.left)
            fillPaint.color = part.color
            canvas.drawRect(left, top, left + tw, top + th, fillPaint)
            val lum = Color.red(part.color) * 0.299 + Color.green(part.color) * 0.587 + Color.blue(part.color) * 0.114
            textPaint.color = if (lum > 150) Color.BLACK else Color.WHITE
            canvas.drawText(label, left + 4 * density, top + fs + 1 * density, textPaint)
        }
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

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val p = photo ?: return false
        val r = result ?: return false
        if (event.action == MotionEvent.ACTION_DOWN) return true
        if (event.action != MotionEvent.ACTION_UP || showOriginal) return false
        val pr = photoRect(p)
        if (!pr.contains(event.x, event.y)) return true
        val px = (event.x - pr.left) * p.width / pr.width()
        val py = (event.y - pr.top) * p.height / pr.height()
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
        return true
    }

    override fun performClick(): Boolean = super.performClick()
}
