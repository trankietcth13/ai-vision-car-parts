package com.enginebay.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import java.io.InputStream
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Decodes a photo upright (EXIF orientation applied) and no larger than [maxSide], like the Python app
 * (downscale to 1600 px first). Big camera photos are subsampled while decoding (power of two, which averages
 * pixels) and then scaled with filtering, so memory stays small.
 */
object PhotoLoader {
    fun fromUri(context: Context, uri: Uri, maxSide: Int): Bitmap =
        load({ context.contentResolver.openInputStream(uri) ?: error("cannot open $uri") }, maxSide)

    fun fromAsset(context: Context, path: String, maxSide: Int): Bitmap = load({ context.assets.open(path) }, maxSide)

    private fun load(open: () -> InputStream, maxSide: Int): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        open().use { BitmapFactory.decodeStream(it, null, bounds) }
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "not an image" }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val decoded = open().use {
            BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply {
                inSampleSize = sample
                inPreferredConfig = Bitmap.Config.ARGB_8888
            })
        } ?: error("decode failed")
        val orientation = open().use { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
        }
        val k = minOf(1f, maxSide.toFloat() / max(decoded.width, decoded.height))
        if (k < 1f) m.postScale(k, k)
        if (m.isIdentity) return decoded
        val out = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, m, true)
        if (out !== decoded) decoded.recycle()
        // createBitmap may round the size; keep the long side exactly at maxSide when it was scaled
        return if (k < 1f && max(out.width, out.height) != maxSide) {
            val s = maxSide.toFloat() / max(out.width, out.height)
            Bitmap.createScaledBitmap(out, (out.width * s).roundToInt(), (out.height * s).roundToInt(), true).also { out.recycle() }
        } else out
    }
}
