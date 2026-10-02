package com.enginebay.vision

import android.graphics.Bitmap
import com.enginebay.vision.core.Diagnosis
import com.enginebay.vision.core.DtcTable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Process-wide state that survives the activity being recreated (switching the app language recreates it):
 * the loaded model and tables, the photo with its result, the entered error codes and the threshold settings.
 * Activity destruction must not clear this singleton: another activity can already be using the same session.
 */
object AppState {
    /** One background thread for model loading, decoding and inference, shared by every activity instance. */
    val worker: ExecutorService = Executors.newSingleThreadExecutor()
    var extrasConsumed = false
    @Volatile var detector: Detector? = null
    var components: Map<String, ComponentInfo> = emptyMap()
    var dtcTable: DtcTable? = null
    var photo: Bitmap? = null
    var result: DetectionResult? = null
    var diagnosis: Diagnosis? = null
    var dtcText: String = ""
    var conf: Float? = null
    var perClass: Boolean? = null
    var inspection = com.enginebay.vision.core.Inspection()
    var inspectionLoaded = false
    var findTarget: String? = null

    fun clear() {
        detector = null; components = emptyMap(); dtcTable = null; photo = null; result = null
        diagnosis = null; dtcText = ""; conf = null; perClass = null
        inspection = com.enginebay.vision.core.Inspection(); inspectionLoaded = false; findTarget = null
    }
}
