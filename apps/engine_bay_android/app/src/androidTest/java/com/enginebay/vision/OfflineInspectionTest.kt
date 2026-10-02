package com.enginebay.vision

import android.Manifest
import android.content.pm.PackageManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.enginebay.vision.core.*
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID
import java.util.concurrent.TimeUnit

/** Runs on a real device; uses only a dedicated cache folder, never the user's inspection repository. */
@RunWith(AndroidJUnit4::class)
class OfflineInspectionTest {
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test fun packagedAppCannotAccessInternet() {
        assertEquals(PackageManager.PERMISSION_DENIED, context.packageManager.checkPermission(Manifest.permission.INTERNET, context.packageName))
    }

    @Test fun localVisionStillDetectsBundledSample() {
        val file = context.assets.list("examples")!!.first { it.endsWith(".jpg") }
        val photo = PhotoLoader.fromAsset(context, "examples/$file", 1600)
        Detector(context).use { detector ->
            val result = detector.detect(photo, detector.config.defaultConf, true)
            assertTrue("No parts in bundled example $file", result.parts.isNotEmpty())
        }
        photo.recycle()
    }

    @Test fun repositoryWritesAndRestoresOnAndroidFilesystem() {
        val root = File(context.cacheDir, "inspection-test-${UUID.randomUUID()}")
        try {
            val repo = InspectionRepository(root)
            val value = Inspection(vehicle = "QA-ONLY", photoId = "test-photo", codes = "P0301",
                checks = listOf(InspectionCheck("battery", Text("Battery", "Ắc quy"), "identity", Text("Check", "Kiểm tra"), CheckOutcome.CONCERN, "Test note")))
            repo.writePhoto(value.photoId, byteArrayOf(1, 2, 3)); repo.saveDraft(value); repo.save(value)
            assertEquals(value, InspectionRepository(root).draft())
            assertEquals(listOf(value), InspectionRepository(root).list().inspections)
        } finally { root.deleteRecursively() }
    }

    @Test fun rapidActivityRelaunchKeepsSharedInspectionUsable() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        fun launch() = instrumentation.startActivitySync(android.content.Intent(context, MainActivity::class.java)
            .addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)) as MainActivity
        fun waitForLoad() {
            // A serial-worker barrier plus UI idle ensures the load callback has finished, without arbitrary sleeps.
            AppState.worker.submit {}.get(30, TimeUnit.SECONDS)
            instrumentation.waitForIdleSync()
        }
        val first = launch()
        waitForLoad()
        instrumentation.runOnMainSync { first.finish() }
        val second = launch()
        try {
            waitForLoad()
            instrumentation.runOnMainSync {
                assertTrue(AppState.inspectionLoaded)
                assertNotNull(AppState.detector)
                assertNotNull(second.findViewById<android.view.View>(R.id.inspection_actions))
            }
        } finally { instrumentation.runOnMainSync { second.finish() } }
    }
}
