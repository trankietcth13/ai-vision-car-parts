package com.enginebay.vision.core

import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class InspectionTest {
    @get:Rule val folder = TemporaryFolder()
    private fun sample() = Inspection(vehicle = "51A-12345 / xe thử", photoId = "photo-a", width = 640, height = 480,
        notes = "Dòng 1\nDòng 2", codes = "P0301 U0100", model = "student / v1 / hash",
        parts = listOf(InspectionPart("battery", Text("Battery", "Ắc quy"), 0.83f, listOf(1f, 2f, 100f, 200f))),
        checks = listOf(InspectionCheck("battery", Text("Battery", "Ắc quy"), "identity", Text("Confirm identity", "Xác nhận"),
            CheckOutcome.CONCERN, "AI label needs review")))

    @Test fun inspectionRoundTripKeepsEvidenceAndManualOutcome() {
        val value = sample()
        assertEquals(value, Inspection.decode(value.encode()))
        assertEquals(CheckOutcome.PENDING, InspectionCheck("battery", Text("Battery", "Ắc quy"), "x", Text("Check", "Kiểm tra")).outcome)
    }

    @Test fun savedHistorySurvivesDraftReplacementAndRepositoryRestart() {
        val root = folder.newFolder()
        val repo = InspectionRepository(root)
        val value = sample()
        repo.writePhoto(value.photoId, byteArrayOf(1, 2, 3))
        repo.saveDraft(value); repo.save(value)
        val next = Inspection(vehicle = "Another vehicle")
        repo.saveDraft(next)
        val restarted = InspectionRepository(root)
        assertEquals(next, restarted.draft())
        assertEquals(listOf(value), restarted.list().inspections)
        assertArrayEquals(byteArrayOf(1, 2, 3), restarted.photo(value.photoId))
        assertFalse(File(root, "draft.json.tmp").exists())
    }

    @Test fun corruptHistoryDoesNotHideHealthyRecordsOrExposeTempWrites() {
        val root = folder.newFolder()
        val repo = InspectionRepository(root)
        val value = sample().copy(photoId = "")
        repo.save(value)
        File(root, "history/broken.json").writeText("{")
        File(root, "history/pending.json.tmp").writeText("{")
        assertEquals(1, repo.list().unreadable)
        assertEquals(listOf(value), repo.list().inspections)
    }

    @Test fun historySortedByDateAndMissingPhotoCannotBeSaved() {
        val repo = InspectionRepository(folder.newFolder())
        repo.save(sample().copy(id = "older", createdAt = 10, photoId = ""))
        repo.save(sample().copy(id = "newer", createdAt = 20, photoId = ""))
        assertEquals(listOf("newer", "older"), repo.list().inspections.map { it.id })
        assertThrows(IllegalArgumentException::class.java) { repo.save(sample()) }
        assertThrows(IllegalArgumentException::class.java) { repo.photo("../../outside") }
        val goodDraft = Inspection(vehicle = "Original draft")
        repo.saveDraft(goodDraft)
        assertThrows(IllegalArgumentException::class.java) { repo.saveDraft(sample()) }
        assertEquals(goodDraft, repo.draft())
    }

    @Test fun htmlIsSelfContainedBilingualAndEscapesUserInput() {
        val labels = listOf("title", "vehicle", "codes", "model", "disclaimer", "photo", "parts", "checks", "notes")
            .associateWith { it } + CheckOutcome.entries.associate { it.name to it.name }
        val report = InspectionHtml.render(sample().copy(vehicle = "<script>alert('x')</script>", notes = "<img src=x onerror=alert(1)>"),
            byteArrayOf(1, 2, 3), "vi", labels, "today")
        assertTrue(report.contains("Ắc quy")); assertTrue(report.contains("Xác nhận")); assertTrue(report.contains("CONCERN"))
        assertTrue(report.contains("data:image/jpeg;base64,AQID")); assertTrue(report.contains("&lt;script&gt;"))
        assertFalse(report.contains("<script>")); assertFalse(report.contains("<img src=x"))
        assertFalse(report.contains("https://"))
    }
}
