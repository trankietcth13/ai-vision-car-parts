package com.enginebay.vision.core

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID

enum class CheckOutcome { PENDING, DONE, CONCERN, SKIPPED }

data class InspectionCheck(
    val component: String, val componentName: Text, val step: String, val instruction: Text,
    val outcome: CheckOutcome = CheckOutcome.PENDING, val note: String = "",
)

data class InspectionPart(val key: String, val name: Text, val score: Float, val box: List<Float>)

/** A frozen observation, not a mechanical diagnosis. Names/instructions are saved so asset updates cannot rewrite history. */
data class Inspection(
    val id: String = UUID.randomUUID().toString(), val createdAt: Long = System.currentTimeMillis(),
    val vehicle: String = "", val notes: String = "", val codes: String = "", val photoId: String = "",
    val width: Int = 0, val height: Int = 0, val model: String = "", val threshold: Float = 0.35f,
    val perClass: Boolean = true, val parts: List<InspectionPart> = emptyList(),
    val checks: List<InspectionCheck> = emptyList(),
) {
    fun encode(): String = JSONObject().apply {
        put("schema", 1); put("id", id); put("createdAt", createdAt); put("vehicle", vehicle); put("notes", notes)
        put("codes", codes); put("photoId", photoId); put("width", width); put("height", height); put("model", model)
        put("threshold", threshold.toDouble()); put("perClass", perClass)
        put("parts", JSONArray(parts.map { p -> JSONObject().apply {
            put("key", p.key); put("name", textJson(p.name)); put("score", p.score.toDouble()); put("box", JSONArray(p.box))
        } }))
        put("checks", JSONArray(checks.map { c -> JSONObject().apply {
            put("component", c.component); put("name", textJson(c.componentName)); put("step", c.step)
            put("instruction", textJson(c.instruction)); put("outcome", c.outcome.name); put("note", c.note)
        } }))
    }.toString()

    companion object {
        private fun textJson(t: Text) = JSONObject().put("en", t.en).put("vi", t.vi)
        private fun text(o: JSONObject) = Text(o.getString("en"), o.getString("vi"))
        fun decode(json: String): Inspection {
            val o = JSONObject(json)
            require(o.getInt("schema") == 1) { "Unsupported inspection format" }
            val parts = o.getJSONArray("parts")
            val checks = o.getJSONArray("checks")
            return Inspection(o.getString("id"), o.getLong("createdAt"), o.getString("vehicle"), o.getString("notes"),
                o.getString("codes"), o.getString("photoId"), o.getInt("width"), o.getInt("height"), o.getString("model"),
                o.getDouble("threshold").toFloat(), o.getBoolean("perClass"),
                List(parts.length()) { i -> parts.getJSONObject(i).let { p ->
                    val b = p.getJSONArray("box")
                    require(b.length() == 4)
                    InspectionPart(p.getString("key"), text(p.getJSONObject("name")), p.getDouble("score").toFloat(),
                        List(4) { b.getDouble(it).toFloat() })
                } },
                List(checks.length()) { i -> checks.getJSONObject(i).let { c ->
                    InspectionCheck(c.getString("component"), text(c.getJSONObject("name")), c.getString("step"),
                        text(c.getJSONObject("instruction")), CheckOutcome.valueOf(c.getString("outcome")), c.getString("note"))
                } })
        }
    }
}

/** Private local files. Atomic metadata replacement and immutable photos keep old reports consistent after retakes. */
class InspectionRepository(private val root: File) {
    private val photos = File(root, "photos").apply { mkdirs() }
    private val history = File(root, "history").apply { mkdirs() }
    private fun safeId(id: String): String { require(Regex("[a-zA-Z0-9-]+").matches(id)); return id }
    private fun atomicWrite(file: File, bytes: ByteArray) {
        val temp = File(file.parentFile, "${file.name}.tmp")
        temp.outputStream().use { it.write(bytes); it.fd.sync() }
        try { Files.move(temp.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING) }
        catch (_: java.nio.file.AtomicMoveNotSupportedException) {
            Files.move(temp.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }
    fun writePhoto(id: String, jpeg: ByteArray) = atomicWrite(File(photos, "${safeId(id)}.jpg"), jpeg)
    fun hasPhoto(id: String) = id.isNotBlank() && File(photos, "${safeId(id)}.jpg").isFile
    fun photo(id: String): ByteArray? = if (id.isBlank()) null else File(photos, "${safeId(id)}.jpg").readBytes()
    fun saveDraft(value: Inspection) {
        require(value.photoId.isBlank() || hasPhoto(value.photoId))
        atomicWrite(File(root, "draft.json"), value.encode().toByteArray(Charsets.UTF_8))
    }
    fun draft(): Inspection? = File(root, "draft.json").takeIf { it.exists() }?.let { Inspection.decode(it.readText()) }
    fun save(value: Inspection) {
        require(value.vehicle.isNotBlank())
        if (value.photoId.isNotBlank()) require(File(photos, "${safeId(value.photoId)}.jpg").isFile)
        atomicWrite(File(history, "${safeId(value.id)}.json"), value.encode().toByteArray(Charsets.UTF_8))
    }
    data class History(val inspections: List<Inspection>, val unreadable: Int)
    fun list(): History {
        var unreadable = 0
        val items = history.listFiles().orEmpty().filter { it.extension == "json" }.mapNotNull {
            try { Inspection.decode(it.readText()) } catch (_: Exception) { unreadable++; null }
        }.sortedByDescending { it.createdAt }
        return History(items, unreadable)
    }
}

object InspectionHtml {
    fun escape(s: String) = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace("\"", "&quot;").replace("'", "&#39;")

    /** All user/model text is escaped. No scripts, network resources or remote fonts in the exported document. */
    fun render(value: Inspection, jpeg: ByteArray?, lang: String, labels: Map<String, String>, date: String): String {
        fun e(s: String) = escape(s)
        fun l(key: String) = e(labels.getValue(key))
        fun row(a: String, b: String) = "<tr><th>${e(a)}</th><td>${e(b)}</td></tr>"
        val body = buildString {
            append("<h1>${l("title")}</h1><p>${e(date)}</p><table>")
            append(row(labels.getValue("vehicle"), value.vehicle)); append(row(labels.getValue("codes"), value.codes))
            append(row(labels.getValue("model"), value.model)); append("</table><p class='note'>${l("disclaimer")}</p>")
            if (jpeg != null) {
                val b64 = java.util.Base64.getEncoder().encodeToString(jpeg)
                append("<img alt='${l("photo")}' src='data:image/jpeg;base64,$b64'>")
            }
            append("<h2>${l("parts")}</h2><ul>")
            value.parts.forEach { append("<li>${e(it.name.get(lang))} · ${(it.score * 100).toInt()}%</li>") }
            append("</ul><h2>${l("checks")}</h2>")
            value.checks.forEach { c ->
                append("<section><h3>${e(c.componentName.get(lang))}</h3><p>${e(c.instruction.get(lang))}</p>")
                append("<strong>${l(c.outcome.name)}</strong><p class='multiline'>${e(c.note)}</p></section>")
            }
            append("<h2>${l("notes")}</h2><p class='multiline'>${e(value.notes)}</p>")
        }
        return "<!doctype html><html lang='${if (lang == "vi") "vi" else "en"}'><head><meta charset='utf-8'>" +
            "<meta name='viewport' content='width=device-width, initial-scale=1'><title>${l("title")}</title>" +
            "<style>body{font:16px system-ui,sans-serif;max-width:900px;margin:auto;padding:24px;color:#182b37}" +
            "img{max-width:100%;max-height:620px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}" +
            "section{border:1px solid #ddd;padding:12px;margin:12px 0;break-inside:avoid}.note{background:#fff3cd;padding:12px}" +
            ".multiline{white-space:pre-wrap;overflow-wrap:anywhere}@media print{body{padding:0}}</style></head><body>$body</body></html>"
    }
}
