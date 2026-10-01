package com.enginebay.vision

import android.content.Context
import android.util.Log
import com.enginebay.vision.core.Glb
import com.enginebay.vision.core.ModelBuilder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * The 3D component models, keyed by the knowledge-table component key (= the detector class for detectable ones).
 * assets/models/<component>.json is a primitive spec (scripts/deployment/build_component_models.py): the mesh is
 * GENERATED ON THE DEVICE by [ModelBuilder]. A <component>.glb (e.g. a scanned or CAD part) is read as is.
 * Building has its own thread so it never waits behind an inference; the last few scenes are cached.
 */
object ModelStore {
    private val loader: ExecutorService = Executors.newSingleThreadExecutor()
    @Volatile private var keys: Set<String>? = null
    private val cache = object : LinkedHashMap<String, ModelScene>(8, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, ModelScene>?) = size > 4
    }

    fun available(context: Context): Set<String> = keys ?: (context.assets.list("models").orEmpty()
        .filter { it.endsWith(".json") || it.endsWith(".glb") }.map { it.substringBeforeLast('.') }.toSet()).also { keys = it }

    fun has(context: Context, key: String?) = key != null && key in available(context)

    /** Calls [done] on the loader thread with the scene, or null if it cannot be read. */
    fun load(context: Context, key: String, done: (ModelScene?) -> Unit) {
        val app = context.applicationContext
        loader.execute {
            val scene = synchronized(cache) { cache[key] } ?: try {
                val t0 = System.nanoTime()
                val spec = "models/$key.json"
                val generated = app.assets.list("models").orEmpty().contains("$key.json")
                val model = if (generated) ModelBuilder.build(app.assets.open(spec).bufferedReader().use { it.readText() })
                else Glb.parse(app.assets.open("models/$key.glb").use { it.readBytes() }, key)
                val t1 = System.nanoTime()
                ModelScene(model).also {
                    Log.i(MainActivity.TAG, "3D model $key ${if (generated) "generated on device" else "from GLB"}: " +
                        "${model.parts.size} parts, ${model.triangles} triangles, build ${(t1 - t0) / 1_000_000} ms, " +
                        "GPU buffers ${(System.nanoTime() - t1) / 1_000_000} ms")
                    synchronized(cache) { cache[key] = it }
                }
            } catch (e: Exception) {
                Log.e(MainActivity.TAG, "3D model $key failed", e)
                null
            }
            done(scene)
        }
    }
}
