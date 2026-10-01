package com.enginebay.vision

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.TextView
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import com.google.android.material.chip.ChipGroup
import com.google.android.material.color.MaterialColors
import kotlin.math.min

/**
 * The 3D block of a component sheet: interactive model, exploded-view and reset buttons, one chip per part (parts
 * with an inspection note carry a magnifier icon) and the selected part's note. Tapping a part on the model and
 * choosing its chip are the same selection.
 */
class ModelPanel(parent: ViewGroup, scene: ModelScene, private val vietnamese: Boolean) {
    val root: View = LayoutInflater.from(parent.context).inflate(R.layout.view_model3d, parent, false)
    val modelView: ModelView = root.findViewById(R.id.modelView)
    private val anchor: TextView = root.findViewById(R.id.anchorLabel)
    private val partName: TextView = root.findViewById(R.id.partName)
    private val partCheck: TextView = root.findViewById(R.id.partCheck)
    private val chips: ChipGroup = root.findViewById(R.id.partChips)
    private val chipIds = IntArray(scene.items.size)
    private val items = scene.items
    private var syncing = false

    init {
        val ctx = parent.context
        val res = ctx.resources
        // about half the screen height (landscape tablets have little height), at most 340 dp
        root.findViewById<View>(R.id.modelFrame).layoutParams.height =
            min((res.displayMetrics.heightPixels * 0.5f).toInt(), (340 * res.displayMetrics.density).toInt())
        modelView.setScene(scene,
            MaterialColors.getColor(root, com.google.android.material.R.attr.colorSurfaceContainerLow),
            MaterialColors.getColor(root, androidx.appcompat.R.attr.colorPrimary))
        modelView.onPartTapped = { i -> select(i, fromModel = true) }
        modelView.onAnchor = ::moveAnchor

        val explode = root.findViewById<MaterialButton>(R.id.explode)
        explode.visibility = if (scene.canExplode) View.VISIBLE else View.GONE
        explode.addOnCheckedChangeListener { _, checked -> modelView.exploded = checked }
        root.findViewById<MaterialButton>(R.id.resetView).setOnClickListener {
            modelView.resetView()
            explode.isChecked = false
            select(-1, fromModel = true)
        }

        // parts with something to inspect first, then the others, in model order
        val order = items.indices.sortedBy { if (items[it].part.check != null) 0 else 1 }
        for (i in order) {
            val chip = Chip(ctx).apply {
                id = View.generateViewId()
                text = items[i].part.label.get(if (vietnamese) "vi" else "en")
                isCheckable = true
                isCheckedIconVisible = false
                if (items[i].part.check != null) {
                    setChipIconResource(R.drawable.ic_inspect)
                    isChipIconVisible = true
                }
            }
            chipIds[i] = chip.id
            chips.addView(chip)
        }
        chips.setOnCheckedStateChangeListener { _, ids ->
            if (syncing) return@setOnCheckedStateChangeListener
            select(chipIds.indexOf(ids.firstOrNull() ?: -1), fromModel = false)
        }
    }

    private fun lang() = if (vietnamese) "vi" else "en"

    /** Select part [i] (-1: none) on the model, the chips and the note. */
    private fun select(i: Int, fromModel: Boolean) {
        modelView.select(i)
        if (fromModel) {
            syncing = true
            if (i >= 0) chips.check(chipIds[i]) else chips.clearCheck()
            syncing = false
        }
        val item = items.getOrNull(i)
        if (item == null) {
            partName.visibility = View.GONE
            partCheck.setText(R.string.model_pick_hint)
            anchor.visibility = View.GONE
            return
        }
        partName.visibility = View.VISIBLE
        partName.text = item.part.label.get(lang())
        partCheck.text = item.part.check?.get(lang()) ?: root.context.getString(R.string.model_no_check)
        anchor.text = item.part.label.get(lang())
    }

    private fun moveAnchor(x: Float, y: Float) {
        if (x.isNaN() || modelView.width == 0) { anchor.visibility = View.GONE; return }
        anchor.visibility = View.VISIBLE
        val w = anchor.width.takeIf { it > 0 } ?: anchor.measuredWidth
        val h = anchor.height.takeIf { it > 0 } ?: anchor.measuredHeight
        val lp = anchor.layoutParams as FrameLayout.LayoutParams
        // above the part's centre, kept inside the view
        anchor.translationX = (x - w / 2f).coerceIn(4f, (modelView.width - w - 4f).coerceAtLeast(4f)) - lp.leftMargin
        anchor.translationY = (y - h - 14f * root.resources.displayMetrics.density).coerceIn(4f, (modelView.height - h - 4f).coerceAtLeast(4f))
    }
}
