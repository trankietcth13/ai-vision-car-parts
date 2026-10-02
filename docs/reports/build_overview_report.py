"""
Build the high-level companion to the KD training report: approach, architecture, benchmark, strengths, weaknesses and
recommendations, without the experiment-by-experiment detail. Numbers are the same real-run numbers as
build_training_report.py (imported from it), so the two reports never disagree.

Output: docs/reports/Engine_Bay_KD_Training_Overview_EN.pdf, charts in docs/reports/fig/.

Usage: python docs/reports/build_overview_report.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import PIL.Image
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from build_training_report import FIG, GT, INK, LINE, LOSS, MUTED, OUT, STUDENT, VERSIONS, C

TEACHER = "#008C7E"  # a little more chroma than the full report's teal so it passes the chart palette checks
GRID = "#DCE4E2"
FILE = "Engine_Bay_KD_Training_Overview_EN.pdf"


# ---------------------------------------------------------------- figures
def _box(ax, x, y, w, h, txt, fc, ec, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def _arrow(ax, p, q, col, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=col, lw=1.1, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_pipeline():
    """End-to-end training pipeline, one box per stage."""
    fig, ax = plt.subplots(figsize=(7.2, 1.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 24)
    ax.axis("off")
    stages = [("1  Photos", "28 vehicles\n1,081 images\n20 classes", "#EEF2F1", LINE),
              ("2  Auto-labels", "AI vision models\ndraw boxes,\nSAM 2 traces outlines", "#E3DBF5", GT),
              ("3  Expert review", "every label checked;\naudits measure\nremaining errors", "#E3DBF5", GT),
              ("4  Teacher", "large model learns\nfrom reviewed labels", "#CFE9E5", TEACHER),
              ("5  Student", "small model learns\nfrom labels + teacher", "#F6E4C2", STUDENT),
              ("6  Deploy", "web app, browser,\nAndroid (ONNX)", "#EEF2F1", LINE)]
    w, gap = 14.2, 2.8
    for i, (head, sub, fc, ec) in enumerate(stages):
        x = 0.6 + i * (w + gap)
        _box(ax, x, 1.5, w, 18, "", fc, ec)
        ax.text(x + w / 2, 16.6, head, ha="center", va="center", fontsize=8.2, fontweight="bold", color=INK)
        ax.text(x + w / 2, 8.4, sub, ha="center", va="center", fontsize=6.9, color=INK, linespacing=1.35)
        if i < len(stages) - 1:
            _arrow(ax, (x + w + 0.35, 10.5), (x + w + gap - 0.35, 10.5), MUTED)
    ax.text(0.6 + 1.5 * (w + gap) + w / 2, 23, "data and labels", ha="center", fontsize=7.2, color=GT)
    ax.text(0.6 + 3.5 * (w + gap) + w / 2, 23, "training (DGX GB10)", ha="center", fontsize=7.2, color=TEACHER)
    fig.tight_layout(pad=0.2)
    p = FIG / "overview_pipeline.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_architecture():
    """Teacher -> student knowledge distillation, without loss formulas."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")
    _box(ax, 1, 17, 14, 9, "Engine-bay\nphoto", "#EEF2F1", LINE, fs=8)
    _box(ax, 1, 2, 14, 9, "Reviewed\nlabels", "#E3DBF5", GT, fs=8)
    _box(ax, 25, 30, 34, 9.5, "Teacher  ·  yolo11l-seg\n27.6 M parameters  ·  frozen", "#CFE9E5", TEACHER, fs=8, bold=True)
    _box(ax, 25, 2, 34, 9.5, "Student  ·  yolo11n-seg\n2.84 M parameters  ·  trained", "#F6E4C2", STUDENT, fs=8, bold=True)
    for x, lab in ((30, "what it\nsees"), (42, "how sure\nit is"), (54, "where the\nedges are")):
        _arrow(ax, (x, 29.6), (x, 12), LOSS)
        ax.text(x + 0.6, 20.8, lab, ha="left", va="center", fontsize=7, color=LOSS, linespacing=1.25)
    ax.text(42, 41.3, "the teacher passes three kinds of knowledge to the student", ha="center", fontsize=7.4, color=LOSS)
    _arrow(ax, (15.3, 23), (25, 34), TEACHER)
    _arrow(ax, (15.3, 20), (25, 8), STUDENT)
    _arrow(ax, (15.3, 6.5), (25, 6.5), GT)
    _box(ax, 70, 2, 28.5, 9.5, "Deployed model\nplain yolo11n-seg · 6 MB", "#FFFFFF", STUDENT, fs=8)
    _arrow(ax, (59.3, 6.75), (70, 6.75), STUDENT)
    ax.text(84.2, 16.5, "helper layers used only\nduring training are removed", ha="center", fontsize=7, color=MUTED,
            linespacing=1.3)
    fig.tight_layout(pad=0.2)
    p = FIG / "overview_architecture.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_benchmark():
    """Left: teacher vs student in the two test settings. Right: progress across versions on unseen vehicles."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.35]})
    groups = ["Known vehicles,\nnew photos", "Unseen\nvehicles"]
    teacher, student = [0.418, 0.354], [0.381, 0.286]
    w = 0.36
    for off, ys, col, lab in ((-w / 2, teacher, TEACHER, "Teacher (large)"), (w / 2, student, STUDENT, "Student (small)")):
        bars = a1.bar([i + off for i in range(2)], ys, w * 0.94, color=col, label=lab)
        for b, y in zip(bars, ys):
            a1.text(b.get_x() + b.get_width() / 2, y + 0.008, f"{y:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    a1.set_xticks(range(2), groups, fontsize=8)
    a1.set_ylim(0, 0.5)
    a1.set_ylabel("mask mAP50-95")
    a1.grid(axis="y", color=GRID, lw=0.8)
    a1.set_axisbelow(True)
    a1.legend(frameon=False, fontsize=7.5, loc="upper right")
    a1.set_title("Accuracy of POC v1", fontsize=9, color=INK, loc="left")

    x = list(range(len(VERSIONS)))
    t = [v[1] for v in VERSIONS]
    s = [v[2] for v in VERSIONS]
    a2.axvspan(4.6, len(VERSIONS) - 0.5, color="#EEF2F1", lw=0)
    a2.text((4.6 + len(VERSIONS) - 0.5) / 2, 0.165, "plateau: same 28 vehicles,\nno measurable gain", ha="center",
            fontsize=7, color=MUTED, linespacing=1.3)
    a2.plot(x, t, color=TEACHER, lw=2, marker="o", ms=4.5, label="Teacher")
    a2.plot(x, s, color=STUDENT, lw=2, marker="o", ms=4.5, label="Student")
    for series, col, dy in ((t, TEACHER, 0.012), (s, STUDENT, -0.026)):
        for i in (0, len(series) - 1):
            a2.text(x[i], series[i] + dy, f"{series[i]:.2f}", ha="center", fontsize=7.2, color=INK)
    a2.set_xticks(x, [v[0].replace("kd_n_p5t", "p5t") for v in VERSIONS], fontsize=7.5)
    a2.set_ylim(0.15, 0.4)
    a2.set_ylabel("mask mAP50-95, unseen vehicles")
    a2.grid(axis="y", color=GRID, lw=0.8)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=7.5, loc="upper left")
    a2.set_title("Progress across training versions", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    p = FIG / "overview_benchmark.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- content
BODY = [
    ("h1", "1. At a glance"),
    ("p", "We trained an AI model that finds and outlines 20 engine-bay components (battery, fuse box, reservoirs, "
          "air filter box, alternator and others) in inspection photos. A large, accurate <b>teacher</b> model is "
          "trained first; its knowledge is then <b>distilled</b> into a <b>student</b> model ten times smaller, which "
          "is fast enough to run in a web browser or on an Android tablet without a server. The current models were "
          "frozen as <b>POC v1</b> on 30 September 2026."),
    ("bullets", [
        "<b>On vehicles it was trained on</b> (new photos), the student is close to the teacher: 0.38 vs 0.42 "
        "mask mAP50-95, about 91% of the teacher's accuracy at one tenth of the size.",
        "<b>On vehicles it has never seen</b>, accuracy drops to 0.27–0.30. There, the output should be treated as "
        "suggestions for a technician to confirm.",
        "<b>The main limit is data, not the training method.</b> More distinct vehicles are the next step; further "
        "tuning on the current 28 vehicles no longer helps measurably.",
    ]),
    ("h1", "2. Training approach"),
    ("img", "pipeline"),
    ("caption", "End-to-end pipeline. Labels are machine-drafted and then expert-reviewed; both models train on the "
                "same reviewed labels."),
    ("bullets", [
        "<b>Data.</b> 1,081 verified inspection photos of 28 vehicles. Accuracy is always measured on photos the "
        "model did not train on: either new photos of the same vehicles (3-fold cross-validation) or 3 vehicles kept "
        "out of training entirely.",
        "<b>Labels.</b> Vision-language models (Qwen, DeepSeek) propose boxes, SAM 2 turns each box into an exact "
        "outline, and every label is reviewed by an expert. Targeted audits measure the errors that remain.",
        "<b>Two-stage training.</b> The teacher learns from the labels alone. The student then learns from the labels "
        "and, at the same time, imitates the teacher (knowledge distillation).",
        "<b>Stability.</b> Each model keeps several snapshots during training and the deployed weights are their "
        "average, which smooths out the epoch-to-epoch swings of a small model.",
    ]),
    ("h1", "3. Model architecture"),
    ("img", "architecture"),
    ("caption", "Knowledge distillation. Teacher and student see the same photo. The student is corrected by the "
                "reviewed labels and by three signals from the teacher; only the student is updated."),
    ("table", [["", "Teacher", "Student (deployed)"],
               ["Network", "YOLO11-L instance segmentation", "YOLO11-N instance segmentation"],
               ["Size", "27.6 M parameters · 56 MB", "2.84 M parameters · 6 MB"],
               ["Learns from", "Reviewed labels", "Reviewed labels + the teacher's features, class confidences and box edges"],
               ["Role", "Accuracy reference; server option", "Default model in the web app, browser demo and Android app"],
               ["Runs on", "Server GPU or CPU", "Browser, phone/tablet, edge device, server"]]),
    ("h1", "4. Benchmark"),
    ("img", "benchmark"),
    ("caption", "Left: POC v1 models. Known vehicles = 3-fold cross-validation on new photos of the 28 vehicles; unseen "
                "vehicles = 3 vehicles never used in training (student: mean of 2 seeds). Right: the score on unseen "
                "vehicles rose steadily up to p5t (student +37%, teacher +52% over v4) and has been flat since. v10 is scored on corrected test labels."),
    ("table", [["Measure", "Teacher", "Student"],
               ["Accuracy, known vehicles (mask mAP50-95)", "0.42", "0.38"],
               ["Accuracy, unseen vehicles (mask mAP50-95)", "0.35", "0.27–0.30"],
               ["Precision / recall, known vehicles (per-class thresholds)", "0.74 / 0.67", "0.72 / 0.61"],
               ["Network time per photo, server GPU", "9.2 ms", "2.9 ms"],
               ["Time per photo, web app on a laptop CPU", "1.8–2.7 s", "0.35–0.6 s"],
               ["Time per photo, in the web browser (no server)", "–", "≈ 0.2 s"]]),
    ("p", "<b>How to read the score.</b> mask mAP50-95 compares each predicted outline with the true outline at "
          "strict and loose overlap levels (0 = nothing found, 1 = perfect). With a lenient overlap of 50% the "
          "student scores 0.63 on known vehicles and about 0.5 on unseen vehicles. Distillation adds 1–3 points over "
          "the same small model trained without a teacher."),
    ("h1", "5. Strengths"),
    ("bullets", [
        "<b>Small and fast.</b> A 6 MB model that answers in about 0.2 s inside a web browser; photos never leave the device.",
        "<b>Distillation works.</b> The student keeps about 91% of the teacher's accuracy at one tenth of the size, "
        "and always beats the same model trained alone.",
        "<b>Reliable on distinctive parts</b> such as the radiator cap and the washer and brake-fluid reservoirs, even on new vehicles.",
        "<b>Trustworthy measurement.</b> Labels are expert-reviewed with measured error rates, and model changes are "
        "accepted only if a paired statistical test shows they are better, not just higher by chance.",
        "<b>Reproducible.</b> Dataset builds, training, evaluation and deployment are scripted end to end.",
    ]),
    ("h1", "6. Weaknesses"),
    ("bullets", [
        "<b>Limited generalisation to new vehicles</b> (0.27–0.30), and it has plateaued: four experiments on the same "
        "28 vehicles brought no measurable gain.",
        "<b>Small test set.</b> Only 3 unseen vehicles, so a single score is uncertain by about ±0.05 and small "
        "improvements cannot be confirmed.",
        "<b>Weak and confusable classes.</b> Alternator, radiator hose, ECU and coolant reservoir are often missed; "
        "look-alike parts (brake-fluid vs coolant reservoir, intake manifold vs engine cover) are sometimes swapped.",
        "<b>Remaining label errors.</b> Audits found wrong outlines in up to 37% of battery labels; part of the "
        "missed detections on the test set are label errors rather than model errors.",
        "<b>Not yet validated on the target hardware</b> with optimised runtimes (TensorRT FP16/INT8).",
    ]),
    ("h1", "7. Recommended improvements"),
    ("table", [["#", "Action", "Expected effect"],
               ["1", "Collect 40–60 new vehicles (about 30 photos each) and enlarge the test and validation sets to "
                     "at least 6 vehicles each", "Largest expected gain on unseen vehicles; makes smaller improvements measurable"],
               ["2", "Retrain on the fully corrected labels (POC v1.1 candidate)", "The partial correction already "
                     "shows +0.01 for the student (94% probability of a real gain); to be confirmed"],
               ["3", "Extend the label audit to the weak classes", "Removes a known error source in both training and scoring"],
               ["4", "Add hard negatives (loose spark plugs, connectors, underbody photos) and an active-learning loop "
                     "where the teacher pre-labels and experts review only disagreements", "Fewer false alarms; cheaper labelling of new data"],
               ["5", "Validate on the target device (TensorRT FP16/INT8, Android tablet)", "Confirms real speed and "
                     "accuracy after optimisation"]]),
    ("p", "<b>Stop for now:</b> further tuning of the distillation loss and changes to the class list on the current "
          "data. Both were tested and showed no gain."),
]


# ---------------------------------------------------------------- PDF
def build():
    figs = {"pipeline": fig_pipeline(), "architecture": fig_architecture(), "benchmark": fig_benchmark()}
    ss = {
        "title": ParagraphStyle("t", fontName="Arial-Bold", fontSize=24, leading=29, textColor=C(INK)),
        "sub": ParagraphStyle("s", fontName="Arial", fontSize=14, leading=19, textColor=C(TEACHER)),
        "meta": ParagraphStyle("m", fontName="Arial", fontSize=9.2, leading=13, textColor=C(MUTED)),
        "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=14, leading=18, textColor=C(INK), spaceBefore=10,
                             spaceAfter=5, keepWithNext=1),
        "p": ParagraphStyle("p", fontName="Arial", fontSize=9.8, leading=14.2, textColor=C(INK), spaceAfter=6, alignment=TA_LEFT),
        "b": ParagraphStyle("b", fontName="Arial", fontSize=9.8, leading=14, textColor=C(INK), leftIndent=12,
                            bulletIndent=2, spaceAfter=3),
        "cap": ParagraphStyle("c", fontName="Arial-Italic", fontSize=8.3, leading=11, textColor=C(MUTED), spaceAfter=8),
        "td": ParagraphStyle("td", fontName="Arial", fontSize=8.6, leading=11, textColor=C(INK)),
        "th": ParagraphStyle("th", fontName="Arial-Bold", fontSize=8.6, leading=11, textColor=C(INK)),
        "big": ParagraphStyle("bg", fontName="Arial-Bold", fontSize=20, leading=24, textColor=C(STUDENT)),
        "bigt": ParagraphStyle("bgt", fontName="Arial-Bold", fontSize=20, leading=24, textColor=C(TEACHER)),
        "small": ParagraphStyle("sm", fontName="Arial", fontSize=8.3, leading=11, textColor=C(MUTED)),
    }
    W = A4[0] - 36 * mm
    story = [Spacer(1, 14 * mm), Paragraph("Engine-Bay Component Segmentation", ss["title"]), Spacer(1, 3),
             Paragraph("Training overview: approach, benchmark and next steps", ss["sub"]), Spacer(1, 8),
             Paragraph("Distillation project · status 2 October 2026 (POC v1) · high-level companion to the "
                       "Engine-Bay KD Training Report", ss["meta"]), Spacer(1, 9 * mm)]
    kv = [("0.38", "big", "Student accuracy on new photos of known vehicles"),
          ("0.27–0.30", "big", "Student accuracy on unseen vehicles"),
          ("10× smaller", "bigt", "2.84 M vs 27.6 M parameters, about 91% of the teacher's accuracy"),
          ("≈ 0.2 s", "bigt", "Per photo in a web browser, no server")]
    t = Table([[Paragraph(v, ss[s]) for v, s, _ in kv], [Paragraph(d, ss["small"]) for _, _, d in kv]], colWidths=[W / 4] * 4)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 1.2, C(LINE)),
                           ("TOPPADDING", (0, 0), (-1, 0), 8), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    story += [t, Spacer(1, 3), Paragraph("Accuracy = mask mAP50-95 (0 to 1), explained in section 4.", ss["small"]),
              Spacer(1, 4 * mm)]

    def is_h1(flowable):
        return isinstance(flowable, Paragraph) and flowable.style.name == "h1"

    for kind, payload in BODY:
        if kind == "h1":
            story.append(Paragraph(payload, ss["h1"]))
        elif kind == "p":
            story.append(Paragraph(payload, ss["p"]))
        elif kind == "caption":  # a caption never leaves its figure, nor the figure a heading right above it
            n = 3 if is_h1(story[-3]) else 2
            group = story[-n:] + [Paragraph(payload, ss["cap"])]
            del story[-n:]
            story.append(KeepTogether(group))
        elif kind == "bullets":
            for b in payload:
                story.append(Paragraph(b, ss["b"], bulletText="•"))
            story.append(Spacer(1, 4))
        elif kind == "img":
            p = figs[payload]
            w, h = PIL.Image.open(p).size
            story += [Image(str(p), width=W, height=W * h / w), Spacer(1, 4)]
        elif kind == "table":
            rows = [[Paragraph(c, ss["th"] if i == 0 else ss["td"]) for c in r] for i, r in enumerate(payload)]
            head = payload[0][0]
            cw = ([W * 0.05, W * 0.55, W * 0.40] if head == "#" else
                  [W * 0.52, W * 0.24, W * 0.24] if head == "Measure" else [W * 0.18, W * 0.36, W * 0.46])
            tb = Table(rows, colWidths=cw, repeatRows=1)
            tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), C("#E3EAE8")), ("LINEBELOW", (0, 0), (-1, -1), 0.4, C(LINE)),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
            lead = [story.pop()] if is_h1(story[-1]) else []
            story += [KeepTogether(lead + [tb]), Spacer(1, 6)]  # a heading never sits alone above its table

    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 8)
        canvas.setFillColor(C(MUTED))
        if doc.page > 1:
            canvas.drawString(18 * mm, A4[1] - 12 * mm, "Engine-bay KD training overview")
            canvas.setStrokeColor(C(LINE))
            canvas.line(18 * mm, A4[1] - 14 * mm, A4[0] - 18 * mm, A4[1] - 14 * mm)
        else:
            canvas.setFillColor(C(TEACHER))
            canvas.rect(0, A4[1] - 8 * mm, A4[0], 8 * mm, fill=1, stroke=0)
            canvas.setFillColor(C(STUDENT))
            canvas.rect(0, A4[1] - 10 * mm, A4[0] * 0.3, 2 * mm, fill=1, stroke=0)
        canvas.setFillColor(C(MUTED))
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    out = OUT / FILE
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm,
                            bottomMargin=18 * mm, title="Engine-Bay Component Segmentation — Training overview",
                            author="Distillation project", subject="High-level training overview")
    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    return out


if __name__ == "__main__":
    print(build())
