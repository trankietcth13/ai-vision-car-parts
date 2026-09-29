"""
Build the engine-bay distillation training report as PDF, in English and Vietnamese.

Inputs (all from real runs): qa_results/version_report/* (QA reports pulled from the DGX), qa_results/curves/*.csv
(per-epoch results), qa_results/test_cases_kd_full_c025/summary.json, scratch per-class JSON (embedded below).
Output: reports/Engine_Bay_KD_Training_Report_EN.pdf and _VI.pdf, charts in reports/fig/.

Usage: python reports/build_training_report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports"
FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- fonts / colours
FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Arial", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Consolas", str(FONTS / "consola.ttf")))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8A9A97", "axes.labelcolor": "#14221F", "xtick.color": "#3B4B48",
                     "ytick.color": "#3B4B48", "savefig.dpi": 200})
INK, MUTED, LINE = "#14221F", "#586966", "#C9D4D1"
TEACHER, STUDENT, BASE, LOSS, GT, GRAD = "#0B7A71", "#C77A00", "#8A9A97", "#C2410C", "#6B4FB8", "#2458C6"
C = colors.HexColor

# ---------------------------------------------------------------- data (real runs)
VERSIONS = [  # name, teacher mAP50-95, student KD mAP50-95 (seed mean), student mAP50, baseline, test cases pass/100
    ("v4", 0.233, 0.209, 0.382, 0.198, None),
    ("v5", 0.219, 0.193, 0.344, 0.165, None),
    ("v6", 0.259, 0.245, 0.423, 0.218, 47),
    ("v7", 0.258, 0.249, 0.441, None, 47),
    ("v8", 0.290, 0.264, 0.466, None, 48),
    ("kd_n_p5t", 0.354, 0.286, 0.491, None, 58),
]
PHASE5 = [("teacher v8", 0.290, None, 0.479), ("freeze 10", 0.310, 0.338, 0.557), ("SGD lr0 0.01", 0.285, 0.319, 0.510),
          ("wd 0.001 + scale 0.7", 0.290, 0.354, 0.577)]
PER_CLASS = json.loads((ROOT / "reports" / "perclass.json").read_text()) if (ROOT / "reports" / "perclass.json").exists() else None
FINAL = json.loads((ROOT / "qa_results" / "test_cases_kd_full_c025" / "summary.json").read_text())


def curves():
    def load(n):
        with open(ROOT / "qa_results" / "curves" / f"{n}.csv") as f:
            return [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    kd, t = load("kd_n_p5t_s0"), load("p5_reg")
    task = [sum(float(r[k]) for k in ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss")) for r in kd]
    return {"task": task, "feat": [float(r["train/kd_feat"]) for r in kd], "cls": [float(r["train/kd_cls"]) for r in kd],
            "loc": [float(r["train/kd_loc"]) for r in kd], "s_map": [float(r["metrics/mAP50-95(M)"]) for r in kd],
            "t_map": [float(r["metrics/mAP50-95(M)"]) for r in t]}


CURVES = curves()


def dec(v, lang, nd=3):
    s = f"{v:.{nd}f}"
    return s.replace(".", ",") if lang == "vi" else s


# ---------------------------------------------------------------- charts
def chart_versions(L, lang):
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    x = range(len(VERSIONS))
    w = 0.26
    for off, idx, col, lab in ((-w, 1, TEACHER, L["c_teacher"]), (0, 2, STUDENT, L["c_student"]), (w, 4, BASE, L["c_base"])):
        xs = [i + off for i, v in enumerate(VERSIONS) if v[idx] is not None]
        ys = [v[idx] for v in VERSIONS if v[idx] is not None]
        bars = ax.bar(xs, ys, w * 0.92, color=col, label=lab)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 0.005, dec(y, lang), ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(0.30, color=LOSS, ls="--", lw=1)
    ax.text(-0.45, 0.306, L["c_target"], color=LOSS, fontsize=7.5, ha="left")
    ax.set_xticks(list(x), [v[0] for v in VERSIONS])
    ax.set_ylim(0, 0.4)
    ax.set_ylabel(L["c_map"])
    ax.grid(axis="y", color="#DCE4E2", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    fig.tight_layout()
    p = FIG / f"versions_{lang}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_curves(L, lang):
    c = CURVES
    ep = list(range(1, len(c["task"]) + 1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    a1.plot(ep, c["t_map"], color=TEACHER, lw=1.6, label=L["c_teacher_short"])
    a1.plot(ep, c["s_map"], color=STUDENT, lw=1.6, label=L["c_student_short"])
    a1.axvspan(90.5, 100, color="#F6E4C2", alpha=0.7, lw=0)
    a1.text(95, 0.02, L["c_nomosaic"], ha="center", fontsize=7, color=MUTED)
    a1.set_xlabel("epoch")
    a1.set_ylabel(L["c_valmap"])
    a1.set_ylim(0, 0.4)
    a1.grid(color="#DCE4E2", lw=0.8)
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    for k, col, lab in (("task", GT, L["c_task"]), ("feat", LOSS, "L_feat"), ("cls", "#E08A5B", "L_cls"), ("loc", "#7A2E0B", "L_loc")):
        v = c[k]
        a2.plot(ep, [x / v[0] for x in v], color=col, lw=1.4, label=lab)
    a2.set_xlabel("epoch")
    a2.set_ylabel(L["c_rel"])
    a2.set_ylim(0, 1.15)
    a2.grid(color="#DCE4E2", lw=0.8)
    a2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = FIG / f"curves_{lang}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_perclass(L, lang):
    s, t = PER_CLASS["kd_n_p5t_s0_avg5"], PER_CLASS["teacher_p5_reg_avg5"]
    names = sorted(s, key=lambda k: s[k])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], [t[n] for n in names], 0.38, color=TEACHER, label=L["c_teacher_short"])
    ax.barh([i - 0.2 for i in y], [s[n] for n in names], 0.38, color=STUDENT, label=L["c_student_short"])
    ax.set_yticks(list(y), [L["cls"].get(n, n) for n in names], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(L["c_ap50"])
    ax.grid(axis="x", color="#DCE4E2", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    p = FIG / f"perclass_{lang}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_phase5(L, lang):
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    names = [L["p5"][i] for i in range(len(PHASE5))]
    y = range(len(PHASE5))
    ax.barh([i + 0.2 for i in y], [p[1] for p in PHASE5], 0.38, color=BASE, label=L["c_best"])
    ax.barh([i - 0.2 for i in y], [p[2] or 0 for p in PHASE5], 0.38, color=TEACHER, label=L["c_avg"])
    for i, p in enumerate(PHASE5):
        ax.text(p[1] + 0.004, i + 0.2, dec(p[1], lang), va="center", fontsize=7)
        if p[2]:
            ax.text(p[2] + 0.004, i - 0.2, dec(p[2], lang), va="center", fontsize=7, color=TEACHER)
    ax.set_yticks(list(y), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 0.39)
    ax.set_xlabel(L["c_map"])
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.tight_layout()
    p = FIG / f"phase5_{lang}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def chart_kd(L, lang):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 46)
    ax.axis("off")

    def box(x, y, w, h, txt, fc, ec, fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(p, q, col, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=col, lw=1.1, ls=ls))

    box(1, 19, 13, 9, L["d_img"], "#EEF2F1", LINE)
    box(1, 6, 13, 7, L["d_gt"], "#E3DBF5", GT)
    box(22, 32, 30, 10, L["d_teacher"], "#CFE9E5", TEACHER, bold=True)
    box(22, 4, 30, 10, L["d_student"], "#F6E4C2", STUDENT, bold=True)
    box(28, 18, 18, 7, L["d_adapter"], "#FFFFFF", LINE, fs=7.5)
    for i, (t, yy) in enumerate(((L["d_feat"], 36), (L["d_cls"], 27.5), (L["d_loc"], 19), (L["d_task"], 6))):
        box(62, yy, 22, 6.5, t, "#F7D9CB" if i < 3 else "#E3DBF5", LOSS if i < 3 else GT, fs=7.5)
    box(90, 20, 8.5, 8, "Σ", "#F7D9CB", LOSS, fs=12, bold=True)
    arrow((14, 25), (22, 36), TEACHER)
    arrow((14, 22), (22, 9), STUDENT)
    arrow((52, 38), (62, 39), TEACHER)
    arrow((52, 35), (62, 31), TEACHER)
    arrow((52, 34), (62, 22), TEACHER)
    arrow((37, 14), (37, 18), STUDENT)
    arrow((46, 23), (62, 38), STUDENT)
    arrow((52, 11), (62, 29), STUDENT)
    arrow((52, 9), (62, 21), STUDENT)
    arrow((52, 7), (62, 9), STUDENT)
    ax.add_patch(FancyArrowPatch((8, 6), (62, 8), arrowstyle="-|>", mutation_scale=9, color=GT, lw=1.1,
                                 connectionstyle="arc3,rad=0.28"))
    for yy in (39, 30.5, 22, 9):
        arrow((84, yy), (90, 24), LOSS)
    ax.plot([94.2, 94.2, 37], [20, -2, -2], color=GRAD, lw=1.2, ls="--")
    arrow((37, -2), (37, 4), GRAD, ls="--")
    ax.text(66, -4.6, L["d_grad"], fontsize=7.5, color=GRAD, ha="center")
    ax.text(37, 44.5, L["d_frozen"], fontsize=7.5, color=TEACHER, ha="center")
    fig.tight_layout()
    p = FIG / f"kd_{lang}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- text
CLS_VI = {"battery": "Ắc quy", "battery_terminal": "Cọc bình", "fuse_relay_box": "Hộp cầu chì", "coolant_reservoir": "Bình nước làm mát",
          "radiator_cap": "Nắp két nước", "brake_fluid_reservoir": "Bình dầu phanh", "washer_fluid_reservoir": "Bình nước rửa kính",
          "engine_cover": "Nắp che động cơ", "oil_filler_cap": "Nắp châm dầu", "oil_dipstick": "Que thăm dầu",
          "air_filter_box": "Hộp lọc gió", "air_intake_duct": "Ống hút gió", "maf_sensor": "Cảm biến MAF", "throttle_body": "Cổ họng ga",
          "alternator": "Máy phát", "ignition_coil": "Bô-bin", "radiator_hose": "Ống két nước", "ecu_module": "Hộp ECU",
          "multimeter_diagnostic_tool": "Đồng hồ đo / chẩn đoán", "intake_manifold": "Cổ hút"}
CLS_EN = {k: k.replace("_", " ") for k in CLS_VI}
CLS_EN["multimeter_diagnostic_tool"] = "multimeter / diagnostic tool"
CLS_EN["maf_sensor"] = "MAF sensor"
CLS_EN["ecu_module"] = "ECU module"

EN = dict(
    lang="en", cls=CLS_EN, file="Engine_Bay_KD_Training_Report_EN.pdf",
    title="Engine-Bay Component Segmentation", subtitle="Knowledge-distillation training report",
    meta="Distillation project · 29 September 2026 · all numbers from runs on the DGX GB10",
    header="Engine-bay KD training report", page="Page",
    c_teacher="Teacher yolo11l-seg", c_student="KD student yolo11n-seg (seed mean)", c_base="Student trained alone",
    c_target="student target 0.30", c_map="mask mAP50-95 (test, 3 unseen vehicles)", c_teacher_short="teacher",
    c_student_short="student", c_nomosaic="mosaic off", c_valmap="mask mAP50-95 (val)", c_task="label loss",
    c_rel="loss relative to epoch 1", c_ap50="mask AP50 on the 3 test vehicles", c_best="best checkpoint",
    c_avg="average of 5 checkpoints",
    p5=["teacher v8 (reference)", "freeze 10 backbone layers", "SGD, lr0 0.01", "weight decay 0.001, scale 0.7"],
    d_img="Engine-bay\nimage batch", d_gt="Reviewed\nlabels", d_teacher="Teacher yolo11l-seg\n27.6M params",
    d_student="Student yolo11n-seg\n2.84M params", d_adapter="1×1 adapters", d_feat="L_feat · FGD", d_cls="L_cls · KL, T=2",
    d_loc="L_loc · LD, T=10", d_task="L_task · labels", d_grad="gradients update the student + adapters only",
    d_frozen="frozen: no weight updates",
)
VI = dict(
    lang="vi", cls=CLS_VI, file="Engine_Bay_KD_Training_Report_VI.pdf",
    title="Nhận diện linh kiện khoang máy", subtitle="Báo cáo huấn luyện bằng knowledge distillation",
    meta="Dự án Distillation · 29/09/2026 · mọi số liệu từ các lần chạy trên DGX GB10",
    header="Báo cáo huấn luyện KD khoang máy", page="Trang",
    c_teacher="Teacher yolo11l-seg", c_student="Student KD yolo11n-seg (TB seed)", c_base="Student train một mình",
    c_target="mục tiêu student 0,30", c_map="mask mAP50-95 (test, 3 xe chưa thấy)", c_teacher_short="teacher",
    c_student_short="student", c_nomosaic="tắt mosaic", c_valmap="mask mAP50-95 (val)", c_task="loss nhãn",
    c_rel="loss so với epoch 1", c_ap50="mask AP50 trên 3 xe test", c_best="checkpoint tốt nhất",
    c_avg="trung bình 5 checkpoint",
    p5=["teacher v8 (mốc)", "freeze 10 lớp backbone", "SGD, lr0 0,01", "weight decay 0,001, scale 0,7"],
    d_img="Batch ảnh\nkhoang máy", d_gt="Nhãn\nđã review", d_teacher="Teacher yolo11l-seg\n27,6 triệu tham số",
    d_student="Student yolo11n-seg\n2,84 triệu tham số", d_adapter="Adapter 1×1", d_feat="L_feat · FGD", d_cls="L_cls · KL, T=2",
    d_loc="L_loc · LD, T=10", d_task="L_task · nhãn", d_grad="gradient chỉ cập nhật student + adapter",
    d_frozen="đóng băng: không cập nhật trọng số",
)


def body(lang):
    """Section content per language: list of (kind, payload)."""
    f = FINAL
    if lang == "en":
        return [
            ("h1", "1. Executive summary"),
            ("p", "We trained an instance-segmentation model that finds 20 engine-bay components in inspection photos, and distilled it into a model small enough for edge devices. A large teacher (yolo11l-seg, 27.6M parameters) learns from expert-reviewed labels; a student ten times smaller (yolo11n-seg, 2.84M parameters, 6 MB) learns from the same labels and imitates the teacher's features, class confidences and box boundaries."),
            ("bullets", [
                "<b>New vehicles (3 held-out cars, 125 images):</b> best student mask mAP50-95 <b>0.302</b> (mAP50 0.514; 2-seed mean 0.286), best teacher <b>0.354</b>. In the per-image test, 58 of 100 images pass.",
                "<b>Progress since the first measured version (v4):</b> student 0.209 → 0.302 (+44%), teacher 0.233 → 0.354 (+52%).",
                f"<b>Final model on all 28 vehicles (kd_n_full):</b> on the 1,081 dataset images it was trained on, {f['pass']}/{f['images']} images pass, recall {f['recall']:.3f}, precision {f['precision']:.3f}. Cross-validation numbers for unseen images of these vehicles follow tonight.",
                "<b>Speed:</b> 4.0 ms per image end to end (PyTorch FP16 on the DGX GB10), 2.9 ms for the network alone.",
                "<b>Recommendation:</b> use the student as an assistant that proposes components for a technician to confirm on new vehicles; use it automatically on the vehicles in the dataset once cross-validation confirms the accuracy. The main lever for new vehicles is more distinct vehicles in the training data.",
            ]),
            ("h1", "2. Data and labels"),
            ("p", "The dataset holds 1,307 photos of 28 vehicles (one Request_ID per vehicle), taken during inspections at 6000×4000 px. The training ontology has 20 classes derived from 36 annotation classes; rare classes are dropped and upper and lower radiator hoses are merged."),
            ("table", [["Stage", "Method", "Result"],
                       ["Machine boxes", "Qwen3-VL-30B; DeepSeek V4 Flash Vision for hard cases", "Box F1 0.41 (Qwen) and 0.45 (DeepSeek) on reviewed images"],
                       ["Masks", "SAM 2.1 from each box", "Polygon masks along the real outline"],
                       ["Expert review", "Claude review skill per image: keep, reclass, re-box, drop, add missing", "918 images reviewed in rounds 1–2"],
                       ["Phase 2 relabel", "332 images without reviewed labels: DeepSeek + Codex drafts, full review", "139 excluded (underbody, fuel door), 193 kept; DeepSeek box precision 0.23"],
                       ["Independent verification", "Second reviewer sees only model/label disagreements", "287 disagreements → 70 label fixes"],
                       ["Empty-image recheck", "Teacher suggestions + review of 134 unlabelled images", "30 more excluded, 23 labels added"],
                       ["Final pool", "Duplicates grouped, consistent scene policy", "1,081 images, 28 vehicles, 13 duplicate photos grouped"]]),
            ("p", "Splits: for new-vehicle measurement the vehicles are split 19 train / 6 validation / 3 test; test images are never trained on. For the final model all 28 vehicles are used and accuracy is measured with image-level 3-fold cross-validation. A 50-image human spot check of the final labels reported no errors (awaiting the reviewer's confirmation)."),
            ("h1", "3. Algorithm"),
            ("img", "kd"),
            ("p", "<b>Knowledge distillation.</b> For every batch the frozen teacher and the student see the same images. The student minimises"),
            ("mono", "L = L_task + 1.0·L_feat + 1.0·L_cls + 1.0·L_loc"),
            ("bullets", [
                "<b>L_task</b>: the standard Ultralytics segmentation loss (box, mask, class, DFL) against the reviewed labels.",
                "<b>L_feat</b>: Focal and Global Distillation on the neck features P3/P4/P5. 1×1 adapters project student channels to teacher channels; label boxes split each map into component (weight 1.0) and background (0.5) regions; attention and global context terms (λ 0.5, temperature 0.5).",
                "<b>L_cls</b>: binary KL divergence between teacher and student class probabilities, softened with temperature 2, background anchors weighted 0.05.",
                "<b>L_loc</b>: localization distillation on the DFL distributions of the four box edges, temperature 10.",
                "Epoch 1 trains only the adapters so the untrained projection does not disturb the student. Adapters are removed after training, leaving a plain yolo11n-seg model.",
            ]),
            ("p", "<b>Training recipe (final).</b> 640 px, batch 16, 100 epochs, cosine learning rate, copy-paste 0.3, mixup 0.1, rotation ±5°, mosaic off for the last 10 epochs, repeat-factor sampling for rare classes (t = 0.1, at most 4 repeats). The teacher adds weight decay 0.001, scale 0.7 and perspective 0.0005. Both models save a checkpoint every 5 epochs; the deployed weights are the average of 5 checkpoints."),
            ("h1", "4. Training process"),
            ("p", "Hardware: NVIDIA DGX GB10 with 121 GB unified memory, shared with a vLLM service, so one job runs at a time. Software: PyTorch 2.14, Ultralytics 8.4.75. A teacher takes about 90 minutes and a student about 60 minutes."),
            ("img", "curves"),
            ("caption", "Real run kd_n_p5t_s0. Left: validation mask mAP50-95 per epoch for the student and its teacher. Right: each loss component relative to epoch 1. The student's mAP swings by several points between epochs, which is why checkpoint averaging helps."),
            ("h1", "5. Benchmark across versions"),
            ("img", "versions"),
            ("table", [["Version", "Main change", "Teacher", "Student KD", "Student mAP50", "Alone", "Pass/100"],
                       ["v4", "324 reviewed train images, teacher at 1024 px", "0.233", "0.209", "0.382", "0.198", "–"],
                       ["v5", "Codex fixed 101 labels; val 3 → 6 vehicles; 60-epoch teacher stopped early", "0.219", "0.193", "0.344", "0.165", "–"],
                       ["v6", "640 px, cosine LR, copy-paste, mixup, rare-class sampling, re-pseudo-labelling", "0.259", "0.245", "0.423", "0.218", "47"],
                       ["v7", "+175 reviewed external images (train only)", "0.258", "0.249", "0.441", "–", "47"],
                       ["v8", "344 pseudo-labels replaced by reviewed hybrid labels", "0.290", "0.264", "0.466", "–", "48"],
                       ["kd_n_p5t", "Regularised teacher + checkpoint averaging for both models", "0.354", "0.286", "0.491", "–", "58"],
                       ["full", "All 28 vehicles, 1,081 verified images", "n/a", "n/a", "n/a", "–", "CV"]]),
            ("caption", "mask mAP50-95 on the same 125 test images of 3 unseen vehicles. Student KD is the mean of 2 seeds from v6 on. v4 and v5 used test labels before Codex's fixes. 'Pass/100' = images with precision ≥ 0.5 and recall ≥ 0.5 at confidence 0.25. v1–v3 used unreviewed machine labels and no fixed test set, so they are not comparable."),
            ("h2", "Phase 5: one change at a time on the teacher"),
            ("img", "phase5"),
            ("caption", "Averaging the 5 best checkpoints (chosen on validation, scored on test) adds 3–6 points; the regularised teacher with averaging is the best teacher."),
            ("h2", "Per-class accuracy on new vehicles"),
            ("img", "perclass"),
            ("caption", "Best student (kd_n_p5t seed 0, averaged) vs best teacher. Classes with 3–7 test instances (coolant reservoir, radiator hose, washer reservoir, ECU) have large uncertainty."),
            ("h2", "Speed"),
            ("table", [["Model", "Params", "Size", "Network", "End to end", "FPS"],
                       ["Student, PyTorch FP16 (DGX GB10)", "2.84 M", "5.7 MB", "2.9 ms", "4.0 ms", "248"],
                       ["Student, ONNX on CPU (DGX)", "2.84 M", "11.1 MB", "36.5 ms", "47.4 ms", "21"],
                       ["Teacher, PyTorch (batch 1)", "27.6 M", "56 MB", "9.2 ms", "–", "–"]]),
            ("caption", "20 warm-up calls, 200 timed calls on real images. TensorRT must be built on the target device and has not been measured yet."),
            ("h1", "6. Final model"),
            ("p", f"<b>kd_n_full</b> (student) and <b>teacher_full</b> are trained on all 1,081 verified images of the 28 vehicles with the kd_n_p5t recipe, averaging the last 5 checkpoints. On the dataset images (which the model has seen): {f['pass']} of {f['images']} images pass (94.1%), {f['tp']} of {f['instances']} components found (recall {f['recall']:.3f}), {f['fp']} extra detections at a shared threshold of 0.25 (precision {f['precision']:.3f}). These numbers show fit, not generalisation. 3-fold cross-validation (train on two thirds, score on the rest) and per-class confidence thresholds run tonight; a v9 run with the new recipe and the 3 test vehicles held out gives the matching new-vehicle number tomorrow."),
            ("p", "Deliverables: runs/train_kd/kd_n_full/weights/best.pt (6 MB), scripts/deployment/weights/kd_n_full_640.onnx, runs/segment/teacher_full/weights/best.pt (server option), web UI defaulting to the student."),
            ("h1", "7. Strengths"),
            ("bullets", [
                "<b>Distillation always helps.</b> The KD student beats the same student trained alone by 1–3 mask mAP50-95 points in every measured version, and reaches 80–95% of the teacher at one tenth of the size.",
                "<b>Fast.</b> 4 ms per image end to end on the DGX; the 6 MB model fits edge devices.",
                "<b>Reliable on distinctive parts.</b> Multimeter, radiator cap, washer and brake-fluid reservoirs reach mask AP50 of about 0.8 on new vehicles.",
                "<b>Clean, audited labels.</b> Every label in the final pool passed expert review, most passed independent verification, and the review process measured machine-label precision instead of assuming it.",
                "<b>Reproducible pipeline.</b> Scripted dataset builds, chained DGX jobs, QA reports and per-image test galleries for every version.",
            ]),
            ("h1", "8. Weaknesses and risks"),
            ("bullets", [
                "<b>Generalisation to new vehicles is limited</b> (student mAP50-95 about 0.29, mAP50 about 0.5). 19 training vehicles make the effective diversity small; the gap between seen images (mAP50 0.94) and new vehicles is large.",
                "<b>Small test set.</b> 3 vehicles and 125 images; two seeds of one recipe differ by up to 3 points, so differences below about 2 points are not reliable.",
                "<b>Weak classes:</b> coolant reservoir (AP50 0.01), fuse/relay box, battery, ECU, MAF sensor. Look-alike confusions remain: brake fluid vs coolant reservoir, intake manifold vs engine cover, loose spark plugs detected as ignition coils.",
                "<b>Extra detections</b> at a single threshold (precision 0.58 on new vehicles, 0.75 on the dataset); per-class thresholds are not applied yet.",
                "<b>Deployment not validated on the target device</b>; TensorRT latency and accuracy after FP16/INT8 conversion are unknown.",
            ]),
            ("h1", "9. Recommended next steps"),
            ("bullets", [
                "<b>Collect 40–60 new vehicles</b> with the capture guide (about 30 photos each, same photo style) and grow the test and validation splits to at least 6 vehicles each. This is the largest expected gain for new vehicles.",
                "<b>Apply per-class confidence thresholds</b> from cross-validation in the app and in test cases.",
                "<b>Hard-negative mining:</b> add loose spark plugs, connectors, EVAP/MAP sensors and underbody shots as explicit negatives; they cause most false detections.",
                "<b>Active-learning loop:</b> pre-label new photos with the teacher, review only disagreements (the verification queue already does this), retrain.",
                "<b>Use unlabelled photos</b> through feature-only distillation (supported in the trainer, not yet used).",
                "<b>Model size by target:</b> if deployment is on a server, try a yolo11s student or deploy the teacher; if on Jetson, build TensorRT FP16/INT8 on the device and re-measure accuracy.",
                "<b>Measurement:</b> report 2–3 seeds for every change, and keep the 3-vehicle held-out test fixed for new-vehicle tracking.",
                "<b>External data:</b> only engine-bay photos with licences that allow commercial training (iFixit images are CC BY-NC-SA).",
            ]),
        ]
    return [
        ("h1", "1. Tóm tắt"),
        ("p", "Dự án huấn luyện một model instance segmentation tìm 20 loại linh kiện khoang máy trong ảnh kiểm tra xe, rồi chưng cất (distill) sang một model đủ nhỏ để chạy trên thiết bị. Teacher lớn (yolo11l-seg, 27,6 triệu tham số) học từ nhãn đã được chuyên gia review. Student nhỏ hơn 10 lần (yolo11n-seg, 2,84 triệu tham số, 6 MB) học từ cùng bộ nhãn và bắt chước đặc trưng, độ tin cậy theo lớp và đường biên hộp của teacher."),
        ("bullets", [
            "<b>Xe mới (3 xe giữ riêng, 125 ảnh):</b> student tốt nhất đạt mask mAP50-95 <b>0,302</b> (mAP50 0,514; trung bình 2 seed 0,286). Teacher tốt nhất đạt <b>0,354</b>. Trong kiểm thử theo ảnh, 58/100 ảnh đạt.",
            "<b>Tiến bộ từ phiên bản đo được đầu tiên (v4):</b> student 0,209 → 0,302 (+44%), teacher 0,233 → 0,354 (+52%).",
            f"<b>Model cuối trên cả 28 xe (kd_n_full):</b> trên 1.081 ảnh dataset mà model đã học, {f['pass']}/{f['images']} ảnh đạt, recall {dec(f['recall'], 'vi')}, precision {dec(f['precision'], 'vi')}. Số cross-validation cho ảnh chưa thấy của các xe này sẽ có tối nay.",
            "<b>Tốc độ:</b> 4,0 ms mỗi ảnh toàn pipeline (PyTorch FP16 trên DGX GB10), 2,9 ms cho riêng mạng.",
            "<b>Khuyến nghị:</b> trên xe mới, dùng student như trợ lý đề xuất linh kiện để kỹ thuật viên xác nhận. Trên các xe trong dataset, dùng tự động sau khi cross-validation xác nhận độ chính xác. Đòn bẩy chính cho xe mới là thêm nhiều xe khác nhau vào dữ liệu train.",
        ]),
        ("h1", "2. Dữ liệu và nhãn"),
        ("p", "Dataset gồm 1.307 ảnh của 28 xe (mỗi xe một Request_ID), chụp khi kiểm tra ở độ phân giải 6000×4000 px. Bộ lớp huấn luyện có 20 lớp, rút từ 36 lớp chú thích: bỏ các lớp quá hiếm và gộp ống két nước trên và dưới."),
        ("table", [["Bước", "Cách làm", "Kết quả"],
                   ["Hộp máy", "Qwen3-VL-30B; DeepSeek V4 Flash Vision cho ca khó", "F1 hộp 0,41 (Qwen) và 0,45 (DeepSeek) trên ảnh đã review"],
                   ["Mask", "SAM 2.1 từ từng hộp", "Polygon theo đường viền thật"],
                   ["Review chuyên gia", "Skill review của Claude cho từng ảnh: giữ, đổi lớp, sửa hộp, bỏ, bổ sung", "918 ảnh review ở vòng 1–2"],
                   ["Gán lại giai đoạn 2", "332 ảnh chưa có nhãn review: nháp DeepSeek + Codex, review toàn bộ", "Loại 139 ảnh (gầm xe, nắp xăng), giữ 193; precision hộp DeepSeek 0,23"],
                   ["Kiểm chứng độc lập", "Người thứ hai chỉ xem chỗ model và nhãn bất đồng", "287 chỗ bất đồng → sửa 70 nhãn"],
                   ["Kiểm tra ảnh trống", "Gợi ý của teacher + review 134 ảnh không nhãn", "Loại thêm 30 ảnh, bổ sung 23 nhãn"],
                   ["Bộ cuối", "Gom ảnh trùng, thống nhất quy ước cảnh", "1.081 ảnh, 28 xe, 13 ảnh trùng được gom nhóm"]]),
        ("p", "Chia dữ liệu: để đo trên xe mới, chia theo xe 19 train / 6 val / 3 test, ảnh test không bao giờ được train. Model cuối dùng cả 28 xe và đo bằng 3-fold cross-validation theo ảnh. Kiểm tra thủ công 50 ảnh của bộ nhãn cuối không ghi nhận lỗi nào (đang chờ người chấm xác nhận)."),
        ("h1", "3. Thuật toán"),
        ("img", "kd"),
        ("p", "<b>Knowledge distillation.</b> Với mỗi batch, teacher (đóng băng) và student cùng nhìn một bộ ảnh. Student tối thiểu hoá"),
        ("mono", "L = L_task + 1,0·L_feat + 1,0·L_cls + 1,0·L_loc"),
        ("bullets", [
            "<b>L_task</b>: loss segmentation chuẩn của Ultralytics (hộp, mask, lớp, DFL) so với nhãn đã review.",
            "<b>L_feat</b>: Focal and Global Distillation trên đặc trưng neck P3/P4/P5. Adapter 1×1 chiếu kênh student sang kênh teacher; hộp nhãn chia bản đồ thành vùng linh kiện (trọng số 1,0) và nền (0,5); thêm thành phần attention và ngữ cảnh toàn cục (λ 0,5, nhiệt độ 0,5).",
            "<b>L_cls</b>: KL divergence nhị phân giữa xác suất lớp của teacher và student, làm mềm bằng nhiệt độ 2, điểm neo ở nền có trọng số 0,05.",
            "<b>L_loc</b>: localization distillation trên phân bố DFL của 4 cạnh hộp, nhiệt độ 10.",
            "Epoch 1 chỉ train adapter để phép chiếu chưa học không làm hỏng student. Sau khi train, adapter được gỡ bỏ, còn lại một yolo11n-seg bình thường.",
        ]),
        ("p", "<b>Công thức train (bản cuối).</b> 640 px, batch 16, 100 epoch, learning rate cosine, copy-paste 0,3, mixup 0,1, xoay ±5°, tắt mosaic 10 epoch cuối, lặp ảnh chứa lớp hiếm (repeat-factor sampling t = 0,1, tối đa 4 lần). Teacher thêm weight decay 0,001, scale 0,7, perspective 0,0005. Cả hai lưu checkpoint mỗi 5 epoch; trọng số triển khai là trung bình 5 checkpoint."),
        ("h1", "4. Quá trình train"),
        ("p", "Phần cứng: NVIDIA DGX GB10, 121 GB bộ nhớ hợp nhất, dùng chung với dịch vụ vLLM nên mỗi lúc chỉ chạy một job. Phần mềm: PyTorch 2.14, Ultralytics 8.4.75. Mỗi teacher mất khoảng 90 phút, mỗi student khoảng 60 phút."),
        ("img", "curves"),
        ("caption", "Lần chạy thật kd_n_p5t_s0. Trái: mask mAP50-95 trên val theo epoch của student và teacher. Phải: từng thành phần loss so với epoch 1. mAP của student dao động vài điểm giữa các epoch, nên trung bình checkpoint có ích."),
        ("h1", "5. Benchmark qua các phiên bản"),
        ("img", "versions"),
        ("table", [["Phiên bản", "Thay đổi chính", "Teacher", "Student KD", "Student mAP50", "Một mình", "Đạt/100"],
                   ["v4", "324 ảnh train review, teacher 1024 px", "0,233", "0,209", "0,382", "0,198", "–"],
                   ["v5", "Codex sửa 101 nhãn; val 3 → 6 xe; teacher 60 epoch dừng sớm", "0,219", "0,193", "0,344", "0,165", "–"],
                   ["v6", "640 px, LR cosine, copy-paste, mixup, lặp lớp hiếm, gán lại nhãn giả", "0,259", "0,245", "0,423", "0,218", "47"],
                   ["v7", "+175 ảnh ngoài đã review (chỉ train)", "0,258", "0,249", "0,441", "–", "47"],
                   ["v8", "344 nhãn giả thay bằng nhãn hybrid đã review", "0,290", "0,264", "0,466", "–", "48"],
                   ["kd_n_p5t", "Teacher có regularization + trung bình checkpoint cho cả hai", "0,354", "0,286", "0,491", "–", "58"],
                   ["full", "Cả 28 xe, 1.081 ảnh đã kiểm chứng", "n/a", "n/a", "n/a", "–", "CV"]]),
        ("caption", "mask mAP50-95 trên cùng 125 ảnh test của 3 xe chưa thấy. Student KD là trung bình 2 seed từ v6. v4 và v5 dùng nhãn test trước khi Codex sửa. 'Đạt/100' = số ảnh có precision ≥ 0,5 và recall ≥ 0,5 ở ngưỡng 0,25. v1–v3 dùng nhãn máy chưa review và chưa có tập test cố định nên không so được."),
        ("h2", "Giai đoạn 5: thay đổi từng thứ một trên teacher"),
        ("img", "phase5"),
        ("caption", "Trung bình 5 checkpoint tốt nhất (chọn trên val, chấm trên test) thêm 3–6 điểm; teacher có regularization kèm trung bình checkpoint là teacher tốt nhất."),
        ("h2", "Độ chính xác theo lớp trên xe mới"),
        ("img", "perclass"),
        ("caption", "Student tốt nhất (kd_n_p5t seed 0, đã lấy trung bình) so với teacher tốt nhất. Các lớp chỉ có 3–7 mẫu test (bình nước làm mát, ống két nước, bình nước rửa kính, ECU) có độ bất định lớn."),
        ("h2", "Tốc độ"),
        ("table", [["Model", "Tham số", "Kích thước", "Chỉ mạng", "Toàn pipeline", "FPS"],
                   ["Student, PyTorch FP16 (DGX GB10)", "2,84 tr", "5,7 MB", "2,9 ms", "4,0 ms", "248"],
                   ["Student, ONNX trên CPU (DGX)", "2,84 tr", "11,1 MB", "36,5 ms", "47,4 ms", "21"],
                   ["Teacher, PyTorch (batch 1)", "27,6 tr", "56 MB", "9,2 ms", "–", "–"]]),
        ("caption", "Warm-up 20 lần, đo 200 lần trên ảnh thật. TensorRT phải build trên thiết bị đích và chưa được đo."),
        ("h1", "6. Model cuối"),
        ("p", f"<b>kd_n_full</b> (student) và <b>teacher_full</b> được train trên toàn bộ 1.081 ảnh đã kiểm chứng của 28 xe theo công thức kd_n_p5t, lấy trung bình 5 checkpoint cuối. Trên chính các ảnh dataset (model đã thấy): {f['pass']}/{f['images']} ảnh đạt (94,1%), tìm đúng {f['tp']}/{f['instances']} linh kiện (recall {dec(f['recall'], 'vi')}), {f['fp']} phát hiện thừa ở ngưỡng chung 0,25 (precision {dec(f['precision'], 'vi')}). Các số này thể hiện mức khớp dữ liệu, không phải khả năng tổng quát. 3-fold cross-validation (train trên 2/3, chấm trên 1/3 còn lại) và ngưỡng riêng từng lớp đang chạy tối nay; lần chạy v9 với công thức mới và 3 xe test giữ riêng sẽ cho số tương ứng trên xe mới vào sáng mai."),
        ("p", "Sản phẩm: runs/train_kd/kd_n_full/weights/best.pt (6 MB), scripts/deployment/weights/kd_n_full_640.onnx, runs/segment/teacher_full/weights/best.pt (phương án máy chủ), web UI mặc định dùng student."),
        ("h1", "7. Điểm mạnh"),
        ("bullets", [
            "<b>Distillation luôn có lợi.</b> Student KD hơn cùng student train một mình 1–3 điểm mask mAP50-95 ở mọi phiên bản đã đo, và đạt 80–95% điểm teacher với kích thước bằng 1/10.",
            "<b>Nhanh.</b> 4 ms mỗi ảnh toàn pipeline trên DGX; model 6 MB phù hợp thiết bị biên.",
            "<b>Ổn định với linh kiện dễ phân biệt.</b> Đồng hồ đo, nắp két nước, bình nước rửa kính và bình dầu phanh đạt mask AP50 khoảng 0,8 trên xe mới.",
            "<b>Nhãn sạch, có kiểm toán.</b> Mọi nhãn trong bộ cuối đã qua review chuyên gia, phần lớn đã qua kiểm chứng độc lập, và quy trình đo được độ chính xác thật của nhãn máy thay vì giả định.",
            "<b>Quy trình tái lập được.</b> Dựng dataset bằng script, chuỗi job trên DGX, báo cáo QA và gallery kiểm thử theo ảnh cho từng phiên bản.",
        ]),
        ("h1", "8. Điểm yếu và rủi ro"),
        ("bullets", [
            "<b>Khả năng tổng quát sang xe mới còn hạn chế</b> (student mAP50-95 khoảng 0,29, mAP50 khoảng 0,5). 19 xe train khiến độ đa dạng thực tế nhỏ; khoảng cách giữa ảnh đã học (mAP50 0,94) và xe mới còn lớn.",
            "<b>Tập test nhỏ.</b> 3 xe, 125 ảnh; hai seed của cùng công thức chênh tới 3 điểm, nên chênh lệch dưới khoảng 2 điểm chưa đáng tin.",
            "<b>Lớp yếu:</b> bình nước làm mát (AP50 0,01), hộp cầu chì, ắc quy, ECU, cảm biến MAF. Vẫn còn nhầm lẫn giữa các cặp giống nhau: bình dầu phanh và bình nước làm mát, cổ hút và nắp che động cơ, bugi tháo rời bị nhận là bô-bin.",
            "<b>Phát hiện thừa</b> khi dùng một ngưỡng chung (precision 0,58 trên xe mới, 0,75 trên dataset); ngưỡng riêng từng lớp chưa được áp dụng.",
            "<b>Chưa kiểm chứng trên thiết bị đích</b>; độ trễ TensorRT và độ chính xác sau khi chuyển FP16/INT8 chưa biết.",
        ]),
        ("h1", "9. Đề xuất cải thiện"),
        ("bullets", [
            "<b>Thu thập 40–60 xe mới</b> theo hướng dẫn chụp (khoảng 30 ảnh mỗi xe, cùng kiểu chụp) và mở rộng tập test, val lên ít nhất 6 xe mỗi tập. Đây là mức tăng kỳ vọng lớn nhất cho xe mới.",
            "<b>Áp dụng ngưỡng tin cậy riêng từng lớp</b> từ cross-validation vào ứng dụng và test case.",
            "<b>Khai thác ảnh âm khó:</b> thêm bugi tháo rời, giắc cắm, cảm biến EVAP/MAP và ảnh gầm xe làm ảnh âm; đây là nguồn phát hiện thừa chính.",
            "<b>Vòng active learning:</b> teacher gán nhãn trước cho ảnh mới, chỉ review chỗ bất đồng (hàng đợi kiểm chứng đã làm việc này), rồi train lại.",
            "<b>Tận dụng ảnh chưa gán nhãn</b> qua distillation chỉ trên đặc trưng (trainer đã hỗ trợ, chưa dùng).",
            "<b>Chọn kích thước model theo thiết bị:</b> nếu chạy trên máy chủ, thử student yolo11s hoặc dùng thẳng teacher; nếu chạy trên Jetson, build TensorRT FP16/INT8 trên thiết bị và đo lại độ chính xác.",
            "<b>Đo lường:</b> báo cáo 2–3 seed cho mỗi thay đổi, và giữ cố định 3 xe test để theo dõi trên xe mới.",
            "<b>Dữ liệu ngoài:</b> chỉ ảnh khoang máy có giấy phép cho phép train thương mại (ảnh iFixit là CC BY-NC-SA).",
        ]),
    ]


# ---------------------------------------------------------------- PDF
def build(L):
    lang = L["lang"]
    figs = {"kd": chart_kd(L, lang), "curves": chart_curves(L, lang), "versions": chart_versions(L, lang),
            "phase5": chart_phase5(L, lang), "perclass": chart_perclass(L, lang)}
    ss = {
        "title": ParagraphStyle("t", fontName="Arial-Bold", fontSize=26, leading=31, textColor=C(INK)),
        "sub": ParagraphStyle("s", fontName="Arial", fontSize=15, leading=20, textColor=C(TEACHER)),
        "meta": ParagraphStyle("m", fontName="Arial", fontSize=9.5, leading=13, textColor=C(MUTED)),
        "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=15, leading=19, textColor=C(INK), spaceBefore=12, spaceAfter=6, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11.5, leading=15, textColor=C(TEACHER), spaceBefore=8, spaceAfter=4, keepWithNext=1),
        "p": ParagraphStyle("p", fontName="Arial", fontSize=9.8, leading=14.2, textColor=C(INK), spaceAfter=6, alignment=TA_LEFT),
        "b": ParagraphStyle("b", fontName="Arial", fontSize=9.8, leading=14, textColor=C(INK), leftIndent=12, bulletIndent=2, spaceAfter=3),
        "cap": ParagraphStyle("c", fontName="Arial-Italic", fontSize=8.3, leading=11, textColor=C(MUTED), spaceAfter=8),
        "mono": ParagraphStyle("mo", fontName="Consolas", fontSize=11, leading=15, textColor=C(INK), backColor=C("#EEF2F1"),
                               borderPadding=(6, 8, 6, 8), spaceBefore=4, spaceAfter=10),
        "td": ParagraphStyle("td", fontName="Arial", fontSize=8.3, leading=10.6, textColor=C(INK)),
        "th": ParagraphStyle("th", fontName="Arial-Bold", fontSize=8.3, leading=10.6, textColor=C(INK)),
        "big": ParagraphStyle("bg", fontName="Arial-Bold", fontSize=22, leading=26, textColor=C(STUDENT)),
        "bigt": ParagraphStyle("bgt", fontName="Arial-Bold", fontSize=22, leading=26, textColor=C(TEACHER)),
        "small": ParagraphStyle("sm", fontName="Arial", fontSize=8.5, leading=11, textColor=C(MUTED)),
    }
    W = A4[0] - 36 * mm
    story = [Spacer(1, 40 * mm), Paragraph(L["title"], ss["title"]), Spacer(1, 4), Paragraph(L["subtitle"], ss["sub"]),
             Spacer(1, 10), Paragraph(L["meta"], ss["meta"]), Spacer(1, 16 * mm)]
    f = FINAL
    kv = [("0.302" if lang == "en" else "0,302", "big", "Best student mAP50-95, new vehicles" if lang == "en" else "Student tốt nhất, mAP50-95 xe mới"),
          ("0.354" if lang == "en" else "0,354", "bigt", "Best teacher mAP50-95, new vehicles" if lang == "en" else "Teacher tốt nhất, mAP50-95 xe mới"),
          ("4.0 ms" if lang == "en" else "4,0 ms", "big", "Student end-to-end latency (DGX)" if lang == "en" else "Độ trễ student toàn pipeline (DGX)"),
          (f"{f['pass']}/{f['images']}", "bigt", "Dataset images passed by the final model (seen data)" if lang == "en" else "Ảnh dataset đạt với model cuối (dữ liệu đã học)")]
    cells = [[Paragraph(v, ss[s]) for v, s, _ in kv], [Paragraph(d, ss["small"]) for _, _, d in kv]]
    t = Table(cells, colWidths=[W / 4] * 4)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEABOVE", (0, 0), (-1, 0), 1.2, C(LINE)),
                           ("TOPPADDING", (0, 0), (-1, 0), 8), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story += [t, PageBreak()]

    for kind, payload in body(lang):
        if kind == "h1":
            story.append(Paragraph(payload, ss["h1"]))
        elif kind == "h2":
            story.append(Paragraph(payload, ss["h2"]))
        elif kind == "p":
            story.append(Paragraph(payload, ss["p"]))
        elif kind == "mono":
            story.append(Paragraph(payload, ss["mono"]))
        elif kind == "caption":
            story.append(Paragraph(payload, ss["cap"]))
        elif kind == "bullets":
            for b in payload:
                story.append(Paragraph(b, ss["b"], bulletText="•"))
            story.append(Spacer(1, 4))
        elif kind == "img":
            p = figs[payload]
            import PIL.Image
            w, h = PIL.Image.open(p).size
            story.append(Image(str(p), width=W, height=W * h / w))
            story.append(Spacer(1, 4))
        elif kind == "table":
            rows = [[Paragraph(c, ss["th"] if i == 0 else ss["td"]) for c in r] for i, r in enumerate(payload)]
            n = len(payload[0])
            if n == 3:
                cw = [W * 0.2, W * 0.42, W * 0.38]
            elif n == 7:
                cw = [W * 0.1, W * 0.36] + [W * 0.108] * 5
            else:
                cw = [W * 0.34] + [W * 0.132] * 5
            tb = Table(rows, colWidths=cw, repeatRows=1)
            tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), C("#E3EAE8")), ("LINEBELOW", (0, 0), (-1, -1), 0.4, C(LINE)),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                                    ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
            if n == 7:
                tb.setStyle(TableStyle([("BACKGROUND", (0, 6), (-1, 6), C("#F6E4C2"))]))
            story.append(tb)
            story.append(Spacer(1, 6))

    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFont("Arial", 8)
        canvas.setFillColor(C(MUTED))
        if doc.page > 1:
            canvas.drawString(18 * mm, A4[1] - 12 * mm, L["header"])
            canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{L['page']} {doc.page}")
            canvas.setStrokeColor(C(LINE))
            canvas.line(18 * mm, A4[1] - 14 * mm, A4[0] - 18 * mm, A4[1] - 14 * mm)
        else:
            canvas.setFillColor(C(TEACHER))
            canvas.rect(0, A4[1] - 8 * mm, A4[0], 8 * mm, fill=1, stroke=0)
            canvas.setFillColor(C(STUDENT))
            canvas.rect(0, A4[1] - 10 * mm, A4[0] * 0.3, 2 * mm, fill=1, stroke=0)
        canvas.restoreState()

    out = OUT / L["file"]
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm,
                            bottomMargin=18 * mm, title=f"{L['title']} — {L['subtitle']}", author="Distillation project",
                            subject=L["subtitle"])
    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    return out


if __name__ == "__main__":
    for L in (EN, VI):
        print(build(L))
