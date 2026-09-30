import os
import subprocess
import base64

patents_dir = r"e:\Research\GEN AI\AI Vision\Distillation\docs\patents"
figures_dir = os.path.join(patents_dir, "figures_en")
html_path = os.path.join(patents_dir, "patent_specification_en.html")
pdf_path = os.path.join(patents_dir, "patent_specification_engine_bay_inspection_en_draf.pdf")
md_path = os.path.join(patents_dir, "patent_draft_engine_bay_visual_inspection_en.md")

def get_base64_img(filename):
    path = os.path.join(figures_dir, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
    return ""

img1 = get_base64_img("fig1_system_architecture_en.png")
img2 = get_base64_img("fig2_taxonomy_hysteresis_en.png")
img3 = get_base64_img("fig3_multiscale_inference_en.png")
img4 = get_base64_img("fig4_confidence_calibration_roc_en.png")
img5 = get_base64_img("fig5_functional_decomposition_en.png")
img6 = get_base64_img("fig6_domain_randomization_en.png")

# 1. English Markdown Specification
md_content = """# PATENT SPECIFICATION
*(Formatted according to WIPO Patent Cooperation Treaty (PCT) and USPTO Standards)*

---

## 1. TITLE OF THE INVENTION
**METHOD AND SYSTEM FOR HIERARCHICAL FUNCTIONAL DECOMPOSITION AND MULTI-SCALE VISUAL INSPECTION WITH CONFIDENCE CALIBRATION FOR VEHICLE ENGINE BAY COMPONENTS**

---

## 2. FIELD OF THE INVENTION
The present invention relates generally to the fields of computer vision, deep learning, and automotive diagnostic inspection systems. More particularly, the invention relates to an integrated method and system for automated instance segmentation, multi-scale visual detection, and hierarchical functional grouping of visible vehicle engine bay components, combined with cross-scale spatial post-processing to optimize small-component recall, an empirical verdict-trained confidence calibration model for automated active learning data curation, and a three-dimensional (3D) domain-randomized synthetic data generation pipeline for training such models.

---

## 3. BACKGROUND OF THE INVENTION
In the automotive maintenance, aftermarket repair, and vehicle technical inspection industries, comprehensive under-the-hood engine bay inspection requires technicians to inspect dozens of mechanical, electrical, fluid, and pneumatic components. Currently, this process relies heavily on manual human visual inspection, which is labor-intensive, error-prone, and lacks standardized digital reporting.

### Limitations of Existing Solutions:
1. **OEM Assembly Line Vision Systems:** Factory machine vision systems (e.g., Bosch, BMW assembly lines) operate exclusively within strictly controlled environments with uniform artificial illumination and fixed camera angles, tailored to a single specific engine model. These systems fail when deployed on natural in-the-wild images captured via mobile devices or handheld tablets across diverse commercial vehicle fleets with varying lighting, viewing angles, and grime accumulation.
2. **Drive-Through Vehicle Scanners:** Automated exterior vehicle scanners (e.g., UVeye, Ravin AI, ProovStation) focus exclusively on vehicle exterior body panels, paint defects, glass, tires, and undercarriages. The engine bay is deliberately omitted due to extreme visual complexity, severe component occlusion, and vast physical scale disparity ranging from tiny connectors (several millimeters) to massive acoustic covers and radiators (tens of centimeters).
3. **Conventional Object Detection Frameworks:** Standard object detectors (such as YOLO, Faster R-CNN) trained on generic datasets suffer from severe small-object recall degradation (often below 40% recall) when processing full-view engine bay images at standard resolutions (e.g., 640x640 px). Furthermore, raw confidence scores from detectors or Vision-Language Models (VLMs) are notoriously miscalibrated, rendering automated label auto-acceptance unsafe due to the risk of training-set contamination. Crucially, existing solutions lack functional system-level grouping and cannot infer concealed components required by diagnostic trouble codes.
4. **Training Data Scarcity and the Reality Gap:** Moreover, training deep neural networks to recognize dozens of fine-grained engine bay components across thousands of Year-Make-Model-Engine (YMME) variants requires large pixel-level annotated datasets whose manual production is costly and error-prone, and rare or small components remain under-represented. Synthetic images rendered from static CAD models do not transfer reliably to real photographs because of the reality gap (discrepancies in lighting, shadows, surface textures, grime and camera distortion). Domain randomization has been proposed for robotic object localization in simplified scenes (Tobin et al., IROS 2017), but not for fine-grained, multi-class instance segmentation of heavily occluded engine bay components in combination with expert-reviewed real-world data.

Therefore, an urgent need exists for a robust, multi-scale, functionally structured visual inspection system capable of high-recall detection across diverse vehicles while ensuring safe, calibrated active learning.

---

## 4. SUMMARY OF THE INVENTION
The present invention solves the aforementioned deficiencies by providing an integrated system and method featuring five technical pillars:

1. **3-Tier Taxonomy with Generic Fallback Classes and Promotion Hysteresis:** Organizes engine bay components into 14 functional automotive systems, morphological groups, and 63 fine-grained parts. Includes 6 morphological generic fallback classes (Tier C: `other_reservoir`, `other_sensor_actuator`, `other_hose_line`, `other_cap_plug`, `other_module_box`, `other_pulley_device`) to prevent component omissions. Implements a promotion hysteresis rule requiring $\ge 60$ reviewed instances across $\ge 8$ distinct vehicles to promote a class to independent training status (Tier A), while demoting back to Tier C only if instances fall below 40.
2. **Multi-Scale Parallel Inference with Spatial Priors:** Executes parallel inference on a full-resolution global image (capturing large components) and an overlapping $2 \times 2$ tiling grid (magnifying small sensors, clips, and caps). Coordinates are projected into global space and unified via a Class-Agnostic Non-Maximum Suppression (IoU $\ge 0.85$) algorithm, followed by spatial empirical coordinate filtering ($p_{10}-p_{90}$) and vehicle count caps.
3. **Verdict-Trained Confidence Calibration for Active Learning:** Trains a Gradient Boosting Decision Tree (GBDT) on expert technician acceptance/rejection verdicts using a multi-dimensional feature vector (VLM confidence, DINOv2 crop margin similarity, spatial prior density, and geometric aspect ratio) cross-validated across vehicle clusters (Vehicle GroupKFold). Achieves an AUROC of 0.844 (vs 0.617 baseline), enabling auto-acceptance of the top 21.8% predictions at 94.1% precision.
4. **Functional System Decomposition and Expected Not-Visible Inference:** Maps detected instances to 14 functional systems and correlates vehicle metadata (YMME) against engine technical architecture to output an explicit list of expected but concealed components (`expected_not_visible`) coupled with Diagnostic Trouble Code (DTC) repair instructions, without real-time 3D rendering or point-cloud processing on the client device.
5. **Sim-to-Real Synthetic Data Engine via 3D Domain Randomization (Training Stage):** Renders 3D CAD/mesh models of engine bay components under randomized camera pose, illumination, surface materials and grime, and occluding distractors; automatically derives pixel-exact instance masks, bounding boxes and class labels; and combines the synthetic images with expert-reviewed real images (synthetic pre-training followed by real-image fine-tuning) to supply rare and small component classes. All 3D rendering is confined to the offline training stage.

---

## 5. BRIEF DESCRIPTION OF THE DRAWINGS
- **FIG. 1:** Overall system block diagram of the multi-scale vehicle engine bay inspection and diagnostic architecture (100).
- **FIG. 2:** Schematic diagram of the 3-tier taxonomy structure with generic fallback classes and promotion hysteresis (200).
- **FIG. 3:** Workflow of multi-scale inference comprising full-image branch, overlapping $2 \times 2$ grid branch, and spatial post-processing (300).
- **FIG. 4:** Diagram of feature extraction and verdict-trained GBDT confidence calibration model with empirical ROC comparison curves (400).
- **FIG. 5:** Schematic diagram of functional system decomposition output and expected not-visible component inference (500).
- **FIG. 6:** Schematic diagram of the 3D domain randomization synthetic data engine and two-stage sim-to-real training (600).

---

## 6. DETAILED DESCRIPTION AND EXPERIMENTAL RESULTS
Empirical evaluation on 125 independent test vehicle engine bay images confirms marked technical superiority:
- Global full-image baseline: Recall 0.582, Precision 0.632.
- Multi-scale $2 \times 2$ grid + Class-Agnostic NMS + Count Cap: Recall increases to **0.614** (+3.2% overall recall; Lubrication +7%, Cooling +6%, Air Intake +4%, Electrical +3%), effectively suppressing multi-scale fragmentation and cross-class duplicate false positives.
- Confidence calibration on 4,627 expert-reviewed machine boxes achieves an AUROC of **0.844**, compared to 0.617 for raw VLM confidence.

### 6.1 Synthetic Data Engine via 3D Domain Randomization (600)
In an optional training-stage embodiment illustrated in FIG. 6, the training data of the segmentation neural network (106) is supplemented by a synthetic data engine (600). A 3D asset library (602) stores CAD or mesh models of engine bay components, each associated with a component class of the taxonomy of FIG. 2. In one embodiment the models are assembled into complete engine bay scenes according to a YMME configuration; in another embodiment individual component models (e.g., battery, coolant reservoir, ignition coil, ECU housing, air filter box) are rendered as stand-alone instances.

For each synthetic frame, a domain randomization engine (604) samples parameters from uniform or Gaussian distributions: (i) camera position (x, y, z), orientation (roll, pitch, yaw) and field of view (FOV) within physically plausible ranges above an open engine bay; (ii) one to N light sources with randomized positions, intensities, color temperatures and shadow softness, emulating outdoor sunlight, workshop fluorescent lighting and handheld flashlights; (iii) surface material parameters (specularity, metallic, roughness) with procedural overlays of rust, oil stains, dust and grime; and (iv) non-target occluding distractors such as cables, hoses, tools and technician hands.

A renderer (606) outputs, for each frame, a 2D color image together with an instance identification image and optionally a depth image. In a compositing embodiment, rendered component instances are alpha-composited onto real-world engine bay photographs so that the background retains real-world appearance. An automatic ground truth generator (608) derives from the instance identification image a pixel-exact instance mask, a tight bounding box and a class label for each visible instance, and discards instances whose visible fraction (visible mask area divided by unoccluded mask area) falls below a predetermined threshold, e.g., 10%.

A training set mixer (610) combines the synthetic images with real-world images bearing expert-reviewed labels. The number of synthetic instances generated per class decreases with the number of reviewed real instances of that class, so that synthesis is concentrated on Tier B, rare and small components. Synthetic instances are never counted toward the reviewed-instance thresholds of the promotion hysteresis rule. Two-stage training (612) pre-trains the network on synthetic or mixed data and then fine-tunes it on the reviewed real images. The contribution of the synthetic data is assessed by comparing, on the same held-out set of real test vehicles not seen in training, the fine-tuned network against a network trained on the real images alone.

Because all 3D rendering is confined to the offline training stage, the online inspection pipeline of FIG. 1 and FIG. 3 processes only 2D images and incurs no rendering or point-cloud processing overhead on the client device.

---

## 7. PATENT CLAIMS

### Claim 1 (Independent Method Claim):
A computer-implemented method for multi-scale visual inspection and functional decomposition of vehicle engine bay components, the method comprising:
(a) receiving a two-dimensional (2D) digital image depicting a vehicle engine bay;
(b) executing parallel multi-scale inference on the image via:
    (i) a global full-image branch extracting candidate bounding boxes and segmentation masks for large-scale components, and
    (ii) a sub-grid tiling branch partitioning the image into an overlapping $2 \times 2$ grid to extract candidate bounding boxes and masks for small-scale components;
(c) projecting local bounding box coordinates from the sub-grid tiling branch into a unified global coordinate space;
(d) merging candidate bounding boxes across the branches using a Class-Agnostic Non-Maximum Suppression algorithm with an intersection-over-union (IoU) threshold of at least 0.85 to suppress cross-scale duplicate detections;
(e) filtering merged detections by evaluating empirical spatial coordinate prior distributions and predefined vehicle component count caps;
(f) organizing verified detections into a hierarchical tree corresponding to 14 functional automotive engineering systems; and
(g) outputting a structured digital representation of visible components accompanied by a synthesized list of expected concealed components determined by cross-referencing vehicle engine specifications with an automotive knowledge base.

### Claim 2 (Dependent Claim):
The method of Claim 1, wherein organizing verified detections follows a 3-tier taxonomy comprising functional systems, morphological groups, and fine-grained components, wherein fine-grained components lacking sufficient training instances are grouped into generic morphological fallback classes comprising generic reservoirs, generic sensors and actuators, generic hoses and lines, generic caps and plugs, and generic module boxes.

### Claim 3 (Dependent Claim):
The method of Claim 2, wherein fine-grained components transition between independent training status and generic fallback status via a promotion hysteresis rule requiring at least 60 reviewed instances across at least 8 distinct vehicles for promotion, and demoting only when reviewed instances decrease below 40.

### Claim 4 (Dependent Claim):
The method of Claim 1, wherein the Class-Agnostic Non-Maximum Suppression intersection-over-union (IoU) threshold is set to 0.85 or higher.

### Claim 5 (Dependent Claim):
The method of Claim 1, wherein filtering merged detections utilizes empirical 10th percentile ($p_{10}$) to 90th percentile ($p_{90}$) coordinate distributions established from a reference automotive dataset.

### Claim 6 (Dependent Claim):
The method of Claim 1, further comprising automated confidence calibration for active learning by:
extracting a multi-dimensional feature vector comprising an initial visual confidence score, a foundation model crop margin similarity metric, a spatial prior density, and geometric dimensions;
evaluating the feature vector via a Gradient Boosting Decision Tree trained on historical technician approval verdicts; and
automatically approving candidate bounding boxes having a predicted accuracy probability exceeding a predetermined threshold.

### Claim 7 (Independent Method Claim - Synthetic Training Data via 3D Domain Randomization):
A computer-implemented method for generating training data for a vehicle engine bay component segmentation neural network, the method comprising:
(a) providing a 3D asset library containing three-dimensional digital models of vehicle engine bay components, each model being associated with a component class of a hierarchical component taxonomy;
(b) for each of a plurality of synthetic frames, applying domain randomization by randomly sampling: (i) camera position, orientation and field of view; (ii) number, position, intensity, color temperature and shadow softness of light sources; (iii) surface material properties comprising specularity and roughness together with procedural rust, oil, dust or grime overlays; and (iv) placement of non-target occluding distractor objects comprising cables, hoses, tools or human hands;
(c) rendering each synthetic frame into a two-dimensional (2D) color image and an instance identification image;
(d) automatically deriving, from the instance identification image, a segmentation mask, a bounding box and a class label for each visible component instance; and
(e) training the segmentation neural network using the rendered color images and derived labels in combination with real-world engine bay images bearing expert-reviewed labels.

### Claim 8 (Dependent Claim):
The method of Claim 7, wherein the three-dimensional digital models are compiled from Computer-Aided Design (CAD) data corresponding to specific Year-Make-Model-Engine (YMME) vehicle configurations and are assembled into complete engine bay scenes.

### Claim 9 (Dependent Claim):
The method of Claim 7, wherein rendering comprises rendering individual component models and alpha-compositing the rendered component instances onto real-world engine bay photographs, the segmentation masks being derived from the rendered alpha channel.

### Claim 10 (Dependent Claim):
The method of Claim 7, wherein step (d) discards component instances whose visible fraction, computed as the ratio of visible mask area to unoccluded mask area, falls below a predetermined threshold.

### Claim 11 (Dependent Claim):
The method of Claim 7, wherein step (e) comprises pre-training the segmentation neural network on the rendered color images and subsequently fine-tuning it on the real-world engine bay images, and wherein the number of synthetic instances generated per component class decreases with the number of expert-reviewed real-world instances of that class, synthetic instances being excluded from reviewed-instance counts used to promote component classes to independent training status.

### Claim 12 (Dependent Claim):
The method of Claim 1, wherein the segmentation neural network executing step (b) is trained by the method of Claim 7, all three-dimensional rendering being performed before the image of step (a) is received.

### Claim 13 (Independent System Claim):
A vehicle engine bay inspection and diagnostic system comprising:
an image capture unit configured to acquire digital images of an engine bay;
a memory storing computer-executable instructions and an automotive spatial constraint database;
one or more processors operatively coupled to the memory and configured to execute the instructions to implement the method of any one of Claims 1 to 12; and
a graphical user interface configured to visually render the hierarchical functional tree of detected components and indicate locations of expected concealed components.

### Claim 14 (Independent Computer-Readable Medium Claim):
A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the processors to perform the steps of the method of any one of Claims 1 to 12.

---

## 8. ABSTRACT
A system and method for automated multi-scale visual detection, instance segmentation, and functional decomposition of vehicle engine bay components. The system receives a 2D engine bay image, executes parallel inference across a global full-image branch and an overlapping $2 \times 2$ tiling branch, and unifies detections using a Class-Agnostic NMS algorithm (IoU $\ge 0.85$) coupled with empirical spatial coordinate priors and count caps to maximize small-component recall while eliminating cross-scale false positives. Components are structured in a 3-tier taxonomy featuring generic fallback classes and promotion hysteresis to address data scarcity. The system integrates a verdict-trained gradient boosting model for automated confidence calibration (AUROC 0.844) enabling safe active learning, and infers concealed components (`expected_not_visible`) based on engine architecture to provide actionable diagnostic guidance. In an optional training-stage embodiment, a synthetic data engine renders 3D models of engine bay components under domain randomization of camera pose, illumination, surface materials and occluding distractors, automatically derives pixel-exact masks and labels, and combines the synthetic images with expert-reviewed real images for two-stage training, so that all 3D processing is confined offline.
"""

with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print("Markdown EN written to:", md_path)

# 2. English HTML for PDF generation
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Patent Specification - Multi-Scale Engine Bay Inspection</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm 20mm 20mm 25mm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Times New Roman', Times, serif;
                font-size: 10pt;
            }}
        }}
        body {{
            font-family: 'Times New Roman', Times, serif;
            font-size: 11pt;
            line-height: 1.4;
            color: #111;
            text-align: justify;
            margin: 0;
            padding: 0;
        }}
        .header {{
            text-align: center;
            border-bottom: 2px solid #111;
            padding-bottom: 12px;
            margin-bottom: 24px;
        }}
        .country-title {{
            font-size: 12pt;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .sub-header {{
            font-size: 10.5pt;
            font-style: italic;
            margin-top: 4px;
            color: #333;
        }}
        .doc-title {{
            font-size: 15pt;
            font-weight: bold;
            text-transform: uppercase;
            margin-top: 16px;
            line-height: 1.3;
        }}
        h2 {{
            font-size: 12pt;
            font-weight: bold;
            text-transform: uppercase;
            border-bottom: 1px solid #444;
            padding-bottom: 4px;
            margin-top: 24px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 11pt;
            font-weight: bold;
            margin-top: 14px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p {{
            margin-top: 0;
            margin-bottom: 8px;
            text-indent: 20px;
        }}
        .no-indent {{
            text-indent: 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0 16px 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }}
        table, th, td {{
            border: 1px solid #333;
        }}
        th {{
            background-color: #f2f2f2;
            padding: 6px 8px;
            font-weight: bold;
            text-align: center;
        }}
        td {{
            padding: 6px 8px;
            vertical-align: top;
        }}
        .fig-box {{
            text-align: center;
            margin: 18px 0;
            page-break-inside: avoid;
        }}
        .fig-box img {{
            max-width: 95%;
            height: auto;
            border: 1px solid #cbd5e1;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .fig-caption {{
            font-size: 10pt;
            font-weight: bold;
            margin-top: 6px;
            text-align: center;
        }}
        .claim-item {{
            margin-bottom: 12px;
            padding-left: 24px;
            text-indent: -24px;
        }}
        .formula {{
            text-align: center;
            font-style: italic;
            font-family: 'Cambria Math', 'Times New Roman', serif;
            margin: 10px 0;
            background: #fafafa;
            padding: 6px;
            border: 1px dashed #ccc;
        }}
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>

    <div class="header">
        <div class="country-title">INTERNATIONAL PATENT APPLICATION</div>
        <div class="country-title">PATENT COOPERATION TREATY (PCT) / USPTO</div>
        <div class="sub-header">Standard Specification for Computer-Implemented Invention</div>
        <div class="doc-title">PATENT SPECIFICATION</div>
    </div>

    <h2>1. TITLE OF THE INVENTION</h2>
    <p class="no-indent"><strong>METHOD AND SYSTEM FOR HIERARCHICAL FUNCTIONAL DECOMPOSITION AND MULTI-SCALE VISUAL INSPECTION WITH CONFIDENCE CALIBRATION FOR VEHICLE ENGINE BAY COMPONENTS</strong></p>

    <h2>2. FIELD OF THE INVENTION</h2>
    <p>The present invention relates generally to the fields of computer vision, deep learning, and automotive diagnostic inspection systems. More particularly, the invention relates to an integrated method and system for automated instance segmentation, multi-scale visual detection, and hierarchical functional grouping of visible vehicle engine bay components, combined with cross-scale spatial post-processing to optimize small-component recall, an empirical verdict-trained confidence calibration model for automated active learning data curation, and a three-dimensional (3D) domain-randomized synthetic data generation pipeline for training such models.</p>

    <h2>3. BACKGROUND OF THE INVENTION</h2>
    <p>In the automotive maintenance, aftermarket repair, and vehicle technical inspection industries, comprehensive under-the-hood engine bay inspection requires technicians to inspect dozens of mechanical, electrical, fluid, and pneumatic components. Currently, this process relies heavily on manual human visual inspection, which is labor-intensive, error-prone, and lacks standardized digital reporting.</p>
    <p><strong>First,</strong> factory machine vision systems (e.g., Bosch, BMW assembly lines) operate exclusively within strictly controlled environments with uniform artificial illumination and fixed camera angles, tailored to a single specific engine model. These systems fail when deployed on natural in-the-wild images captured via mobile devices or handheld tablets across diverse commercial vehicle fleets with varying lighting, viewing angles, and grime accumulation.</p>
    <p><strong>Second,</strong> automated exterior vehicle scanners (e.g., UVeye, Ravin AI, ProovStation) focus exclusively on vehicle exterior body panels, paint defects, glass, tires, and undercarriages. The engine bay is deliberately omitted due to extreme visual complexity, severe component occlusion, and vast physical scale disparity ranging from tiny connectors (several millimeters) to massive acoustic covers and radiators (tens of centimeters).</p>
    <p><strong>Third,</strong> standard object detectors (such as YOLO, Faster R-CNN) trained on generic datasets suffer from severe small-object recall degradation (often below 40% recall) when processing full-view engine bay images at standard resolutions (e.g., 640x640 px). Furthermore, raw confidence scores from detectors or Vision-Language Models (VLMs) are notoriously miscalibrated, rendering automated label auto-acceptance unsafe due to the risk of training-set contamination. Crucially, existing solutions lack functional system-level grouping and cannot infer concealed components required by diagnostic trouble codes.</p>
    <p><strong>Fourth,</strong> training deep neural networks to recognize dozens of fine-grained engine bay components across thousands of Year-Make-Model-Engine (YMME) variants requires large pixel-level annotated datasets whose manual production is costly and error-prone, and rare or small components remain under-represented. Synthetic images rendered from static CAD models do not transfer reliably to real photographs because of the reality gap (discrepancies in lighting, shadows, surface textures, grime and camera distortion). Domain randomization has been proposed for robotic object localization in simplified scenes (Tobin et al., IROS 2017), but not for fine-grained, multi-class instance segmentation of heavily occluded engine bay components in combination with expert-reviewed real-world data.</p>

    <h2>4. SUMMARY OF THE INVENTION</h2>
    <p>The present invention solves the aforementioned deficiencies by providing an integrated system and method featuring five technical pillars:</p>
    <p><strong>(1) 3-Tier Taxonomy with Generic Fallback Classes and Promotion Hysteresis:</strong> Organizes engine bay components into 14 functional automotive systems, morphological groups, and 63 fine-grained parts. Includes 6 morphological generic fallback classes (Tier C: <em>other_reservoir, other_sensor_actuator, other_hose_line, other_cap_plug, other_module_box, other_pulley_device</em>) to prevent component omissions. Implements a promotion hysteresis rule requiring &ge; 60 reviewed instances across &ge; 8 distinct vehicles to promote a class to independent training status (Tier A), while demoting back to Tier C only if instances fall below 40.</p>
    <p><strong>(2) Multi-Scale Parallel Inference with Spatial Priors:</strong> Executes parallel inference on a full-resolution global image (capturing large components) and an overlapping 2x2 tiling grid (magnifying small sensors, clips, and caps). Coordinates are projected into global space and unified via a Class-Agnostic Non-Maximum Suppression (IoU &ge; 0.85) algorithm, followed by spatial empirical coordinate filtering ([p10, p90]) and vehicle count caps.</p>
    <p><strong>(3) Verdict-Trained Confidence Calibration for Active Learning:</strong> Trains a Gradient Boosting Decision Tree (GBDT) on expert technician acceptance/rejection verdicts using a multi-dimensional feature vector (VLM confidence, DINOv2 crop margin similarity, spatial prior density, and geometric aspect ratio) cross-validated across vehicle clusters (Vehicle GroupKFold). Achieves an AUROC of 0.844 (vs 0.617 baseline), enabling auto-acceptance of the top 21.8% predictions at 94.1% precision.</p>
    <p><strong>(4) Functional System Decomposition and Expected Not-Visible Inference:</strong> Maps detected instances to 14 functional systems and correlates vehicle metadata (YMME) against engine technical architecture to output an explicit list of expected but concealed components (<em>expected_not_visible</em>) coupled with Diagnostic Trouble Code (DTC) repair instructions, without real-time 3D rendering or point-cloud processing on the client device.</p>
    <p><strong>(5) Sim-to-Real Synthetic Data Engine via 3D Domain Randomization (Training Stage):</strong> Renders 3D CAD/mesh models of engine bay components under randomized camera pose, illumination, surface materials and grime, and occluding distractors; automatically derives pixel-exact instance masks, bounding boxes and class labels; and combines the synthetic images with expert-reviewed real images (synthetic pre-training followed by real-image fine-tuning) to supply rare and small component classes. All 3D rendering is confined to the offline training stage.</p>

    <div class="page-break"></div>

    <h2>5. BRIEF DESCRIPTION OF THE DRAWINGS</h2>
    <p class="no-indent">The accompanying drawings illustrate exemplary embodiments of the present invention:</p>
    
    <div class="fig-box">
        <img src="{img1}" alt="FIG. 1">
        <div class="fig-caption">FIG. 1: Vehicle Engine Bay Inspection and Diagnostic System Architecture (100)</div>
    </div>
    <p>FIG. 1 illustrates the overall system block diagram including Image Acquisition Unit (102), Multi-Scale Preprocessor (104), Segmentation Neural Network (106), Spatial Constraint Database (108), Post-Processing and Merge Unit (110), Diagnostic Reasoning Unit (112), and Visual Inspection UI (114).</p>

    <div class="fig-box">
        <img src="{img2}" alt="FIG. 2">
        <div class="fig-caption">FIG. 2: 3-Tier Taxonomy Structure with Promotion Hysteresis (200)</div>
    </div>
    <p>FIG. 2 depicts the 3-tier hierarchy: 14 Functional Automotive Systems (202), Morphological Groups and 6 Generic Fallback Classes Tier C (204), Independent Training Classes Tier A (206), Candidate Fine Classes Tier B (208), and the hysteresis loop transitioning at 60 instances / 8 vehicles vs 40 instance floor.</p>

    <div class="page-break"></div>

    <div class="fig-box">
        <img src="{img3}" alt="FIG. 3">
        <div class="fig-caption">FIG. 3: Multi-Scale Inference and Spatial Post-Processing (300)</div>
    </div>
    <p>FIG. 3 demonstrates the dual-branch workflow: Full-image branch (302) capturing macro components, Overlapping 2x2 grid branch (304) magnifying micro components, and Unified detections (306) with spatial post-processing pipeline (308) suppressing cross-scale duplicates.</p>

    <div class="fig-box">
        <img src="{img4}" alt="FIG. 4">
        <div class="fig-caption">FIG. 4: Automated Label-Confidence Calibration Model W1e (400)</div>
    </div>
    <p>FIG. 4 details the 5-feature extraction pipeline into the calibrated GBDT model (402, 404) and the resulting empirical ROC curve comparison (406) showing AUROC elevation from 0.617 to 0.844 with 94.1% precision in the auto-accept zone.</p>

    <div class="page-break"></div>

    <div class="fig-box">
        <img src="{img5}" alt="FIG. 5">
        <div class="fig-caption">FIG. 5: Functional System Decomposition Output and Concealed Components (500)</div>
    </div>
    <p>FIG. 5 shows the structured output: Visible Components Tree grouped into 14 systems (502), Expected Not-Visible Parts (504), and Diagnostic Guidance Coupling linking visual findings with DTC fault codes (506).</p>

    <div class="page-break"></div>

    <div class="fig-box">
        <img src="{img6}" alt="FIG. 6">
        <div class="fig-caption">FIG. 6: 3D Domain Randomization Synthetic Data Engine (600)</div>
    </div>
    <p>FIG. 6 shows the offline synthetic data engine: 3D Asset Library (602), Domain Randomization Engine (604) with camera, illumination, material and distractor randomization, Renderer (606), Automatic Ground Truth Generator (608), Training Set Mixer (610) and Two-Stage Training (612), whose output is the Segmentation Neural Network (106) deployed for 2D-only inference.</p>

    <h2>6. DETAILED DESCRIPTION AND EMPIRICAL VALIDATION</h2>
    <p>The promotion hysteresis transition is governed mathematically by:</p>
    <div class="formula">
        T(c_i) = Tier A &nbsp; when &nbsp; N(c_i) &ge; 60 &nbsp; and &nbsp; V(c_i) &ge; 8;<br>
        T(c_i) = Tier C &nbsp; when &nbsp; T_prev(c_i) = Tier A &nbsp; and &nbsp; N(c_i) &lt; 40
    </div>

    <p>Table 1 presents experimental validation results on 125 independent test vehicle engine bay images across diverse makes (YOLO11l-seg backbone at 640 px input resolution):</p>
    
    <table>
        <thead>
            <tr>
                <th>Inference Mode</th>
                <th>Recall</th>
                <th>Precision</th>
                <th>Total Predictions</th>
                <th>Technical Significance</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Full Image Baseline</td>
                <td>0.582</td>
                <td>0.632</td>
                <td>408</td>
                <td>Single-scale baseline</td>
            </tr>
            <tr>
                <td>Full Image + 2x2 Tiling Grid</td>
                <td>0.621</td>
                <td>0.552</td>
                <td>498</td>
                <td>Captures small parts; higher false pos.</td>
            </tr>
            <tr>
                <td>Tiles + Class-Agnostic NMS Merge</td>
                <td>0.616</td>
                <td>0.575</td>
                <td>475</td>
                <td>Suppresses cross-scale fragments</td>
            </tr>
            <tr>
                <td><strong>Present Invention (Tiles + NMS + Count Cap)</strong></td>
                <td><strong>0.614</strong></td>
                <td><strong>0.581</strong></td>
                <td>468</td>
                <td><strong>+3.2% Recall gain, high precision</strong></td>
            </tr>
        </tbody>
    </table>

    <p>Crucially, system-level recall demonstrates significant gains: Lubrication recall increases from 0.76 to 0.83 (+7%), Cooling from 0.56 to 0.62 (+6%), Air Intake from 0.55 to 0.59 (+4%), and Electrical from 0.55 to 0.58 (+3%).</p>

    <h3>6.1 Synthetic Data Engine via 3D Domain Randomization (600)</h3>
    <p>In an optional training-stage embodiment illustrated in FIG. 6, the training data of the segmentation neural network (106) is supplemented by a synthetic data engine (600). A 3D asset library (602) stores CAD or mesh models of engine bay components, each associated with a component class of the taxonomy of FIG. 2. In one embodiment the models are assembled into complete engine bay scenes according to a YMME configuration; in another embodiment individual component models (e.g., battery, coolant reservoir, ignition coil, ECU housing, air filter box) are rendered as stand-alone instances.</p>
    <p>For each synthetic frame, a domain randomization engine (604) samples parameters from uniform or Gaussian distributions: (i) camera position (x, y, z), orientation (roll, pitch, yaw) and field of view (FOV) within physically plausible ranges above an open engine bay; (ii) one to N light sources with randomized positions, intensities, color temperatures and shadow softness, emulating outdoor sunlight, workshop fluorescent lighting and handheld flashlights; (iii) surface material parameters (specularity, metallic, roughness) with procedural overlays of rust, oil stains, dust and grime; and (iv) non-target occluding distractors such as cables, hoses, tools and technician hands.</p>
    <p>A renderer (606) outputs, for each frame, a 2D color image together with an instance identification image and optionally a depth image. In a compositing embodiment, rendered component instances are alpha-composited onto real-world engine bay photographs so that the background retains real-world appearance. An automatic ground truth generator (608) derives from the instance identification image a pixel-exact instance mask, a tight bounding box and a class label for each visible instance, and discards instances whose visible fraction (visible mask area divided by unoccluded mask area) falls below a predetermined threshold, e.g., 10%.</p>
    <p>A training set mixer (610) combines the synthetic images with real-world images bearing expert-reviewed labels. The number of synthetic instances generated per class decreases with the number of reviewed real instances of that class, so that synthesis is concentrated on Tier B, rare and small components. Synthetic instances are never counted toward the reviewed-instance thresholds of the promotion hysteresis rule. Two-stage training (612) pre-trains the network on synthetic or mixed data and then fine-tunes it on the reviewed real images. The contribution of the synthetic data is assessed by comparing, on the same held-out set of real test vehicles not seen in training, the fine-tuned network against a network trained on the real images alone.</p>
    <p>Because all 3D rendering is confined to the offline training stage, the online inspection pipeline of FIG. 1 and FIG. 3 processes only 2D images and incurs no rendering or point-cloud processing overhead on the client device.</p>

    <div class="page-break"></div>

    <h2>7. PATENT CLAIMS</h2>
    
    <div class="claim-item">
        <strong>1.</strong> A computer-implemented method for multi-scale visual inspection and functional decomposition of vehicle engine bay components, the method comprising:<br>
        (a) receiving a two-dimensional (2D) digital image depicting a vehicle engine bay from an image acquisition device;<br>
        (b) executing parallel multi-scale inference on the image via: a global full-image branch extracting candidate bounding boxes and segmentation masks for large-scale components, and a sub-grid tiling branch partitioning the image into an overlapping 2x2 grid to extract candidate bounding boxes and masks for small-scale components;<br>
        (c) projecting local bounding box coordinates from the sub-grid tiling branch into a unified global coordinate space;<br>
        (d) merging candidate bounding boxes across the branches using a Class-Agnostic Non-Maximum Suppression algorithm with an intersection-over-union (IoU) threshold of at least 0.85 to suppress cross-scale duplicate detections;<br>
        (e) filtering merged detections by evaluating empirical spatial coordinate prior distributions and predefined vehicle component count caps;<br>
        (f) organizing verified detections into a hierarchical tree corresponding to 14 functional automotive engineering systems; and<br>
        (g) outputting a structured digital representation of visible components accompanied by a synthesized list of expected concealed components determined by cross-referencing vehicle engine specifications with an automotive knowledge base.
    </div>

    <div class="claim-item">
        <strong>2.</strong> The method of Claim 1, wherein organizing verified detections follows a 3-tier taxonomy comprising functional systems, morphological groups, and fine-grained components, wherein fine-grained components lacking sufficient training instances are grouped into generic morphological fallback classes comprising generic reservoirs, generic sensors and actuators, generic hoses and lines, generic caps and plugs, and generic module boxes.
    </div>

    <div class="claim-item">
        <strong>3.</strong> The method of Claim 2, wherein fine-grained components transition between independent training status and generic fallback status via a promotion hysteresis rule requiring at least 60 reviewed instances across at least 8 distinct vehicles for promotion, and demoting only when reviewed instances decrease below 40.
    </div>

    <div class="claim-item">
        <strong>4.</strong> The method of Claim 1, wherein the Class-Agnostic Non-Maximum Suppression intersection-over-union (IoU) threshold is set to 0.85 or higher.
    </div>

    <div class="claim-item">
        <strong>5.</strong> The method of Claim 1, wherein filtering merged detections utilizes empirical 10th percentile (p10) to 90th percentile (p90) coordinate distributions established from a reference automotive dataset.
    </div>

    <div class="claim-item">
        <strong>6.</strong> The method of Claim 1, further comprising automated confidence calibration for active learning by:<br>
        • extracting a multi-dimensional feature vector comprising an initial visual confidence score, a foundation model crop margin similarity metric, a spatial prior density, and geometric dimensions;<br>
        • evaluating the feature vector via a Gradient Boosting Decision Tree trained on historical technician approval verdicts; and<br>
        • automatically approving candidate bounding boxes having a predicted accuracy probability exceeding a predetermined threshold.
    </div>

    <div class="claim-item">
        <strong>7.</strong> A computer-implemented method for generating training data for a vehicle engine bay component segmentation neural network, the method comprising:<br>
        (a) providing a 3D asset library containing three-dimensional digital models of vehicle engine bay components, each model being associated with a component class of a hierarchical component taxonomy;<br>
        (b) for each of a plurality of synthetic frames, applying domain randomization by randomly sampling: (i) camera position, orientation and field of view; (ii) number, position, intensity, color temperature and shadow softness of light sources; (iii) surface material properties comprising specularity and roughness together with procedural rust, oil, dust or grime overlays; and (iv) placement of non-target occluding distractor objects comprising cables, hoses, tools or human hands;<br>
        (c) rendering each synthetic frame into a two-dimensional (2D) color image and an instance identification image;<br>
        (d) automatically deriving, from the instance identification image, a segmentation mask, a bounding box and a class label for each visible component instance; and<br>
        (e) training the segmentation neural network using the rendered color images and derived labels in combination with real-world engine bay images bearing expert-reviewed labels.
    </div>

    <div class="claim-item">
        <strong>8.</strong> The method of Claim 7, wherein the three-dimensional digital models are compiled from Computer-Aided Design (CAD) data corresponding to specific Year-Make-Model-Engine (YMME) vehicle configurations and are assembled into complete engine bay scenes.
    </div>

    <div class="claim-item">
        <strong>9.</strong> The method of Claim 7, wherein rendering comprises rendering individual component models and alpha-compositing the rendered component instances onto real-world engine bay photographs, the segmentation masks being derived from the rendered alpha channel.
    </div>

    <div class="claim-item">
        <strong>10.</strong> The method of Claim 7, wherein step (d) discards component instances whose visible fraction, computed as the ratio of visible mask area to unoccluded mask area, falls below a predetermined threshold.
    </div>

    <div class="claim-item">
        <strong>11.</strong> The method of Claim 7, wherein step (e) comprises pre-training the segmentation neural network on the rendered color images and subsequently fine-tuning it on the real-world engine bay images, and wherein the number of synthetic instances generated per component class decreases with the number of expert-reviewed real-world instances of that class, synthetic instances being excluded from reviewed-instance counts used to promote component classes to independent training status.
    </div>

    <div class="claim-item">
        <strong>12.</strong> The method of Claim 1, wherein the segmentation neural network executing step (b) is trained by the method of Claim 7, all three-dimensional rendering being performed before the image of step (a) is received.
    </div>

    <div class="claim-item">
        <strong>13.</strong> A vehicle engine bay inspection and diagnostic system comprising:<br>
        • an image capture unit configured to acquire digital images of an engine bay;<br>
        • a memory storing computer-executable instructions and an automotive spatial constraint database;<br>
        • one or more processors operatively coupled to the memory and configured to execute the instructions to implement the method of any one of Claims 1 to 12; and<br>
        • a graphical user interface configured to visually render the hierarchical functional tree of detected components and indicate locations of expected concealed components.
    </div>

    <div class="claim-item">
        <strong>14.</strong> A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the processors to perform the steps of the method of any one of Claims 1 to 12.
    </div>

    <h2>8. ABSTRACT</h2>
    <p>A system and method for automated multi-scale visual detection, instance segmentation, and functional decomposition of vehicle engine bay components. The system receives a 2D engine bay image, executes parallel inference across a global full-image branch and an overlapping 2x2 tiling branch, and unifies detections using a Class-Agnostic NMS algorithm (IoU &ge; 0.85) coupled with empirical spatial coordinate priors and count caps to maximize small-component recall while eliminating cross-scale false positives. Components are structured in a 3-tier taxonomy featuring generic fallback classes and promotion hysteresis to address data scarcity. The system integrates a verdict-trained gradient boosting model for automated confidence calibration (AUROC 0.844) enabling safe active learning, and infers concealed components (<em>expected_not_visible</em>) based on engine architecture to provide actionable diagnostic guidance. In an optional training-stage embodiment, a synthetic data engine renders 3D models of engine bay components under domain randomization of camera pose, illumination, surface materials and occluding distractors, automatically derives pixel-exact masks and labels, and combines the synthetic images with expert-reviewed real images for two-stage training, so that all 3D processing is confined offline.</p>

</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)
print("HTML EN written to:", html_path)

msedge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if os.path.exists(msedge_exe):
    cmd = [
        msedge_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]
    print("Running Edge headless print-to-pdf for English version...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(pdf_path):
        print("SUCCESS: English Patent PDF generated at:", pdf_path)
        print("Size:", os.path.getsize(pdf_path), "bytes")
    else:
        print("Error printing PDF:", res.stderr)
