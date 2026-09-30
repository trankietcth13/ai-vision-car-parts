### PATENT SPECIFICATION
*(Formatted according to WIPO Patent Cooperation Treaty (PCT) and USPTO Standards)*

--------------------------------------------------------------------------------

#### 1. TITLE OF THE INVENTION
**METHOD AND SYSTEM FOR SIMULATED-TO-REAL SYNTHETIC DATA GENERATION VIA DOMAIN RANDOMIZATION AND MULTI-SCALE VISUAL INSPECTION WITH FUNCTIONAL DECOMPOSITION FOR VEHICLE ENGINE BAY COMPONENTS**

--------------------------------------------------------------------------------

#### 2. FIELD OF THE INVENTION
The present invention relates generally to the fields of computer vision, deep learning, synthetic data generation, and automotive diagnostic inspection systems. More particularly, the invention relates to an integrated method and system for: (a) synthetic data generation using 3D domain randomization (DR) to bridge the reality gap for deep neural network training; (b) automated instance segmentation and multi-scale visual detection on two-dimensional (2D) engine bay images; (c) hierarchical functional grouping of detected components into 14 technical systems; (d) empirical verdict-trained confidence calibration for automated active learning; and (e) deductive inference of concealed engine components (`expected_not_visible`) coupled with Diagnostic Trouble Code (DTC) repair instructions without requiring real-time 3D rendering.

--------------------------------------------------------------------------------

#### 3. BACKGROUND OF THE INVENTION
In the automotive maintenance, aftermarket repair, and vehicle technical inspection industries, comprehensive under-the-hood engine bay inspection requires technicians to inspect dozens of mechanical, electrical, fluid, and pneumatic components. Currently, this process relies heavily on manual human visual inspection, which is labor-intensive, error-prone, and lacks standardized digital reporting.

##### Limitations of Existing Solutions:
1. **OEM Assembly Line Vision Systems:** Factory machine vision systems (e.g., Bosch, BMW assembly lines) operate exclusively within strictly controlled environments with uniform artificial illumination and fixed camera angles, tailored to a single specific engine model. These systems fail when deployed on natural in-the-wild images captured via mobile devices or handheld tablets across diverse commercial vehicle fleets with varying lighting, viewing angles, and grime accumulation.
2. **Drive-Through Vehicle Scanners:** Automated exterior vehicle scanners (e.g., UVeye, Ravin AI, ProovStation) focus exclusively on vehicle exterior body panels, paint defects, glass, tires, and undercarriages. The engine bay is deliberately omitted due to extreme visual complexity, severe component occlusion, and vast physical scale disparity ranging from tiny connectors (several millimeters) to massive acoustic covers and radiators (tens of centimeters).
3. **Conventional Object Detection Frameworks:** Standard object detectors (such as YOLO, Faster R-CNN) trained on generic datasets suffer from severe small-object recall degradation (often below 40% recall) when processing full-view engine bay images at standard resolutions (e.g., 640x640 px). Furthermore, raw confidence scores from detectors or Vision-Language Models (VLMs) are notoriously miscalibrated, rendering automated label auto-acceptance unsafe due to the risk of training-set contamination. Crucially, existing solutions lack functional system-level grouping and cannot infer concealed components required by diagnostic trouble codes.
4. **Real-World Training Data Scarcity and Manual Annotation Bottleneck:** Training deep learning models to recognize hundreds of fine-grained engine components across thousands of Year-Make-Model-Engine (YMME) vehicle variants requires vast amounts of labeled training data. Manual collection and pixel-level annotation of real-world engine bay images is prohibitively expensive, time-consuming, and prone to human labeling errors. Standard synthetic data generated from static CAD models fails to generalize to real-world images due to the "reality gap" (discrepancies in lighting, shadows, surface textures, dirt, grime, and camera distortion).

Therefore, an urgent need exists for a comprehensive solution that combines an automated synthetic data engine with a lightweight, high-precision 2D visual inspection system capable of high-recall small-component detection, calibrated active learning, and deductive inference of concealed components.

--------------------------------------------------------------------------------

#### 4. SUMMARY OF THE INVENTION
The present invention solves the aforementioned deficiencies by providing an integrated system and method featuring five technical pillars:

1. **Sim-to-Real Synthetic Data Engine via 3D Domain Randomization (Training Pipeline):** Generates vast, perfectly labeled training datasets from 3D CAD/mesh models of vehicle engine bays in a simulated 3D environment. Applies extreme domain randomization (DR) by randomly varying camera poses, lighting conditions, background distractor objects (wires, hoses, tools), and surface material properties (metal finishes, plastic aging, dirt, grease, oil stains). Automatically outputs rendered 2D synthetic images with pixel-exact bounding boxes and segmentation masks, enabling deep neural networks to achieve high zero-shot or low-shot generalization on real-world vehicle images without requiring manual data annotation.
2. **3-Tier Taxonomy with Generic Fallback Classes and Promotion Hysteresis:** Organizes engine bay components into 14 functional automotive systems, morphological groups, and 63 fine-grained parts. Includes 6 morphological generic fallback classes (Tier C: `other_reservoir`, `other_sensor_actuator`, `other_hose_line`, `other_cap_plug`, `other_module_box`, `other_pulley_device`) to prevent component omissions. Implements a promotion hysteresis rule requiring $\ge 60$ reviewed instances across $\ge 8$ distinct vehicles to promote a class to independent training status (Tier A), while demoting back to Tier C only if instances fall below 40.
3. **Lightweight Multi-Scale Parallel Inference with Spatial Priors (Inference Pipeline):** Executes parallel inference on a single 2D image captured via a mobile device using a full-resolution global image branch (capturing large components) and an overlapping $2 \times 2$ sub-grid tiling branch (magnifying small sensors, clips, and caps). Coordinates are projected into global space and unified via a Class-Agnostic Non-Maximum Suppression (IoU $\ge 0.85$) algorithm, followed by spatial empirical coordinate filtering ($p_{10}-p_{90}$) and vehicle component count caps.
4. **Verdict-Trained Confidence Calibration for Active Learning:** Trains a Gradient Boosting Decision Tree (GBDT) on expert technician acceptance/rejection verdicts using a multi-dimensional feature vector (VLM confidence, DINOv2 crop margin similarity, spatial prior density, and geometric aspect ratio) cross-validated across vehicle clusters (Vehicle GroupKFold). Achieves an AUROC of 0.844 (vs 0.617 baseline), enabling auto-acceptance of the top 21.8% predictions at 94.1% precision for continuous model improvement.
5. **Functional System Decomposition and Deductive Concealed Component Inference:** Maps detected 2D instances to 14 functional technical systems and correlates vehicle metadata (YMME) against engine technical architecture to deductively output an explicit list of expected but concealed components (`expected_not_visible`) coupled with Diagnostic Trouble Code (DTC) repair instructions—all achieved without running heavy real-time 3D rendering or point-cloud processing on the mobile client device.

--------------------------------------------------------------------------------

#### 5. BRIEF DESCRIPTION OF THE DRAWINGS
* **FIG. 1:** Overall system architecture diagram showing the 3D simulation training pipeline and the multi-scale 2D mobile inspection workflow (100).
* **FIG. 2:** Schematic diagram of the 3D Domain Randomization (DR) synthetic data generation pipeline, illustrating randomization of camera pose, lighting, textures, grime, and distractors (200).
* **FIG. 3:** Schematic diagram of the 3-tier taxonomy structure with generic fallback classes and promotion hysteresis (300).
* **FIG. 4:** Workflow of lightweight multi-scale parallel inference comprising full-image branch, overlapping $2 \times 2$ grid branch, and spatial post-processing (400).
* **FIG. 5:** Diagram of feature extraction and verdict-trained GBDT confidence calibration model with empirical ROC comparison curves (500).
* **FIG. 6:** Schematic diagram of functional system decomposition output, YMME cross-referencing, and deductive concealed component (`expected_not_visible`) inference linked to DTC repair guides (600).

--------------------------------------------------------------------------------

#### 6. DETAILED DESCRIPTION AND EXPERIMENTAL RESULTS

##### A. 3D Domain Randomization (DR) Sim-to-Real Synthetic Data Engine
To eliminate manual data collection and annotation costs, the system constructs a virtual 3D simulation environment populated with 3D CAD models of engine components across various YMME configurations. During synthetic image generation, the rendering engine applies randomized parameters sampled from uniform or Gaussian distributions:
* **Camera Pose & Intrinsic Randomization:** Camera position $(x, y, z)$, orientation angles $(\text{roll}, \text{pitch}, \text{yaw})$, and Field-of-View (FOV) are randomized within plausible physical ranges above the engine bay.
* **Illumination Randomization:** N-point light sources are generated with randomized intensities, color temperatures, positions, and shadow softness to mimic outdoor sunlight, indoor garage fluorescent lighting, and flashlight beams.
* **Material & Texture Randomization:** Surface textures of 3D models are randomly perturbed with varying specularities, metallic parameters, roughness, rust, oil smudges, dust, and grime layers.
* **Distractor Insertion:** Synthetic cables, hoses, diagnostic tools, and technician hands are randomly positioned across the scene as occluding distractors.

Models trained exclusively on DR synthetic data demonstrate robust zero-shot transfer when evaluated on real-world vehicle engine bay images, bridging the sim-to-real domain gap.

##### B. Empirical Validation
Empirical evaluation on 125 independent test vehicle engine bay images confirms marked technical superiority:
* Global full-image baseline: Recall 0.582, Precision 0.632.
* Multi-scale $2 \times 2$ grid + Class-Agnostic NMS + Count Cap: Recall increases to **0.614** (+3.2% overall recall; Lubrication +7%, Cooling +6%, Air Intake +4%, Electrical +3%), effectively suppressing multi-scale fragmentation and cross-class duplicate false positives.
* Confidence calibration on 4,627 expert-reviewed machine boxes achieves an AUROC of **0.844**, compared to 0.617 for raw VLM confidence.
* Runtime inference on standard mobile hardware executes in $<1.2$ seconds per image, as 3D rendering overhead is confined strictly to the offline training stage.

--------------------------------------------------------------------------------

#### 7. PATENT CLAIMS

##### Claim 1 (Independent Method Claim - Visual Inspection & Deductive Inference):
A computer-implemented method for multi-scale visual inspection, functional decomposition, and deductive diagnostic inference of vehicle engine bay components, the method comprising:
(a) receiving a single two-dimensional (2D) digital image depicting a vehicle engine bay captured by a camera device;
(b) executing parallel multi-scale inference on the 2D image via:
    (i) a global full-image branch extracting candidate bounding boxes and segmentation masks for large-scale components, and
    (ii) a sub-grid tiling branch partitioning the 2D image into an overlapping $2 \times 2$ grid to extract candidate bounding boxes and masks for small-scale components;
(c) projecting local bounding box coordinates from the sub-grid tiling branch into a unified global coordinate space;
(d) merging candidate bounding boxes across the branches using a Class-Agnostic Non-Maximum Suppression (NMS) algorithm with an intersection-over-union (IoU) threshold of at least 0.85 to suppress cross-scale duplicate detections;
(e) filtering merged detections by evaluating empirical spatial coordinate prior distributions and predefined vehicle component count caps;
(f) organizing verified detections into a hierarchical tree corresponding to 14 functional automotive engineering systems; and
(g) cross-referencing verified visible 2D component detections with vehicle specification metadata comprising Year, Make, Model, and Engine (YMME) data against an automotive architecture knowledge base to deductively generate an explicit list of expected but concealed components (`expected_not_visible`) without requiring real-time 3D model rendering, wherein said list is coupled with Diagnostic Trouble Code (DTC) repair instructions.

##### Claim 2 (Dependent Claim):
The method of Claim 1, wherein organizing verified detections follows a 3-tier taxonomy comprising functional systems, morphological groups, and fine-grained components, wherein fine-grained components lacking sufficient training instances are grouped into generic morphological fallback classes comprising generic reservoirs, generic sensors and actuators, generic hoses and lines, generic caps and plugs, generic module boxes, and generic pulley devices.

##### Claim 3 (Dependent Claim):
The method of Claim 2, wherein fine-grained components transition between independent training status and generic fallback status via a promotion hysteresis rule requiring at least 60 reviewed instances across at least 8 distinct vehicles for promotion, and demoting only when reviewed instances decrease below 40.

##### Claim 4 (Dependent Claim):
The method of Claim 1, wherein the Class-Agnostic Non-Maximum Suppression intersection-over-union (IoU) threshold is set to 0.85 or higher.

##### Claim 5 (Dependent Claim):
The method of Claim 1, wherein filtering merged detections utilizes empirical 10th percentile ($p_{10}$) to 90th percentile ($p_{90}$) coordinate distributions established from a reference automotive dataset.

##### Claim 6 (Dependent Claim):
The method of Claim 1, further comprising automated confidence calibration for active learning by:
extracting a multi-dimensional feature vector comprising an initial visual confidence score, a foundation model crop margin similarity metric, a spatial prior density, and geometric dimensions;
evaluating the feature vector via a Gradient Boosting Decision Tree (GBDT) trained on historical technician approval verdicts; and
automatically approving candidate bounding boxes having a predicted accuracy probability exceeding a predetermined threshold.

##### Claim 7 (Independent Method Claim - Synthetic Data Generation via 3D Domain Randomization):
A computer-implemented method for synthetic data generation to train a deep learning model for vehicle engine bay visual inspection, the method comprising:
(a) generating a virtual three-dimensional (3D) simulation environment containing 3D digital models of vehicle engine bay components organized according to vehicle technical specifications;
(b) applying domain randomization across rendering parameters of the virtual 3D simulation environment, wherein said domain randomization comprises randomly varying:
    (i) camera position, viewing angle, and Field-of-View (FOV),
    (ii) light source positions, intensities, shadow softness, and color temperatures,
    (iii) surface material properties comprising specularity, roughness, dirt, grime, and oil smudges, and
    (iv) placement of non-target occluding distractor objects comprising wires, hoses, and tools;
(c) rendering a plurality of synthetic 2D images from the randomized virtual 3D simulation environment;
(d) automatically generating ground-truth bounding boxes, instance segmentation masks, and class labels for each rendered synthetic 2D image based on spatial parameters of the 3D digital models; and
(e) training a multi-scale deep neural network object detector using the rendered synthetic 2D images and automatically generated ground-truth labels, wherein the trained neural network achieves zero-shot or low-shot sim-to-real transfer when deployed on real-world vehicle engine bay images.

##### Claim 8 (Dependent Claim):
The method of Claim 7, wherein the 3D digital models are compiled from Computer-Aided Design (CAD) specifications corresponding to specific Year-Make-Model-Engine (YMME) vehicle configurations.

##### Claim 9 (Independent System Claim):
A vehicle engine bay inspection and diagnostic system comprising:
an image capture device configured to acquire 2D digital images of a vehicle engine bay;
a memory storing computer-executable instructions, a YMME automotive architecture knowledge base, and spatial constraint parameters;
one or more processors operatively coupled to the memory and configured to execute the instructions to perform the method of any one of Claims 1 to 6; and
a graphical user interface configured to visually render the hierarchical functional tree of detected visible components and display the deductively inferred list of expected concealed components alongside associated Diagnostic Trouble Code (DTC) repair guides.

##### Claim 10 (Independent Computer-Readable Medium Claim):
A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the processors to perform the steps of the method of any one of Claims 1 to 8.

--------------------------------------------------------------------------------

#### 8. ABSTRACT
A system and method for automated synthetic data generation, multi-scale visual detection, and deductive diagnostic inference of vehicle engine bay components. A synthetic data engine utilizes 3D CAD models and domain randomization (varying camera pose, lighting, textures, grime, and distractors) to render perfectly labeled 2D training images, enabling high sim-to-real neural network generalization without manual data annotation. During inspection, the system receives a single 2D image, executes lightweight parallel inference across a global full-image branch and an overlapping $2 \times 2$ sub-grid tiling branch, and unifies detections using a Class-Agnostic NMS algorithm (IoU $\ge 0.85$) coupled with spatial coordinate priors and count caps. Components are organized into a 14-system functional hierarchy governed by a 3-tier taxonomy with generic fallback classes and promotion hysteresis. A verdict-trained GBDT model calibrates prediction confidence (AUROC 0.844) for safe active learning. Finally, visible 2D detections are cross-referenced with vehicle metadata (YMME) to deductively infer expected concealed components (`expected_not_visible`) and provide DTC repair guidance without real-time 3D rendering overhead.
