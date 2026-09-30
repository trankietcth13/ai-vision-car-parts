# PATENT SPECIFICATION - REVISED WORKING DRAFT

Revision date: 30 September 2026. This specification distinguishes reported experimental results from illustrative and optional embodiments. Proposed embodiments are not assertions of completed implementation or measured performance.

## 1. TITLE OF THE INVENTION

METHOD AND SYSTEM FOR HIERARCHICAL FUNCTIONAL DECOMPOSITION AND MULTI-SCALE VISUAL INSPECTION WITH VERDICT-TRAINED CONFIDENCE SCORING FOR VEHICLE ENGINE BAY COMPONENTS

## 2. FIELD OF THE INVENTION

The disclosure relates to computer-implemented visual inspection of vehicle engine bays, including instance segmentation, multi-scale detection, hierarchical component representation, review-driven training-data curation, and optional offline synthetic-data generation. An optional probability-calibration stage converts a verdict-trained score into an estimate of a specifically defined label-quality event.

## 3. BACKGROUND OF THE INVENTION

Engine bay photographs contain components of widely differing sizes, partially occluded structures, variable illumination, surface contamination, and vehicle-specific layouts. A full-image detector may lose image detail relevant to small components when resizing an entire photograph to its network input resolution. Tile inference increases local detail but can introduce duplicate detections, truncated objects, and conflicting class assignments.

Factory inspection arrangements may use controlled acquisition conditions. Applying an inspection model to handheld photographs across different vehicles introduces additional variation. Inspection systems addressing other vehicle surfaces do not by themselves resolve component-level engine bay interpretation. These observations describe technical challenges and do not assert that named commercial systems lack particular capabilities.

Detector or vision-language-model confidence is not necessarily an estimate that a proposed training label satisfies class, geometry, and instance-uniqueness requirements. Furthermore, a component that is not observed in a photograph may be outside the field of view, obscured, unrecognized, absent, or inapplicable to the vehicle configuration. Treating non-detection as proof of physical absence or failure can therefore produce incorrect diagnostic statements.

Domain randomization, object detection, hierarchical classification, and confidence-based selection are known technique families. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," IROS 2017, DOI 10.1109/IROS.2017.8202133, describes domain randomization. The present disclosure addresses a coordinated engine bay pipeline in which globally mapped multi-scale instances, review-supported class states, spatial evidence, and vehicle-specific expectations are retained in a structured representation. No assertion of novelty over all prior art is made by this background.

## 4. SUMMARY OF THE INVENTION

In an inspection embodiment, a processor applies an instance-segmentation network to a global image and overlapping local image regions, projects boxes and masks into a common coordinate system, suppresses duplicate candidates, and evaluates component-count and spatial evidence. Retained instances are mapped into a hierarchy of functional systems, morphological groups, and component identities. Vehicle-configuration records optionally identify applicable components that were not observed, together with an explicit uncertainty state.

Taxonomic hierarchy levels are distinct from training states. A fine component can retain its identity in the annotation record while being trained under a generic morphological fallback. Promotion to an independent training class requires reviewed support from multiple vehicles. Separate promotion and retention thresholds reduce state oscillation when labels are corrected or withdrawn.

An optional curation embodiment trains a verdict model from reviewed candidate instances using appearance, geometry, position, overlap, source, and class features. Class correctness and geometry acceptance are tracked separately. A candidate enters a training set automatically only when the configured class-score and geometry/uniqueness gates are satisfied; otherwise it is retained for review or rejected from training. Threshold selection and any probability calibration use data separated by vehicle from final evaluation data.

An optional training embodiment renders component scenes with randomized camera, lighting, material, contamination, and occlusion parameters. Synthetic labels share the inspection taxonomy, and synthetic counts do not contribute to reviewed-real-instance promotion thresholds. Rendering is offline; deployed inspection operates on two-dimensional images. Specific grid dimensions, class counts, thresholds, model families, and input resolutions below are examples rather than universal requirements.

## 5. BRIEF DESCRIPTION OF THE DRAWINGS

FIG. 1 depicts image acquisition (102), multi-scale preprocessing (104), segmentation (106), spatial constraints (108), duplicate and count processing (110), expectation reasoning (112), and output display (114).

FIG. 2 separates hierarchy levels from training states and depicts candidate-to-independent promotion and independent-to-fallback demotion.

FIG. 3 depicts image and tile transforms, global box/mask projection, border handling, duplicate suppression, and count processing.

FIG. 4 depicts verdict feature extraction, model fitting, optional probability calibration, and class/geometry/uniqueness gates. It is a schematic, not a reconstructed empirical ROC curve.

FIG. 5 depicts observed instances, expected-but-not-observed entries, uncertainty, and diagnostic inspection guidance.

FIG. 6 depicts asset library (602), randomization (604), renderer (606), ground-truth generation (608), data mixing (610), and training (612).

FIG. 7 depicts the illustrative coordinate conversion and duplicate-removal example of Section 6.8.

## 6. DETAILED DESCRIPTION

### 6.1 Architecture, data records, and execution

An image acquisition unit supplies an RGB image I with width W and height H. The image identifier, acquisition timestamp, orientation, vehicle identifier when available, and optional Year-Make-Model-Engine (YMME) metadata are stored with a processing record. A processor may execute the inference branches sequentially, concurrently, or in batches on a server or local computing device. Parallel branches denote distinct processing paths and do not require simultaneous hardware execution.

A candidate record contains an instance identifier, claimed class, detector score, branch identifier, original-image box, mask polygon or raster mask, truncation flag, and model version. A constraint record contains a class identifier, applicable vehicle configuration if known, spatial statistics, reviewed-data support, and count limit. A taxonomy record contains a fine identity, functional system, morphological group, fallback identity, declared training state, and provenance. Unrecognized components can remain generic without an unsupported fine identity or functional assignment.

### 6.2 Global and local inference; coordinate and mask projection

The global branch processes the complete image after an invertible resize and optional letterbox transformation. Local branches crop overlapping regions. In an implementation-compatible example, a grid has gx columns and gy rows, overlap fraction o = 0.25, gx = gy = 2, and network input size 640. Tile widths and heights are tw = W / [gx - (gx - 1)o] and th = H / [gy - (gy - 1)o]. Tile origins are x0 = i tw(1-o), y0 = j th(1-o), for the respective column and row indices. Integer crop bounds are recorded exactly. Other grid dimensions and overlap values are optional embodiments selected for image size and compute budget.

If a network-space point is (u,v), the crop was scaled by s, and letterbox offsets are (px,py), its original-image coordinates are x = x0 + (u-px)/s and y = y0 + (v-py)/s. When the network library already returns coordinates in crop pixels, the inverse resize step is omitted. Coordinates are clipped to the image boundaries, and normalized coordinates are x/W and y/H.

Every box corner and mask polygon vertex undergoes the same transform. A raster mask is inverse-warped to the crop dimensions and placed at the recorded crop origin on the global canvas; binary masks use nearest-neighbor resampling. Mask confidence maps may instead be interpolated before thresholding. Keeping the same transform for boxes and masks prevents a retained box from being associated with a displaced tile mask.

In the implemented border-handling example, a box within three crop pixels of an internal tile boundary is removed as potentially truncated. Outer image boundaries do not trigger this internal-boundary rule. Overlapping neighboring tiles and the global branch may supply a more complete instance, but recovery is not guaranteed. An optional alternative retains truncated candidates with a flag and prefers a complete candidate when equivalent detections exist.

### 6.3 Duplicate suppression and masks

Candidates are first screened by a detector threshold, exemplarily 0.25. In the reported implementation, class-wise NMS at box IoU 0.5 is followed by class-agnostic NMS at box IoU 0.85. IoU is intersection area divided by union area. Candidates are ranked by detector score; the retained record includes the mask of the winning candidate. Scores are not described as calibrated probabilities solely because they determine ranking.

The class-agnostic stage removes strongly overlapping candidates even when their claimed classes differ. This can suppress a genuinely distinct overlapping component; it is therefore a tunable duplicate heuristic. In an optional embodiment, conflicting parent/subcomponent pairs remain separate unless mask overlap and class-compatibility evidence also support a duplicate decision. Mask union or weighted fusion, if used, is limited to compatible instances and is recorded as an alternative rather than as the reported NMS implementation.

### 6.4 Spatial evidence and component-count constraints

Reviewed training labels supply normalized box centers cx = (x1+x2)/(2W), cy = (y1+y2)/(2H), and normalized size sqrt[(x2-x1)(y2-y1)/(WH)]. A 10 by 10 histogram with additive smoothing alpha = 1 provides a density relative to a uniform distribution: density[j,k] = 100(n[j,k]+alpha) / sum(n+alpha). The p10-p90 intervals are marginal center-coordinate intervals; the size interval is separately calculated. A class with fewer than five supporting instances returns a neutral density of 1 and no percentile interval.

Unregistered handheld photographs have different viewpoints, making global spatial priors weak. The implemented embodiment exposes density as evidence rather than applying a hard p10-p90 exclusion. An optional configuration-conditioned embodiment first registers image orientation and landmarks or selects a matched viewpoint/YMME prior; only then may a percentile condition route a candidate to review or exclusion. Unmatched configurations, close-ups, and absent landmarks use neutral or soft priors. A candidate outside an interval is not thereby established to be false.

For the reported count-cap implementation, count_max is the largest per-image count observed in supporting training images of a class, where at least five such images exist. The processing limit is count_max + 1. Detections are sorted by score, and candidates exceeding that limit are removed from the displayed retained set, while audit records may preserve the rejected candidates. If no supported count exists, no cap is imposed. Optional manufacturer configuration counts replace empirical limits when applicable. Neither type of cap is treated as evidence that a physically plausible extra component cannot exist.

### 6.5 Hierarchy, fallback mapping, and promotion hysteresis

The hierarchy comprises functional system, morphological group, and fine component. Training states A, B, and C are a separate axis: A denotes a component trained independently; B denotes a candidate fine component retaining its annotation identity but training under a fallback, or being ignored if no fallback is available; C denotes a generic fallback training class. Demotion of an A component therefore returns its fine record to candidate state B and maps its training labels to a C fallback. It does not erase the fine identity.

Table 1 records the current repository taxonomy rather than the original draft's fixed system count. Table 2 lists all fine-component records, and Table 3 lists current fallback identifiers. Legacy other_pulley_device is an alias for other_rotating_device in this disclosure; the deployed identifier is recorded explicitly. The additional high-voltage fallback is distinct from the six general morphological fallbacks.

Taxonomy snapshot: 15 systems, 68 fine-component records, and 7 fallback identifiers. Counts describe this version and do not limit the claims.

Table 1. Functional system identifiers

| Identifier | System |
| --- | --- |
| electrical | Charging, starting & electrical |
| ignition | Ignition |
| air_intake | Air intake |
| forced_induction | Forced induction |
| fuel_evap | Fuel & EVAP |
| exhaust_emissions | Exhaust & emissions |
| cooling | Cooling |
| lubrication | Lubrication |
| accessory_drive | Accessory drive |
| brakes | Brakes |
| steering | Steering |
| washer | Washer & wiper |
| body | Body & reference |
| tools | Tools |
| high_voltage | Hybrid/EV high voltage (identify only) |

Table 2. Fine identity and declared training-state mapping

| Fine identity | System | State | Fallback |
| --- | --- | --- | --- |
| battery | electrical | A | other_module_box |
| battery_terminal | electrical | A | other_cap_plug |
| fuse_relay_box | electrical | A | other_module_box |
| ecu_module | electrical | A | other_module_box |
| alternator | electrical | A | other_rotating_device |
| starter_motor | electrical | B | other_rotating_device |
| jump_start_post | electrical | B | other_cap_plug |
| wiring_harness | electrical | B | null |
| ignition_coil | ignition | A | other_sensor_actuator |
| coil_pack | ignition | B | other_module_box |
| spark_plug_wire | ignition | B | other_hose_line |
| air_filter_box | air_intake | A | other_module_box |
| air_intake_duct | air_intake | A | other_hose_line |
| maf_sensor | air_intake | A | other_sensor_actuator |
| throttle_body | air_intake | A | other_sensor_actuator |
| intake_manifold | air_intake | A | null |
| map_sensor | air_intake | B | other_sensor_actuator |
| iat_sensor | air_intake | B | other_sensor_actuator |
| intake_resonator | air_intake | B | other_module_box |
| vacuum_hose | air_intake | B | other_hose_line |
| turbocharger | forced_induction | B | other_rotating_device |
| intercooler_piping | forced_induction | B | other_hose_line |
| wastegate_actuator | forced_induction | B | other_sensor_actuator |
| fuel_rail | fuel_evap | B | other_hose_line |
| fuel_injector | fuel_evap | B | other_sensor_actuator |
| evap_purge_valve | fuel_evap | B | other_sensor_actuator |
| fuel_supply_line | fuel_evap | B | other_hose_line |
| hp_fuel_pump | fuel_evap | B | other_sensor_actuator |
| exhaust_manifold_heat_shield | exhaust_emissions | A | null |
| exhaust_manifold | exhaust_emissions | B | null |
| o2_sensor | exhaust_emissions | B | other_sensor_actuator |
| egr_valve | exhaust_emissions | B | other_sensor_actuator |
| coolant_reservoir | cooling | A | other_reservoir |
| radiator_cap | cooling | A | other_cap_plug |
| radiator_hose | cooling | A | other_hose_line |
| heater_hose | cooling | B | other_hose_line |
| thermostat_housing | cooling | B | null |
| radiator | cooling | B | null |
| radiator_cooling_fan | cooling | B | null |
| ect_sensor | cooling | B | other_sensor_actuator |
| oil_filler_cap | lubrication | A | other_cap_plug |
| oil_dipstick | lubrication | A | other_cap_plug |
| oil_filter | lubrication | B | null |
| valve_cover | lubrication | B | null |
| pcv_valve | lubrication | B | other_sensor_actuator |
| breather_hose | lubrication | B | other_hose_line |
| transmission_oil_dipstick | lubrication | B | other_cap_plug |
| serpentine_belt | accessory_drive | B | null |
| ac_compressor | accessory_drive | B | other_rotating_device |
| belt_tensioner | accessory_drive | B | other_rotating_device |
| power_steering_pump | accessory_drive | B | other_rotating_device |
| brake_fluid_reservoir | brakes | A | other_reservoir |
| brake_booster | brakes | B | null |
| abs_modulator_unit | brakes | B | other_module_box |
| brake_master_cylinder | brakes | B | null |
| clutch_reservoir | brakes | B | other_reservoir |
| power_steering_reservoir | steering | B | other_reservoir |
| washer_fluid_reservoir | washer | A | other_reservoir |
| windshield_wiper_motor | washer | B | null |
| engine_cover | body | A | null |
| strut_tower_brace | body | B | null |
| hood_latch_mechanism | body | B | null |
| hv_cable | high_voltage | B | hv_component |
| inverter_converter | high_voltage | B | hv_component |
| electric_ac_compressor | high_voltage | B | hv_component |
| hv_service_plug | high_voltage | B | hv_component |
| inverter_coolant_reservoir | high_voltage | B | other_reservoir |
| multimeter_diagnostic_tool | tools | A | null |

Table 3. Fallback identifiers

| Identifier | Description |
| --- | --- |
| other_reservoir | other fluid reservoir |
| other_module_box | other box or module |
| other_sensor_actuator | other sensor or actuator |
| other_hose_line | other hose or line |
| other_cap_plug | other cap, plug or handle |
| other_rotating_device | other belt-driven device |
| hv_component | high-voltage component (hybrid/EV) |

Let N(c) be the number of distinct, eligible, expert-reviewed real instances of fine component c, and V(c) the number of distinct supporting vehicles in the active training snapshot. Illustrative thresholds are promotion P = 60, diversity K = 8, and retention D = 40, with D < P. A B component becomes A when N(c) >= P and V(c) >= K. An A component becomes B with fallback training when N(c) < D. Otherwise its prior state is retained. Falling below K alone does not demote an A class under this example rule. Counts can decrease through withdrawn labels, duplicate removal, image exclusion, or changes to a versioned eligible dataset. No sliding-window behavior is assumed unless explicitly configured.

Synthetic instances never increment N or V. A promotion decision creates a new versioned label mapping and training manifest; the deployment model is updated after retraining. Metadata-only state changes do not create a new neural output class in an already trained network. The current taxonomy declares states, while the statistics utility reports recommendations; automatic state transitions are an optional embodiment.

### 6.6 Verdict model, optional calibration, and training admission

The available review implementation labels correct and bad_geometry as class-right (y_class = 1), and wrong_class, not_a_component, and duplicate as class-wrong (y_class = 0). Consequently a high class-right score does not certify an acceptable box or mask. In an optional admission embodiment, y_geometry separately records acceptable geometry or completed expert correction, and y_unique records a non-duplicate instance. A candidate is admitted only if class scoring and geometry/uniqueness gates satisfy the policy.

The implemented feature vector contains more than five scalar features: raw VLM confidence; source indicator; class one-hot indicators; log normalized size; log aspect ratio; proximity to the image border; log prior density; size-outlier distance; same-class count; maximum same-class and other-class IoU; total candidate count; and crop similarity signals. Vehicle GroupKFold is an evaluation procedure, not a predictive feature.

For crop evidence, normalized DINOv2 embeddings are compared by cosine similarity. s_pos is the mean of the top five similarities to reviewed class-right crops of the claimed class; s_neg is the mean of the top five similarities to class-wrong crops. The margin is s_pos - s_neg. A missing reference group yields zero in the implemented example, and an optional missing-support flag enables review routing. An additional feature is similarity to any class-right reference crop.

An example GBDT uses histogram gradient boosting, maximum 300 iterations, learning rate 0.05, maximum 15 leaves, and L2 regularization 1.0. These values describe an implementation example rather than limits of the method. The model provides a class-right score q. Optional isotonic or logistic calibration fitted on a distinct vehicle-separated validation set produces a probability for a defined event. Reliability diagrams, Brier score, and expected calibration error must be measured before asserting empirical probability calibration.

In a fold-isolated validation embodiment, all reviewed crop-reference banks, learned priors, preprocessing statistics, model fitting, and probability calibration use training-fold vehicles only; thresholds use a separate validation subset or nested folds. Final evaluation vehicles supply neither reference verdicts nor threshold tuning. The historical experiment excluded the query vehicle from crop matching, but computed crop features before the GroupKFold model split. Reference verdicts from other held-out vehicles could therefore influence features. Its metrics are retained as historical exploratory results, not evidence of fully isolated generalization.

A high score above t_accept with accepted geometry and uniqueness can enter a provenance-tracked training queue; intermediate scores enter expert review; low scores are excluded from training or queued for targeted inspection. Thresholds reflect the desired precision/coverage tradeoff. Automatic admission is optional and is not claimed to be enabled or validated by the historical report. Periodic sampled review, class/source stratification, drift monitoring, and reversible label withdrawal provide an optional feedback loop.

### 6.7 Vehicle expectations, non-observation, and inspection guidance

An optional knowledge-base record includes configuration identifier, applicable component identity, minimum expected count, functional system, visibility conditions, source/version, optional two-dimensional landmark-relative region, and inspection guidance references. A record can be conditioned on engine variant and equipment options rather than make/model alone. Metadata resolution returns matched, ambiguous, or unknown configuration states.

For a matched configuration, the reasoning unit compares expected identities with retained observed instances. An unmatched expected identity is emitted as expected_not_visible, meaning expected but not observed in this image. Reasons can include outside_view, possible_occlusion, unresolved_generic_match, low_detection_confidence, or unknown. A generic instance that could represent the expected fine class prevents a definitive missing-component assertion and is linked as unresolved evidence. Unknown or ambiguous vehicle metadata yields a conditional expectation or metadata_required state.

Non-observation alone does not establish absence, concealment, or malfunction. A DTC is separately supplied by diagnostic input, not inferred merely from a missing detection. A versioned relation maps the DTC and applicable component to a suggested inspection step, for example acquiring another viewpoint or checking manufacturer instructions. The output distinguishes observations from hypotheses and guidance.

A location marker for a non-observed component is optional. It requires a supported configuration diagram or visible landmarks with a stored relative region transformed into the image. Without those data, location is null and the UI shows a textual expectation only. A marker is labeled an estimated region rather than a detected box. High-voltage components in the current taxonomy are identified only, with specialist referral rather than disassembly guidance.

### 6.8 Illustrative worked example (not experimental evidence)

Assume an image W = 1600, H = 1200 and a 2 by 2 grid with o = 0.25. The tile dimensions are approximately 914.29 by 685.71 pixels. The lower-right tile has recorded integer origin (685,514). Suppose inference has already returned crop-pixel box (100,80,160,140) for an oil filler cap. Projection gives global box (785,594,845,654), or normalized box (0.490625,0.495,0.528125,0.545). Its mask vertices receive the same offset. This box is not adjacent to an internal tile border.

A global candidate has box (786,595,846,655). Their intersection is 59 by 59 pixels and union is 3719 square pixels, giving IoU about 0.936. With detector scores 0.78 and 0.71, respectively, NMS at 0.85 retains the tile candidate and its globally projected mask. An unrelated candidate with low spatial density is flagged for review rather than discarded by a hard prior. In this example only, a second battery candidate exceeds a configured count limit of one and is excluded from the retained set with reason count_limit; this limit is not the empirical count_max + 1 used in the reported experiment.

Assume the verdict model assigns q = 0.92. This is a hypothetical class-right score, not a measured result or certified calibrated probability. Without accepted geometry the candidate remains in review. A matched illustrative configuration expects a spark plug, but none is resolved in the photograph; the output preserves uncertainty and provides no unsupported location. FIG. 7 shows the coordinate and suppression steps.

```json
{
  "example_type": "illustrative_only",
  "vehicle_configuration": "example_petrol_configuration",
  "configuration_status": "matched",
  "visible_components": [{
    "id": "cap_1", "system": "lubrication",
    "class": "oil_filler_cap",
    "box_norm": [0.490625, 0.495, 0.528125, 0.545],
    "mask_transform": "tile_offset_685_514",
    "class_right_score": 0.92,
    "geometry_status": "review_required",
    "training_admission": "pending_review"
  }],
  "expected_not_visible": [{
    "class": "spark_plug", "reason": "unknown",
    "physical_absence_confirmed": false,
    "estimated_region": null,
    "guidance": "Check applicable service information and acquire another view."
  }]
}
```

### 6.9 Optional offline synthetic-data engine

A versioned asset library (602) associates each mesh/CAD asset with taxonomy identity, scale, pose, material slots, and configuration applicability. Complete engine bay scenes or individual instances are rendered. Randomization (604) samples camera pose, illumination, roughness/specularity, contamination overlays, and distractor placement; render seeds and asset identifiers are retained for reproducibility. Example proposed ranges include camera yaw +/-30 degrees about a reference view, 1-4 lights, light color temperature 3000-6500 K, and roughness 0.2-0.9. These are illustrative configuration ranges, not measured optimal values.

Renderer (606) outputs an RGB image and an instance-identification image. Generator (608) computes each visible binary mask from instance pixels and a tight box from their extrema. An optional second render without occluders provides the unoccluded projected mask; visible fraction is visible area divided by that unoccluded area, using the same camera pose. Zero denominators and fully invisible objects produce no visible-instance label. A sample inclusion threshold is 0.10.

For compositing, the final label follows the rendered alpha support after all occlusion and compositing operations, rather than the pre-composite mask. Mixer (610) assigns class synthesis budgets inversely to reviewed real support, for example B(c) = min(Bmax, ceil(k/(N(c)+1))) for configured positive k and cap Bmax. Synthetic records remain marked synthetic. Training (612) pretrains on synthetic or mixed images and fine-tunes on reviewed real images with a recorded schedule. Synthetic samples do not affect class promotion counts.

An assessment uses the same real held-out vehicle set, backbone, input resolution, annotation policy, and inference settings for real-only and synthetic-plus-real models. Report per-class and small-object metrics, mask metrics, compute cost, and uncertainty. No quantitative synthetic-data benefit has been established by the source reports cited below. All rendering precedes deployment inspection and does not require online 3D processing.

### 6.10 Reported experimental results and limitations

Table 4 reproduces the historical teacher-system report: 125 test images, checkpoint runs/segment/p5_reg/weights/avg5.pt, network input size 640, confidence threshold 0.25, class-matched box IoU >= 0.5, and ground-truth classes known to the detector. These are box-detection metrics, not mask-quality metrics. The report lists 443 eligible ground-truth instances across nine represented system groups. Distinct vehicle count, fleet distribution, and train/test vehicle-separation evidence are not supplied by this summary.

| Inference mode | Recall | Precision | Predictions |
| --- | --- | --- | --- |
| Full image | 0.582 | 0.632 | 408 |
| Full image + class-agnostic merge | 0.578 | 0.638 | 401 |
| Full image + 2x2 tiles | 0.621 | 0.552 | 498 |
| Tiles + class-agnostic merge | 0.616 | 0.575 | 475 |
| Tiles + merge + count cap | 0.614 | 0.581 | 468 |

Compared with full-image inference, the final mode raises recall by 3.2 percentage points and lowers precision by 5.1 percentage points. Relative to unmerged tiled inference, precision increases by 2.9 points and recall decreases by 0.7 points. Spatial hard filtering was not separately measured in this ablation. Reported system recall changes include lubrication 0.76 to 0.83 (42 GT instances), cooling 0.56 to 0.62 (16), air intake 0.55 to 0.59 (155), and electrical 0.55 to 0.58 (126). Rounded differences are percentage-point changes, not significance tests.

Table 5 reproduces historical verdict-model results on 4627 reviewed boxes from 177 vehicles: 2688 class-right and 1939 class-wrong. The model uses five-fold vehicle GroupKFold with the feature-bank limitation in Section 6.6.

| Model | AUROC | Coverage at class precision >=0.95 | Coverage at >=0.90 |
| --- | --- | --- | --- |
| VLM confidence only | 0.617 | 0.0% | 0.0% |
| Logistic model, all features | 0.823 | 12.1% | 29.7% |
| Gradient boosting, all features | 0.844 | 14.4% | 31.6% |
| Gradient boosting, without crop evidence | 0.797 | 8.7% | 19.1% |

At score >=0.9, reported coverage is 21.8% and class precision is 0.941. Approximately 5.9% of selected boxes are class-wrong under that target; additional class-right boxes may have bad geometry. Coverage-at-precision thresholds were selected using the reported out-of-fold results and are descriptive, not guarantees for an independently locked deployment threshold. AUROC measures discrimination and does not establish probability calibration. No empirical ROC curve is reproduced without original prediction/label pairs.

### 6.11 Technical interaction and additional evaluation

The proposed interaction is that tiled inference supplies small-component detail while shared-coordinate records allow global/tile duplicates to be resolved; count and position evidence then qualify retained instances; fallback states preserve uncertain identity for review; and configuration expectations explicitly separate unresolved components from observed detections. Review outcomes alter eligible class support and future training manifests, while synthetic sampling addresses low-support classes without artificially satisfying real-review promotion criteria.

The reported evidence supports only the stated historical detection and class-score observations. Further evaluation should isolate spatial evidence, each duplicate stage, count caps, fallback/promotion behavior, and synthetic training. Vehicle-separated manifests, one-to-one class-matched matching rules, annotation policy, object-size bins, mask AP/IoU, latency and memory, class/source calibration, and vehicle-bootstrap confidence intervals should be retained. Performance on unseen YMME configurations and expected-not-observed errors should be measured separately. These are evaluation requirements for future substantiation, not fabricated results.

## 7. PROPOSED PATENT CLAIMS

1. A computer-implemented method for visual inspection and functional decomposition of vehicle engine bay components, comprising: (a) receiving a two-dimensional image of a vehicle engine bay; (b) executing an instance-segmentation network on the image and on a plurality of overlapping local regions of the image to obtain candidate component instances having class assignments, bounding boxes, and segmentation masks; (c) transforming bounding boxes and segmentation masks from the local regions into a common coordinate system of the image; (d) suppressing duplicate candidate instances across image and local-region inference based on overlap in the common coordinate system; (e) processing the remaining instances using class-associated spatial evidence and supported component-count constraints; (f) associating retained instances with functional systems through a hierarchy comprising morphological groups and component identities; and (g) outputting a structured representation of retained observed instances and components expected for a matched vehicle configuration but not resolved among the observed instances, the latter being represented with non-observation uncertainty.

2. The method of claim 1, wherein a fine component identity is retained in an annotation record while a training label maps the identity to a generic fallback selected from other reservoirs, other sensors or actuators, other hoses or lines, other caps or plugs, other modules or boxes, and other rotating devices.

3. The method of claim 2, wherein an independent training state is entered when a reviewed real-instance count reaches a promotion threshold and supporting vehicles reach a diversity threshold, the independent training state is retained until the count falls below a lower retention threshold, and synthetic instances are excluded from both reviewed counts.

4. The method of claim 3, wherein the promotion threshold is 60 instances, the diversity threshold is eight vehicles, and the retention threshold is 40 instances, demotion retaining the fine identity while remapping its training label to the generic fallback.

5. The method of claim 1, wherein the local regions comprise an overlapping two-by-two grid and duplicate suppression comprises class-wise non-maximum suppression followed by class-agnostic non-maximum suppression at an intersection-over-union threshold of 0.85.

6. The method of claim 1, wherein candidates adjacent to an internal boundary of a local region are removed or flagged as truncated, and each retained mask is transformed using the same inverse resize and region offset as its corresponding box.

7. The method of claim 1, wherein spatial evidence comprises a smoothed histogram of normalized box centers and marginal tenth-to-ninetieth-percentile intervals, the intervals being used as soft evidence for unregistered views and a neutral prior being used when supporting data are insufficient.

8. The method of claim 1, wherein a component-count constraint is obtained from a matched vehicle configuration or from a reviewed training-image maximum plus a margin, and an unsupported count constraint is disabled.

9. The method of claim 1, further comprising scoring proposed labels using a verdict-trained gradient-boosting model with crop embedding similarity margin, visual confidence, spatial density, and geometric features, and admitting a proposed label to a training set only when configured class-score, geometry, and uniqueness gates are satisfied.

10. The method of claim 1, wherein an unresolved generic instance is linked to an expected fine component, and an estimated location for an unobserved component is displayed only when supported by a configuration diagram or registered visible landmarks.

11. A computer-implemented method for curating engine bay component training labels, comprising: (a) receiving reviewed candidate records with class-correctness verdicts and geometry status; (b) computing candidate features comprising visual confidence, geometry, spatial evidence, overlap statistics, and a difference between embedding similarity to reviewed class-right reference crops of a claimed class and similarity to class-wrong reference crops; (c) fitting a verdict-scoring model using the features; (d) selecting an admission threshold using vehicle-separated validation data with reference banks restricted to training vehicles; and (e) routing a new candidate to training admission, expert review, or exclusion according to the score and separate geometry and uniqueness conditions, while retaining provenance of the routing decision.

12. The method of claim 11, further comprising fitting a probability-calibration mapping on vehicle-separated validation data for a defined correctness event, and evaluating calibration on a separate vehicle-held-out set.

13. The method of claim 11, further comprising updating reviewed real-instance and vehicle counts following label correction or withdrawal and changing a fine component's independent-versus-fallback training mapping using distinct promotion and retention thresholds.

14. The method of claim 1, further comprising training the instance-segmentation network before inspection by rendering three-dimensional component assets associated with the hierarchy under randomized camera pose, illumination, surface properties, contamination, and occlusion, deriving instance masks and boxes from an instance-identification image, and fine-tuning on expert-reviewed real images, all rendering being confined to offline training.

15. The method of claim 14, wherein asset scenes are associated with vehicle configurations, visible-instance labels are excluded below a visible-to-unoccluded projected-area ratio, and synthetic sampling budgets decrease with reviewed real-class support without contributing to class-promotion counts.

16. The method of claim 14, wherein rendered instances are composited onto real backgrounds and final segmentation labels account for occlusion after compositing.

17. A vehicle engine bay inspection system comprising an image acquisition interface, memory storing a hierarchical component taxonomy, class-associated spatial evidence, component-count constraints, and vehicle-configuration expectations, and one or more processors configured to execute the method of claim 1, together with a display configured to distinguish observed instances from expected-but-not-observed components and their uncertainty.

18. A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the processors to perform the method of claim 1.

19. A training-label curation system comprising memory storing reviewed candidate records and reference crop embeddings, and one or more processors configured to perform the method of claim 11.

20. A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the processors to perform the method of claim 11.

## 8. ABSTRACT

A computer-implemented system inspects vehicle engine bay components using a global image branch and overlapping local-region branches of an instance-segmentation network. Candidate boxes and masks are transformed into image coordinates, duplicate instances are suppressed, and spatial evidence and supported component-count constraints qualify retained detections. A hierarchy associates detections with functional systems, morphological groups, and fine component identities. Reviewed real-instance and vehicle counts govern independent-versus-fallback training states using separate promotion and retention thresholds. A verdict-trained model scores proposed labels using appearance, geometry, position, and overlap evidence; training admission additionally requires geometry and uniqueness checks. Vehicle-configuration records identify expected but unobserved components with explicit uncertainty. An optional offline engine renders randomized three-dimensional component scenes, derives instance labels, and combines synthetic images with reviewed real images for training. Synthetic instances do not satisfy real-review promotion thresholds, and deployed inspection processes two-dimensional images.
