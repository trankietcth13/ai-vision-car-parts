# PATENT SPECIFICATION - WHOLE-VEHICLE WORKING DRAFT

Revision date: 30 September 2026. The engine bay is a representative embodiment within a vehicle-wide inspection architecture. Broader embodiments described below are proposed disclosures; historical measurements concern engine bay data only.

## 1. TITLE OF THE INVENTION

METHOD AND SYSTEM FOR HIERARCHICAL MULTI-SCALE VISUAL INSPECTION AND EVIDENCE-BASED ASSESSMENT OF VEHICLE COMPONENTS AND REGIONS

## 2. FIELD OF THE INVENTION

The disclosure relates to computer-implemented inspection of vehicles using images, hierarchical region and component representations, multi-scale visual analysis, observation coverage, confidence-based review, and optional diagnostic information. Inspection can encompass exterior surfaces, passenger and cargo interiors, engine or propulsion compartments, wheels and tires, underbody structures, electrical systems, and accessible service components. Embodiments include combustion, hybrid, and electric vehicles, passenger vehicles, commercial vehicles, buses, and motorcycles, with the region hierarchy adapted to vehicle configuration.

## 3. BACKGROUND OF THE INVENTION

Vehicle inspection frequently involves separate processes for exterior appearance, interior equipment, tires, mechanical components, and diagnostic information. Captured views differ in scale, viewpoint, visibility, acquisition device, and vehicle configuration. A local detail may be detectable in a close-up while its relationship to a vehicle region is ambiguous. Conversely, a wide view can identify a region while obscuring small components or surface conditions. Combining these views can duplicate observations or incorrectly merge distinct components.

Vehicle-wide assessment also requires distinguishing a region that has not been inspected from an inspected region in which a component was not detected. A component may be absent from the vehicle configuration, outside the captured views, occluded, or unresolved by visual analysis. A diagnostic code or maintenance record supplies additional context but does not itself establish a visually confirmed defect.

Multi-scale detection, segmentation, image association, hierarchical classification, confidence estimation, and synthetic domain randomization are known technique families. The disclosed architecture coordinates these operations through evidence records associated with vehicle regions, component identities, and observation coverage. No claim that all existing systems lack any individual operation is made. The claims are proposed for prior-art comparison and jurisdiction-specific review.

## 4. SUMMARY OF THE INVENTION

An inspection processor receives one or more views associated with a vehicle and identifies a region represented by each view. It analyzes a view at a contextual scale and at one or more local scales, resolves repeated observations within a view using a shared image coordinate system, and associates resulting evidence with a vehicle hierarchy. Evidence from separate views may subsequently be associated with a common component using region, appearance, configuration, and geometric context.

The resulting inspection representation separates observed component or surface findings from the evidence coverage of applicable regions. A region can be observed, partially observed, or not inspected; an expected component can be observed, unresolved, or expected but not observed. These states govern whether the system reports a finding, requests a targeted additional view, or retains uncertainty. A hierarchical representation can use a tree for ownership and a graph for shared functions or relationships.

Optional embodiments add region-specific models, condition assessment, multi-view association, change tracking, diagnostic information, review-trained confidence scoring, class-promotion hysteresis, and synthetic training. Any such operation uses the same region/component evidence records. The method need not receive views of every region to inspect a vehicle region, and it does not require a particular grid size, model vendor, vehicle-system count, score threshold, or three-dimensional reconstruction.

## 5. BRIEF DESCRIPTION OF THE DRAWINGS

FIG. 1 illustrates the vehicle-wide coverage concept: exterior and body (120), cabin and cargo (130), wheels and tires (140), underbody and chassis (150), propulsion compartment (160), and electrical/energy components (170).

FIG. 2 illustrates acquisition (202), region routing (204), contextual/local analysis (206), evidence resolution (208), hierarchy and coverage store (210), assessment (212), and output or targeted recapture (214).

FIG. 3 illustrates a vehicle-region-system-component hierarchy (300) with a separate relationship graph (320).

FIG. 4 illustrates contextual analysis (402), local-region analysis (404), common-coordinate resolution (406), and optional association of evidence across separate views (408).

FIG. 5 illustrates observation coverage (502), applicability (504), evidence states (506), and targeted additional capture (508).

FIG. 6 illustrates review and training feedback (600), including separate class and geometry review, admission gates, and real-review-supported class states.

FIG. 7 illustrates component-linked condition evidence (700), optional diagnostic context (720), and evidence-preserving longitudinal comparison (740).

FIG. 8 illustrates an optional synthetic-data engine (800) with assets from multiple vehicle regions and offline training for deployed inspection.

The drawings are schematic vector graphics. Dimensions, component placements, and illustrated view counts are not physical measurements or limits of the claims.

## 6. DESCRIPTION OF REPRESENTATIVE EMBODIMENTS

### 6.1 Inspection scope and region profiles

A vehicle profile defines applicable regions, systems, component types, and optionally equipment variants. Vehicle metadata can include VIN-derived configuration, year/make/model, propulsion type, axle arrangement, trim, or fleet asset identifier. An unresolved profile is retained as unknown or conditional rather than replaced by an assumed configuration.

Table 1 gives representative inspection regions. A region is a semantic acquisition and ownership context; it need not be a fixed rectangle. A vehicle may have more than one propulsion compartment or cargo region, and different region names may map to the same hierarchy node. A motorcycle profile can omit a passenger cabin while retaining exposed drivetrain, wheels, frame, and rider controls. A bus or truck profile can include multiple axles and separate cargo or equipment compartments.

| Region family | Representative targets | View-dependent limitations |
| --- | --- | --- |
| Exterior/body | Panels, bumpers, doors, glazing, lights, mirrors, trim, visible surface conditions | Reflections, lighting and viewpoints affect apparent surface condition |
| Interior/cargo | Seats, restraints, dashboard, controls, displays, trim, accessible cargo fixtures | Hidden mechanisms and internal electronics are not established from appearance |
| Wheels/tires/brakes | Tires, rims, valve areas, wheel fasteners, visible brake parts | Internal wear and calibrated dimensions need appropriate views or measurements |
| Underbody/chassis | Frame, suspension, steering links, exhaust, shields, accessible lines | Access, road contamination and occlusion limit coverage |
| Propulsion compartment | Engine, intake, cooling, lubrication, accessories, service points | A covered or internal part may remain expected but unobserved |
| Electrical/energy | Visible wiring, connectors, charge port, battery enclosure, energy-system housings | Appearance does not establish internal battery health or electrical isolation |

Whole-vehicle coverage means the architecture can represent these applicable regions. A complete-inspection indication is emitted only when the configured coverage criteria for that vehicle and inspection task are satisfied. It is not inferred from one image or from the mere existence of a whole-vehicle hierarchy.

### 6.2 Acquisition and region routing

Inputs include still images, selected video frames, or views from handheld, fixed, drive-through, underbody, robotic, or accessible borescope cameras. A view record retains vehicle/session identifier, view identifier, acquisition time, image orientation, source, and region assignment. A region assignment may be supplied by the operator, inferred from visual landmarks, or determined by a known capture station. Ambiguous assignments can remain candidates until reviewed.

Image quality and coverage are checked separately from component inference. Blur, glare, excessive distance, cropping, or an unopened compartment can reduce the observation status of a region. Region routing selects an applicable detector, segmenter, condition model, or shared model with region context. Images need not be jointly reconstructed into a three-dimensional vehicle model.

### 6.3 Contextual and local visual analysis

A contextual branch evaluates a full view or region overview. Local branches evaluate overlapping subregions, selected regions of interest, or a higher-detail image pyramid. These branches may execute sequentially or concurrently. Their purpose is to preserve both region context and visual detail; a fixed two-by-two tiling scheme is one embodiment.

Each resulting candidate carries a class or generic identity, score, box or mask, branch/view provenance, and region context. Local results are transformed into the coordinates of their parent view using the stored crop offset and inverse resize. Masks, when used, undergo the same transform as their boxes. Candidates truncated at internal crop boundaries can be flagged or excluded in favor of more complete observations.

Within-view resolution uses overlap and compatible instance evidence to suppress repeated candidates or retain the best-supported box/mask. Class-wise and class-agnostic NMS are examples; compatible mask fusion is an alternative. Distinct nested targets, such as a wheel and a brake component, are preserved through a hierarchy-aware compatibility rule rather than automatically merged because their boxes overlap. Region-specific count limits and position evidence can flag implausible candidates, while unsupported limits are disabled. Viewpoint-dependent spatial priors are used as soft evidence unless image registration justifies stronger constraints.

### 6.4 Hierarchical evidence and association across views

The evidence store associates findings with a hierarchy such as vehicle -> region -> functional system -> component group -> component instance. Ownership can use a tree, while edges such as connected_to, part_of, serves_system, adjacent_to, and possible_same_instance express additional relationships. An electrical connector can belong to a physical harness while serving a propulsion or lighting subsystem. A condition finding is attached to its component or surface region, not treated as an unrelated vehicle-level label.

For separate views, pixel overlap alone is insufficient for identity matching. An optional association unit compares vehicle/session identity, region and side, component class, appearance embeddings, visible landmarks, and available geometric correspondence. A left-front wheel is not merged with a right-front wheel solely because they look alike. Known station/view labels, landmark matches, or operator-confirmed positions resolve that distinction. An ambiguous match remains an association hypothesis with provenance rather than a confirmed physical identity.

The common-coordinate operation in Section 6.3 applies within each parent view. Cross-view association instead links view-local evidence to a shared semantic instance; it does not assume every image has a shared pixel frame. Independent observations and contradictions are retained for review. This separation permits cameras with different perspectives to contribute evidence without mandatory 3D registration.

### 6.5 Coverage, applicability, and expected components

For each applicable region, the processor stores observation coverage derived from accepted view quality, view/region overlap, and visibility of task-relevant landmarks or target surfaces. A policy can require multiple viewpoints, a sufficiently complete surface mask, or operator-confirmed access. The policy distinguishes observed, partially observed, not inspected, and not applicable. These states concern evidence coverage rather than physical component condition.

Expected component records are selected from a resolved vehicle profile or inspection template and compared with retained evidence. A component can be observed, unresolved_generic_match, expected_not_observed, or applicability_unknown. The identifier expected_not_visible may be retained for compatibility, but means expected and not observed in the available views. It does not confirm concealment, absence, or failure. A generic detection that could represent the component is linked as unresolved evidence.

If a target lacks acceptable coverage, the processor generates a targeted recapture instruction. An instruction identifies the region or landmark, required viewpoint or detail, and the reason: glare, missing side, unresolved component, or possible occlusion. A location overlay for a non-observed target requires supported landmarks, a matched configuration diagram, or an established station coordinate model; otherwise guidance remains textual. After recapture the system updates coverage and reconciles evidence, retaining prior view provenance.

### 6.6 Component condition and diagnostic context

An optional condition unit classifies visible findings such as a panel dent, surface crack, fluid trace, connector separation, corrosion, missing visible trim, or tire surface anomaly. It can use segmentation, a trained image model, a configuration rule, or reviewed visual evidence. A record stores finding type, target instance or region, supporting view, localization, confidence, and assessment state. Uncertain findings remain suspected or review_required. A visual anomaly is not automatically translated into an internal mechanical failure.

An optional severity field is qualitative unless a supported measurement or calibrated model establishes a quantitative scale. Tire tread depth, deformation dimensions, and similar measurements require scale references or a validated measurement arrangement. Internal battery capacity, electrical isolation, and hidden brake thickness are not asserted solely from RGB appearance.

Optional OBD/DTC input, maintenance history, mileage, or supported sensor observations remain separately typed evidence with timestamps and provenance. A configured relation can connect that evidence to a component and suggest an inspection step. Conflicting visual and diagnostic evidence is explicitly retained. No DTC is asserted from non-detection alone, and high-voltage disassembly guidance is not inferred from recognition of an enclosure.

### 6.7 Review, uncertainty, and training feedback

A review-trained model can score class correctness using detector or vision-language confidence, reference-crop appearance similarity, geometry, spatial context, overlap, and region-specific features. Similarity can use the difference between agreement with reviewed class-right references and agreement with rejected references. Class correctness, localization acceptance, condition correctness, and duplicate status are separately defined review targets. A high class score does not certify all targets.

Candidate labels are admitted, reviewed, or excluded according to supported score thresholds and geometry/uniqueness gates. Optional probability calibration is fitted for a defined target using separate validation vehicles. Reference banks and learned priors in each evaluation fold are built from training vehicles only; final threshold evaluation uses untouched vehicles. The system retains reviewer decisions and can withdraw a previously admitted label.

Fine identities remain available in annotation records even where the training label uses a generic fallback, such as other enclosure, other line, other fastener, or other surface anomaly. An independent class is promoted only when eligible expert-reviewed real support and vehicle diversity reach configured thresholds. A lower retention threshold governs demotion, preserving identity and changing the training mapping. Changes enter a versioned training manifest and take effect in a corresponding updated model. Synthetic examples do not increment real-review support. Thresholds and region catalogs are configurable rather than fixed vehicle-wide counts.

### 6.8 Longitudinal evidence and report generation

An optional longitudinal embodiment links sessions to the same vehicle and matches corresponding region/component identities. It compares normalized finding states or registered corresponding surfaces, accounting for viewpoint, lighting, replacement, and capture-quality differences. A new visual mark is reported as a candidate change until evidence supports association and condition comparison. Replaced parts create an identity/version event rather than an assumed persistent instance.

Outputs include a region/system tree or graph, observation coverage, localized findings, uncertainty, linked diagnostic context, and requested additional views. Reports retain model/taxonomy versions, source-view references, vehicle-profile status, and review history. A fleet or maintenance interface can receive the structured records without becoming a required element of the inspection method. Execution may be local, on a server, or split between acquisition and processing devices.

### 6.9 Optional vehicle-wide synthetic training

An offline asset library associates exterior panels, interior fixtures, wheels, chassis parts, propulsion components, and electrical enclosures with the same region and component identifiers used by inspection. Region scenes, individual components, and proposed visual conditions can be rendered under randomized camera pose, illumination, material appearance, contamination, and occlusion. The renderer outputs color images and instance/surface identifiers from which boxes, masks, region labels, and configured condition labels are derived.

Labels follow the final rendered visibility and compositing operations. Unobserved instances do not receive visible-instance masks. Synthetic records preserve asset, region, configuration, and render provenance. Sampling can concentrate on rare components or conditions, while real-reviewed counts alone control class promotion. Pretraining or mixed training is followed by adaptation to reviewed real images. Real held-out vehicles evaluate transfer; the present draft supplies no invented whole-vehicle or synthetic-transfer measurements. Offline scene rendering does not require online 3D reconstruction.

### 6.10 Representative inspection scenarios

Scenario A - propulsion compartment: an overview resolves major housings while local crops identify a small service cap. Shared parent-view coordinates resolve duplicates. A covered expected spark plug remains unobserved with uncertainty, and a cap whose identity is plausible but box geometry is unresolved remains in review. This preserves the original engine bay embodiment.

Scenario B - exterior and wheels: a left-side overview and a detail view provide evidence for a front door surface and a left-front wheel. The door condition is linked to its surface region; the wheel is associated using the left-front position and viewpoint evidence. A missing right-side view keeps that side not inspected and triggers a right-side capture request. The system does not report an uninspected side as defect-free.

Scenario C - underbody: an accepted underbody view identifies an accessible suspension link and a fluid-like visual trace. The trace is a localized suspected condition associated with nearby components. Occluded brake surfaces remain partially observed; a closer view is requested. The trace alone does not establish a leak source or internal brake wear.

Scenario D - electric vehicle and cabin: the profile makes a combustion engine inapplicable and identifies charge-port and battery-enclosure regions. Visible enclosure findings and a supplied diagnostic code remain separate evidence. Cabin images resolve seats and controls; a covered restraint mounting remains unobserved. Battery state of health is not inferred from enclosure appearance. These scenarios are illustrative proposed embodiments, not reported trials.

### 6.11 Existing evidence and applicability limits

The repository's engine bay teacher report records 125 test images at network input size 640 and class-matched box IoU >=0.5. Full-image recall/precision is 0.582/0.632; tiled inference plus duplicate processing and count caps yields 0.614/0.581. This is a recall gain of 3.2 percentage points with precision loss of 5.1 points. It establishes no exterior, cabin, wheel, underbody, or electric-vehicle performance.

The historical verdict report records 4627 reviewed boxes from 177 vehicles and AUROC 0.844 versus 0.617 for raw VLM confidence. At score >=0.9, reported coverage is 21.8% and class precision is 0.941. Its target counts bad_geometry as class-right, and crop references were computed before the vehicle-fold model split. Accordingly these are exploratory class-scoring observations, not fully isolated calibration, localization-admission, or vehicle-wide validation. No empirical ROC curve is reconstructed in the drawings.

Vehicle-wide validation should separately measure region coverage errors, cross-view false merges, class and condition errors, recapture usefulness, uncertainty, and transfer across vehicle profiles. The method can inspect all applicable regions over an accepted view set, but hidden internal functions remain outside visual evidence unless supported by separately identified measurements.

## 7. PROPOSED PATENT CLAIMS

1. A computer-implemented method for visual inspection of a vehicle, comprising: (a) receiving one or more images associated with the vehicle; (b) assigning each image to one or more vehicle regions represented in a hierarchical vehicle representation; (c) analyzing at least one image at a contextual scale and one or more local scales to generate candidate component or surface observations; (d) transforming local-scale observation locations into coordinates of their parent image and resolving repeated observations within that image; (e) associating retained observations with region and component or surface identities in the hierarchical vehicle representation; (f) determining an observation-coverage state for a represented region from accepted image evidence; and (g) generating a structured inspection output distinguishing observed findings from unresolved or uninspected regions according to the observation-coverage state.

2. The method of claim 1, wherein represented regions include exterior body regions, interior or cargo regions, wheels or tires, underbody or chassis regions, propulsion compartments, and electrical or energy-system regions, and the method is applicable to a subset or all of the regions applicable to a vehicle configuration.

3. The method of claim 1, wherein a vehicle profile determines region and component applicability for a combustion, hybrid, or electric vehicle, and a component not applicable to the profile is distinguished from an applicable component not observed in available images.

4. The method of claim 1, wherein acquisition comprises handheld imaging, fixed-station imaging, selected video frames, underbody imaging, robotic imaging, or accessible borescope imaging, with a retained view/session identifier and region assignment.

5. The method of claim 1, wherein a region routing unit selects a region-specific visual model or supplies region context to a shared visual model.

6. The method of claim 1, wherein local-scale analysis comprises overlapping tiles, selected regions of interest, or a multi-resolution image representation, and contextual and local branches are executed sequentially or concurrently.

7. The method of claim 1, wherein observations include segmentation masks, and the masks and corresponding boxes are transformed using the same crop offset and inverse resize before duplicate resolution.

8. The method of claim 1, wherein repeated-observation resolution comprises class-wise or class-agnostic non-maximum suppression or compatible mask fusion, and a component relationship prevents distinct nested targets from being merged solely because their locations overlap.

9. The method of claim 1, wherein position evidence or component-count constraints are conditioned on a region or vehicle profile, and an unsupported constraint is disabled or treated as soft evidence.

10. The method of claim 1, further comprising associating observations from separate images with a shared component identity using vehicle/session identity, region or side, appearance, landmarks, or geometric correspondence, while retaining ambiguous associations as hypotheses.

11. The method of claim 10, wherein within-image coordinate resolution is separate from semantic instance association across images, and the association does not require a shared pixel coordinate frame or a three-dimensional vehicle reconstruction.

12. The method of claim 1, wherein the hierarchical representation comprises vehicle, region, system, component group, and instance nodes, with relationship edges identifying shared functions, connectivity, or ownership.

13. The method of claim 1, wherein observation-coverage states distinguish observed, partially observed, not inspected, and not applicable according to an inspection policy and accepted view quality.

14. The method of claim 1, further comprising comparing retained observations with vehicle-profile expectations and representing an expected but unobserved component with uncertainty distinguishing non-observation from confirmed physical absence or malfunction.

15. The method of claim 14, wherein a generic observation is linked to an unresolved expected fine component, and a location overlay for an unobserved component is generated only when supported by landmarks, a configuration diagram, or a station coordinate model.

16. The method of claim 1, further comprising generating a targeted additional-capture instruction identifying a region, required viewpoint or detail, and deficient evidence, and updating the observation-coverage state after receiving a further image.

17. The method of claim 1, further comprising associating a localized visible condition finding with a retained component or surface identity and storing supporting view provenance and a condition-assessment state.

18. The method of claim 17, wherein a supplied diagnostic code, maintenance record, or sensor observation is retained as separately typed evidence linked to the component or region, and contradictions between that evidence and visual findings are preserved.

19. The method of claim 1, further comprising comparing evidence from different sessions associated with the same vehicle using corresponding region or component identities and recording a condition change or component replacement with view provenance.

20. The method of claim 1, further comprising scoring proposed training labels using reviewed-verdict evidence and admitting a label only when class-score, geometry, and uniqueness conditions satisfy a configured policy.

21. The method of claim 20, wherein a fine component retains its annotation identity while its training label maps to a generic fallback, promotion to an independent training class requires reviewed real-instance support and vehicle diversity, and demotion uses a lower retention threshold.

22. The method of claim 1, wherein a deployed visual model is trained using offline synthetic images from taxonomy-linked assets representing multiple vehicle regions, rendered under randomized viewpoint, lighting, material, and occlusion, and adapted to expert-reviewed real images, synthetic instances being excluded from real-review support for class promotion.

23. A vehicle visual inspection system comprising an acquisition interface, memory storing a hierarchical vehicle representation and region-coverage records, and one or more processors configured to: associate received vehicle images with vehicle regions; generate component or surface observations through contextual and local-scale analysis; transform and resolve local observations in their parent-image coordinates; associate retained observations with region and component or surface identities; determine region observation coverage from accepted image evidence; and output observed findings separately from unresolved or uninspected regions according to the coverage, together with a display or output interface for the resulting structured inspection representation.

24. A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the processors to perform the method of claim 1.

25. A computer-implemented method for curating visual inspection training labels across vehicle regions, comprising: receiving reviewed component or surface candidate records with separately defined class and localization verdicts; computing region-contextual candidate features including visual confidence and a margin between embedding similarity to reviewed class-right references and similarity to rejected references; fitting a verdict-scoring model using those features; selecting a routing threshold using vehicle-separated validation data with reference banks restricted to training vehicles; and routing a new candidate to admission, review, or exclusion based on the score and separate localization and uniqueness conditions while retaining review provenance.

26. The method of claim 25, further comprising fitting a probability-calibration mapping for a defined review target using separate validation vehicles and evaluating the mapping on vehicle-held-out data.

27. The method of claim 25, further comprising maintaining fine identities with generic fallback training labels and changing independent-class state according to reviewed real-instance and vehicle counts using distinct promotion and retention thresholds, without counting synthetic instances.

28. A training-label curation system comprising memory storing region-associated reviewed candidate records and reference embeddings, and one or more processors configured to perform the method of claim 25.

29. A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the processors to perform the method of claim 25.

30. The method of claim 1, wherein an inspection-completion indicator is issued only when configured observation-coverage criteria for the applicable regions of the vehicle profile and inspection task are satisfied.

## 8. ABSTRACT

A computer-implemented system visually inspects vehicle regions using contextual and local-scale analysis of one or more images. Local observations are transformed into their parent-image coordinates and repeated observations are resolved. Retained component and surface evidence is associated with a hierarchical vehicle representation spanning applicable exterior, interior, wheel, underbody, propulsion, and electrical regions. Observation coverage distinguishes inspected, partially observed, and uninspected regions. Vehicle-profile expectations identify applicable but unobserved components with uncertainty, and deficient evidence can trigger targeted additional capture. Optional embodiments associate instances across views, link localized condition findings with diagnostic context, compare inspection sessions, and curate labels using review-trained confidence with separate localization and uniqueness gates. Generic fallback training states use reviewed real support and promotion hysteresis. Offline synthetic training supplies region-linked examples without satisfying real-review promotion counts. The output preserves evidence provenance and separates observations from unresolved assessments.
