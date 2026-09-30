# Revision notes and evidence checklist

Updated English specification: patent_specification_engine_bay_inspection_en_revised.md / .pdf / patent_specification_en_revised.html. Original review PDF is preserved. Rebuild with scripts/operations/build_patent_pdf_en_revised.py; the old builder still generates the original draft and should not be used for this revised version.

## Principal revisions

- Expanded description of tile generation, inverse coordinate transforms, box/mask alignment, border handling, two-stage NMS, spatial evidence, count caps, taxonomy state mapping, verdict features, review gates, metadata reasoning, and synthetic training.
- Added a full repository-derived taxonomy snapshot, seven grayscale schematics, and a clearly hypothetical worked example with JSON.
- Replaced misleading empirical ROC artwork: the old plotting script constructs power-law curves, not ROC points from actual predictions; its 0.059 vertical line is not established as false-positive rate by a precision of 0.941.
- Corrected recall gain to +3.2 percentage points and disclosed precision loss of 5.1 points against the full-image baseline. Added evaluation conditions and source limitations.
- Distinguished class-right labels from geometry acceptance. The source labels bad_geometry as positive, so the reported 94.1% class precision does not demonstrate correct boxes or masks.
- Disclosed that crop features are computed before GroupKFold and use other vehicles' reviewed verdicts. Leaving out the query vehicle does not isolate the held-out fold; historical AUROC remains exploratory until references are restricted to training folds.
- Replaced fixed system and class counts with versioned taxonomy records; legacy pulley fallback is explicitly aliased to the current rotating-device identifier. Added the repository high-voltage category.
- Replaced 14 claims with 20 proposed claims, including separate curation claims and inspection-specific system/media claims. Offline synthesis is dependent on inspection rather than a second stand-alone synthesis method. Claim grouping/unity and breadth still require prior-art and filing-jurisdiction review.
- Removed unsupported exclusivity, superiority, safe-auto-acceptance, and universal-calibration claims. Abstract is within 150 words.

## Sources used to verify repository behavior

- docs/reports/teacher_system_eval.md: detection metrics and known-class evaluation conditions.
- scripts/inference/teacher_system.py: overlap 0.25, border margin three pixels, class-wise NMS 0.5, agnostic NMS 0.85, cap count_max + 1, mask polygon transforms.
- configs/taxonomy_v2.yaml: systems, fine components, fallback identifiers, declared states, promotion thresholds, and high-voltage identify-only handling.
- scripts/data_pipeline/position_priors.py: training-split histogram, smoothing, p10-p90 support, neutral density and soft-prior caveat.
- docs/reports/label_confidence_v2.md and scripts/data_pipeline/label_confidence.py: review targets, features, estimator settings, cross-validation, threshold coverage, and reference-bank limitation.
- docs/references/IROS.2017.8202133.pdf: known domain-randomization background reference.

## Inventor verification and evidence still needed

1. Confirm the added optional embodiments reflect the inventors' conceived contribution. The new mathematical examples, geometry/uniqueness admission gates, fold-isolated validation, optional calibration, expectation schema, landmark location rules, and synthesis budget are proposed disclosures, not verified deployed features.
2. Confirm filing status. If an application has already been filed, compare every added paragraph and claim with the filed disclosure before using this draft as an amendment; do not assume new embodiments have the original priority date.
3. Supply vehicle-level train/validation/test manifests and fleet/YMME distribution. The source table is 125 images, not confirmed 125 independent vehicles. Supply one-to-one matching details and confirm the eligible-GT count against the evaluation manifest.
4. Re-evaluate verdict scoring with training-fold-only reference banks and priors, independent threshold selection, correct vehicle identity grouping, and a final untouched vehicle set. Measure calibration and geometry/uniqueness admission errors separately.
5. Provide real-only versus synthetic-plus-real results, small-object metrics, mask metrics, latency, memory, uncertainty intervals, and expected-not-observed error measurements. No numbers have been invented to fill these gaps.
6. Complete a patent/non-patent prior-art search and claim chart. Relevant comparison dimensions: local/global inference and duplicate handling; hierarchical fallback and support-based promotion; verdict-trained reference-crop curation; vehicle-conditioned non-observation; and low-support synthetic sampling excluding real-review counts. No novelty determination is implied by the revised background.
7. Review claims for eligibility, clarity, support, unity, and scope in the intended jurisdictions. The grayscale figures improve readability but have not been certified against all national drawing formalities.

## Prior-art comparison worksheet (research pending)

| Technique family | Established baseline to investigate | Claimed interaction requiring comparison |
| --- | --- | --- |
| Tiled detection/segmentation | Global/local inference and NMS literature/patents | Shared box and mask transforms plus structured engine-bay output |
| Hierarchical recognition | Generic fallback and incremental class learning | Reviewed real support/diversity hysteresis with identity preservation |
| Review-based label curation | Confidence selection, calibration, reference embeddings | Verdict target plus separate geometry/uniqueness admission conditions |
| Vehicle diagnostic reasoning | Configuration parts lists and DTC knowledge systems | Expected-but-unobserved uncertainty linked to unresolved visual instances |
| Domain randomization | Tobin et al., IROS 2017, DOI 10.1109/IROS.2017.8202133 | Taxonomy-linked rare-class sampling without inflating reviewed-real promotion counts |

This worksheet is not a completed prior-art search and supplies no assumed patent publication numbers.
