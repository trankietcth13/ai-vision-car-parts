"""
Audit Qwen engine-bay box labels with Laya (text-only System-1 judgments).

Qwen writes a class name plus a short ``visual_evidence`` sentence for every box, but its
``confidence`` is mostly 1.0/0.95/0.9 (or missing -> 0.5), so it cannot rank labels for review.
Laya reads the evidence text and returns calibrated probabilities:

    match_noul   P(evidence describes the claimed class definition)      state = label+definition+evidence
    blind_choice P(claimed class) when Laya picks the class from the evidence alone
                 (shortlisted to --shortlist options + "none_of_these")   state = evidence only

Labels on reviewed images (data/engine_bay_review/verdicts) give ground truth:
    semantic error = wrong_class | not_a_component ; ok = correct | bad_geometry
The script reports AUROC and review-yield per signal against Qwen's own confidence, and writes
per-detection scores for building review queues.

Read-only on data/engine_bay_labeled (owned by the Qwen labeling job).

Usage:
    python scripts/data_pipeline/laya_label_audit.py                    # reviewed images only (evaluation)
    python scripts/data_pipeline/laya_label_audit.py --all              # score every detection
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data_pipeline"))
from qwen_grounding_annotator import CLASS_DEFINITIONS  # noqa: E402

SEMANTIC_ERRORS = {"wrong_class", "not_a_component"}
SEMANTIC_OK = {"correct", "bad_geometry"}
NONE_LABEL = "none_of_these"

MATCH_QUESTION = {
    "match": {
        "type": "noul",
        "instructions": (
            "An annotator labeled one object in a car engine-bay photo as `claimed_label` and described what "
            "they saw in `visual_evidence`. Does `visual_evidence` describe the component defined in "
            "`definition`, rather than a different engine-bay component, a generic hose/wire/connector, or "
            "a body panel?"
        ),
    }
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled", type=Path, default=PROJECT_ROOT / "data/engine_bay_labeled")
    ap.add_argument("--review", type=Path, default=PROJECT_ROOT / "data/engine_bay_review")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/engine_bay_laya_audit")
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shortlist", type=int, default=12, help="classes kept for blind_choice")
    ap.add_argument("--all", action="store_true", help="score every detection, not only reviewed ones")
    return ap.parse_args()


def box_key(class_name: str, box) -> tuple:
    return class_name, tuple(round(float(v), 3) for v in box)


def load_verdict_labels(review: Path) -> dict[tuple, str]:
    """(split, stem, class, box) -> reviewer decision, joined through the review packets."""
    labels: dict[tuple, str] = {}
    for vp in sorted((review / "verdicts").glob("*/*.json")):
        split, stem = vp.parent.name, vp.stem
        pp = review / "packets" / split / f"{stem}.json"
        if not pp.exists():
            continue
        packet = json.loads(pp.read_text(encoding="utf-8"))
        verdict = json.loads(vp.read_text(encoding="utf-8"))
        dec = {int(v["id"]): v.get("decision") for v in verdict.get("instances", [])}
        for inst in packet["instances"]:
            if inst.get("status") == "labeled" and inst["id"] in dec:
                labels[(split, stem) + box_key(inst["class_name"], inst["box_norm"])] = dec[inst["id"]]
    return labels


def load_detections(labeled: Path, verdicts: dict[tuple, str], only_reviewed: bool) -> list[dict]:
    rows = []
    for jp in sorted((labeled / "raw_annotations").glob("*/*.json")):
        ann = json.loads(jp.read_text(encoding="utf-8"))
        split, stem = jp.parent.name, jp.stem
        for idx, det in enumerate(ann.get("detections", [])):
            decision = verdicts.get((split, stem) + box_key(det["class_name"], det["bbox_norm_xyxy"]))
            if only_reviewed and decision is None:
                continue
            rows.append({
                "split": split, "image": stem, "det_index": idx,
                "class_name": det["class_name"], "bbox_norm_xyxy": det["bbox_norm_xyxy"],
                "qwen_confidence": det.get("confidence"),
                "visual_evidence": det.get("visual_evidence", ""),
                "decision": decision,
            })
    return rows


def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties count half."""
    pos, neg = scores[positives], scores[~positives]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def yield_at(suspicion: np.ndarray, positives: np.ndarray, frac: float) -> tuple[float, float]:
    """Review the top `frac` most suspicious: (precision of that queue, share of all errors caught)."""
    k = max(1, int(round(frac * len(suspicion))))
    top = np.argsort(-suspicion, kind="mergesort")[:k]
    hits = positives[top].sum()
    return hits / k, hits / max(1, positives.sum())


def main() -> None:
    args = parse_args()
    import laya

    verdicts = load_verdict_labels(args.review)
    rows = load_detections(args.labeled, verdicts, only_reviewed=not args.all)
    print(f"{len(verdicts)} reviewed instances; scoring {len(rows)} detections", flush=True)

    t0 = time.time()
    agent = laya.load(args.model, device=args.device) if args.device else laya.load(args.model)
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    # 1) match_noul: claimed label + definition + evidence
    t0 = time.time()
    states = [{"claimed_label": r["class_name"], "definition": CLASS_DEFINITIONS[r["class_name"]],
               "visual_evidence": r["visual_evidence"]} for r in rows]
    for r, res in zip(rows, agent.predict_batch(states, MATCH_QUESTION, batch_size=args.batch_size,
                                                sort_by_length=True)):
        r["laya_match"] = res["answers"]["match"]["noul"]
    print(f"match_noul: {time.time() - t0:.0f}s", flush=True)

    # 2) blind_choice: evidence only, shortlist classes by embedding, then choose
    t0 = time.time()
    criteria = dict(CLASS_DEFINITIONS)
    embed = laya.embed_fn_from_agent(agent)
    blind_instr = "Which engine-bay component does this annotator's visual description refer to?"
    for i, r in enumerate(rows):
        short = laya.shortlist_choice(r["visual_evidence"], criteria, embed, k=args.shortlist,
                                      instructions=blind_instr)
        opts = {c: criteria[c] for c in short}
        opts[NONE_LABEL] = "a generic hose, wire, connector, bracket, body panel or any part not listed above"
        ans = agent.predict(r["visual_evidence"], {"cls": {"type": "choice", "instructions": blind_instr,
                                                            "criteria": opts}})["answers"]["cls"]
        r["laya_blind_choice"] = ans["choice"]
        r["laya_blind_p_claimed"] = float(ans["probabilities"].get(r["class_name"], 0.0))
        r["laya_blind_confidence"] = ans["confidence"]
        if (i + 1) % 200 == 0:
            print(f"  blind_choice {i + 1}/{len(rows)}", flush=True)
    print(f"blind_choice: {time.time() - t0:.0f}s", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output / ("laya_scores_all.jsonl" if args.all else "laya_scores_reviewed.jsonl")
    out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    # evaluation on reviewed rows with a semantic decision
    ev = [r for r in rows if r["decision"] in SEMANTIC_ERRORS | SEMANTIC_OK]
    if not ev:
        print(f"wrote {out_jsonl}; no reviewed rows to evaluate")
        return
    pos = np.array([r["decision"] in SEMANTIC_ERRORS for r in ev])
    signals = {
        "qwen_confidence (baseline)": np.array([-(r["qwen_confidence"] or 0.5) for r in ev]),
        "laya match_noul": np.array([-r["laya_match"] for r in ev]),
        "laya blind P(claimed)": np.array([-r["laya_blind_p_claimed"] for r in ev]),
        "laya mean(match, blind)": np.array([-(r["laya_match"] + r["laya_blind_p_claimed"]) / 2 for r in ev]),
    }
    lines = [
        "# Laya label audit vs expert review",
        "",
        f"Evaluated {len(ev)} reviewed detections: {int(pos.sum())} semantic errors "
        f"({pos.mean():.1%}; {dict(Counter(r['decision'] for r in ev))}).",
        "Suspicion = negated probability; higher = more likely a wrong_class / not_a_component label.",
        "",
        "| signal | AUROC | queue top 10%: precision / errors caught | top 20% | top 30% |",
        "|---|---|---|---|---|",
    ]
    for name, s in signals.items():
        cells = [f"{p:.0%} / {c:.0%}" for p, c in (yield_at(s, pos, f) for f in (0.1, 0.2, 0.3))]
        lines.append(f"| {name} | {auroc(s, pos):.3f} | " + " | ".join(cells) + " |")

    agree = [r for r in ev if r["laya_blind_choice"] == r["class_name"]]
    disagree = [r for r in ev if r["laya_blind_choice"] != r["class_name"]]
    for tag, grp in (("blind choice == Qwen class", agree), ("blind choice != Qwen class", disagree)):
        if grp:
            err = sum(r["decision"] in SEMANTIC_ERRORS for r in grp) / len(grp)
            lines.append(f"\n- {tag}: {len(grp)} detections, semantic error rate {err:.1%}")

    per_cls = Counter(r["class_name"] for r in ev)
    lines += ["", "## Per class (>=15 reviewed)", "", "| class | n | error rate | AUROC match_noul |", "|---|---|---|---|"]
    for cls, n in per_cls.most_common():
        if n < 15:
            continue
        idx = np.array([r["class_name"] == cls for r in ev])
        lines.append(f"| {cls} | {n} | {pos[idx].mean():.0%} | {auroc(signals['laya match_noul'][idx], pos[idx]):.3f} |")

    report = "\n".join(lines) + "\n"
    (args.output / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out_jsonl}")


if __name__ == "__main__":
    main()
