"""P5: rank candidate new detection classes by diagnostic value, with Jev judgments (text only).

    priority(c) = sum over DTC groups of w(dtc) * P(inspect c | dtc)  *  P(visible c)  *  P(visual check useful c)

Jev questions (one request per DTC group, one per component; all questions of a request run in parallel):
  inspect  "for DTC <code, meaning>, should a technician locate and visually check <component> in the engine bay?"
  visible  "is <component> visible from above with the hood open, without removing covers, on most gasoline cars?"
  useful   "can a visual check of <component> reveal a likely fault (unplugged connector, cracked hose, leak, level)?"
DTC weights: uniform unless --dtc-weights (json {first code of the group: weight}) gives real scan frequencies
(Innova data); without them the ranking is diagnostic breadth, not frequency.
Validation: `inspect` vs the expert-curated components of configs/diagnosis_knowledge.yaml (AUROC).
Candidates: taxonomy-v2 tier-B components + non-detectable components of the knowledge table.

    python scripts/data_pipeline/jev_class_priority.py --out docs/reports/class_priority_jev.md
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT / "artifacts" / "jev" / "class_priority.json"


def candidates(tax, knowledge):
    """{id: (name, description, current tier/detector state)}; tier-A classes are included as anchors."""
    out = {}
    for name, c in tax.components.items():
        cues = "; ".join(c.cues[:2])
        out[name] = (c.en or name, cues, f"tier {c.tier}")
    for cid, comp in knowledge.get("components", {}).items():
        if cid not in out:
            out[cid] = (comp["name"], comp["description"], "not in taxonomy v2")
        else:
            n, _, t = out[cid]
            out[cid] = (n, comp["description"], t)
    return out


def ask_all(cands, dtc_groups, workers):
    from typesafe_sdk import Noul, NoulCriteria
    from data_pipeline.jev_client import make_client

    def inspect_req(g):
        state = {"dtc_codes": ", ".join(g["codes"]), "meaning": g["meaning"]}
        qs = {cid: Noul(instructions=f"When diagnosing the fault in `meaning` (codes `dtc_codes`) on a gasoline car, should a technician "
                                     f"locate and visually check the {n} ({d}) in the engine bay?",
                        criteria=NoulCriteria(true="This part is a common cause of, or a standard first check for, this fault",
                                              false="This part is not normally related to this fault"))
              for cid, (n, d, _) in cands.items()}
        with make_client(timeout=60) as c:
            a = c.system_one(state=state, questions=qs).answers
        return g["codes"][0], {cid: round(float(a[cid].noul), 3) for cid in cands}

    def comp_req(item):
        cid, (n, d, _) = item
        state = {"component": n, "description": d}
        qs = {
            "visible": Noul(instructions="On most gasoline passenger cars, is `component` visible from above with the hood open, "
                                         "without removing covers, the air box or other parts?"),
            "useful": Noul(instructions="Can a visual check of `component` in a photo reveal a likely fault, such as an unplugged "
                                        "connector, a cracked or disconnected hose, a leak, corrosion or a low fluid level?"),
        }
        with make_client(timeout=60) as c:
            a = c.system_one(state=state, questions=qs).answers
        return cid, {"visible": round(float(a["visible"].noul), 3), "useful": round(float(a["useful"].noul), 3)}

    with ThreadPoolExecutor(workers) as pool:
        inspect = dict(pool.map(inspect_req, dtc_groups))
        comp = dict(pool.map(comp_req, list(cands.items())))
    return inspect, comp


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knowledge", type=Path, default=ROOT / "configs" / "diagnosis_knowledge.yaml")
    ap.add_argument("--dtc-weights", type=Path, default=None)
    ap.add_argument("--stats", type=Path, default=ROOT / "docs" / "reports" / "taxonomy_v2_stats.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--refresh", action="store_true", help="ignore the cached Jev answers")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "reports" / "class_priority_jev.md")
    args = ap.parse_args()

    from data_pipeline.taxonomy import load_taxonomy
    tax = load_taxonomy()
    knowledge = yaml.safe_load(args.knowledge.read_text(encoding="utf-8"))
    groups = knowledge["dtc"]
    cands = candidates(tax, knowledge)

    if CACHE.exists() and not args.refresh:
        cached = json.loads(CACHE.read_text(encoding="utf-8"))
        inspect, comp = cached["inspect"], cached["component"]
    else:
        inspect, comp = ask_all(cands, groups, args.workers)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"inspect": inspect, "component": comp}, indent=1), encoding="utf-8")

    weights = json.loads(args.dtc_weights.read_text()) if args.dtc_weights else {}
    stats = json.loads(args.stats.read_text(encoding="utf-8"))["components"] if args.stats.exists() else {}

    # validation against the curated DTC -> component lists
    pos, neg = [], []
    for g in groups:
        listed = set(g["components"])
        for cid in cands:
            (pos if cid in listed else neg).append(inspect[g["codes"][0]][cid])
    agree = auroc(pos, neg)

    rows = []
    for cid, (n, _, state) in cands.items():
        breadth = sum(weights.get(g["codes"][0], 1.0) * inspect[g["codes"][0]][cid] for g in groups)
        top = sorted(groups, key=lambda g: -inspect[g["codes"][0]][cid])[:3]
        prio = breadth * comp[cid]["visible"] * comp[cid]["useful"]
        rows.append((prio, cid, n, state, breadth, comp[cid]["visible"], comp[cid]["useful"],
                     stats.get(cid, {}).get("instances", 0), ", ".join(f"{g['codes'][0]} ({inspect[g['codes'][0]][cid]:.2f})" for g in top)))
    rows.sort(reverse=True)
    lines = ["# Candidate classes ranked by diagnostic value (P5, Jev)", "",
             f"{len(groups)} DTC groups from configs/diagnosis_knowledge.yaml · {len(cands)} components · DTC weights: "
             f"{'from ' + str(args.dtc_weights) if weights else 'uniform (no scan-frequency data yet)'}", "",
             f"**Jev vs the expert-curated DTC→component lists: AUROC {agree:.3f}** "
             f"({len(pos)} listed pairs, {len(neg)} unlisted). Values near 0.5 would mean Jev's inspect judgments are noise.", "",
             "priority = Σ_dtc w·P(inspect) × P(visible with hood open) × P(visual check useful)", "",
             "| rank | component | state | priority | DTC breadth | visible | visual useful | reviewed instances | strongest DTC links |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, (prio, cid, n, state, br, vis, use, inst, top) in enumerate(rows, 1):
        lines.append(f"| {k} | {cid} | {state} | {prio:.2f} | {br:.2f} | {vis:.2f} | {use:.2f} | {inst} | {top} |")
    new = [r for r in rows if r[3] != "tier A"][:5]
    lines += ["", "## Suggested first new classes (not yet trained)", ""]
    lines += [f"{i}. **{cid}** ({n}) — priority {p:.2f}, visible {v:.2f}, {inst} reviewed instances today"
              for i, (p, cid, n, _, _, v, _, inst, _) in enumerate(new, 1)]
    lines += ["", "Caveats: Jev judges general automotive knowledge from text; check the ranking with the automotive expert; "
              "replace uniform DTC weights with Innova scan frequencies before deciding."]
    report = "\n".join(lines) + "\n"
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
