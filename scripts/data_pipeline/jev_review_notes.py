"""P1 label-policy audit with Jev: structure the expert reviewers' free-text notes on reservoir / cap / heat-shield boxes.

Why Jev here: the notes are written by the expert reviewer (not a VLM rationalising its own label), and the
question is a text judgment ("what does this note say the box covers / what evidence was used"). Pixels are
not involved; the pixel-level decision was already made by the reviewer. Output = policy findings, e.g. how
often coolant labels cover only the cap, which evidence identifies each reservoir, which confusions dominate.

    python scripts/data_pipeline/jev_review_notes.py run      # one Jev request per note, cached (resumable)
    python scripts/data_pipeline/jev_review_notes.py report --out docs/reports/reservoir_label_audit.md
Key: TYPESAFE_API_KEY or the project .env (via src/jev/client.py, never printed).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT / "artifacts" / "jev" / "review_notes.jsonl"
CLASSES = {"coolant_reservoir", "washer_fluid_reservoir", "brake_fluid_reservoir", "power_steering_reservoir",
           "radiator_cap", "exhaust_manifold_heat_shield"}
REVIEW_ROOTS = ["engine_bay_review", "engine_bay_review_hybrid", "engine_bay_review_p2"]

COVERS = {
    "whole_body": "the whole reservoir, tank or part body (with or without its cap)",
    "cap_only": "only a cap, filler neck or small top part, not the tank or part body",
    "partial": "a part that is partly hidden, cut off by the image border or only partly boxed",
    "other_object": "a different object from the one the label names",
    "not_stated": "the note does not say what the box covers",
}
EVIDENCE = {
    "cap_marking": "an icon, text or marking on the cap or label (DOT, windshield or spray symbol, hot-coolant warning, steering-wheel symbol, oil can, MIN/MAX)",
    "position": "where it sits in the engine bay (on the brake master cylinder, at the firewall, a front corner, on the radiator, next to a pump)",
    "fluid_colour": "the colour of the fluid or of the translucent tank",
    "shape_material": "only the shape, size or material",
    "none": "no identifying evidence is given",
}
IDENTITY = {
    "coolant_reservoir": "a coolant reservoir or expansion tank",
    "brake_fluid_reservoir": "a brake fluid reservoir",
    "washer_fluid_reservoir": "a windshield washer fluid reservoir or its filler neck",
    "power_steering_reservoir": "a power steering fluid reservoir",
    "radiator_cap": "a radiator or its pressure cap",
    "heat_shield_or_cover": "a metal heat shield, exhaust cover or other cover",
    "other_part": "another engine-bay part",
    "nothing": "background, an empty area or nothing identifiable",
}


def collect():
    rows = []
    for root in REVIEW_ROOTS:
        base = ROOT / "data" / root
        for vp in sorted(base.glob("verdicts/*/*.json")):
            pp = base / "packets" / vp.parent.name / f"{vp.stem}.json"
            if not pp.exists():
                continue
            v = json.loads(vp.read_text(encoding="utf-8"))
            insts = {i["id"]: i for i in json.loads(pp.read_text(encoding="utf-8"))["instances"]}
            for it in v.get("instances", []):
                cls = insts.get(it["id"], {}).get("class_name")
                if (cls in CLASSES or it.get("new_class") in CLASSES) and it.get("note"):
                    rows.append({"key": f"{root}/{vp.parent.name}/{vp.stem}#{it['id']}", "vehicle": vp.stem.split("__")[0],
                                 "claimed_class": cls, "decision": it["decision"],
                                 "corrected_class": it.get("new_class"), "note": it["note"]})
            for k, m in enumerate(v.get("missing", [])):
                if m.get("class_name") in CLASSES and m.get("note"):
                    rows.append({"key": f"{root}/{vp.parent.name}/{vp.stem}#missing{k}", "vehicle": vp.stem.split("__")[0],
                                 "claimed_class": m["class_name"], "decision": "missing_added",
                                 "corrected_class": None, "note": m["note"]})
    return rows


def questions():
    from typesafe_sdk import Choice, Noul
    return {
        "covers": Choice(instructions="According to `reviewer_note`, what does the labelled box cover?", criteria=COVERS),
        "evidence": Choice(instructions="Which visible evidence does `reviewer_note` rely on to identify the object?", criteria=EVIDENCE),
        "identity": Choice(instructions="According to `reviewer_note` and `decision`, what is the boxed object actually?", criteria=IDENTITY),
        "uncertain": Noul(instructions="Does `reviewer_note` express doubt about what the object is, for example 'likely', 'probably', 'unclear', 'may be', 'cannot confirm'?"),
    }


def cmd_run(args):
    from data_pipeline.jev_client import make_client
    rows = collect()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if CACHE.exists():
        done = {json.loads(l)["key"] for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if r["key"] not in done][: args.limit or None]
    print(f"{len(rows)} notes, {len(done)} cached, {len(todo)} to ask Jev", flush=True)
    qs = questions()

    def one(r):
        state = {"label_class": r["claimed_class"], "decision": r["decision"],
                 "corrected_class": r["corrected_class"] or "none", "reviewer_note": r["note"]}
        try:
            with make_client(timeout=30) as c:
                a = c.system_one(state=state, questions=qs).answers
            return {**r, "covers": a["covers"].choice, "covers_conf": round(float(a["covers"].confidence), 3),
                    "evidence": a["evidence"].choice, "identity": a["identity"].choice,
                    "identity_conf": round(float(a["identity"].confidence), 3), "uncertain": round(float(a["uncertain"].noul), 3)}
        except Exception as e:  # noqa: BLE001 - retried on the next run
            return {"key": r["key"], "error": f"{type(e).__name__}: {str(e)[:150]}"}

    ok = err = 0
    with ThreadPoolExecutor(args.workers) as pool, CACHE.open("a", encoding="utf-8") as f:
        for n, res in enumerate(pool.map(one, todo), 1):
            if "error" in res:
                err += 1
                if err <= 3:
                    print("  error:", res["error"], flush=True)
                continue
            ok += 1
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    print(f"done: {ok} ok, {err} errors (errors are retried on the next run)")


def pct(c, total):
    return ", ".join(f"{k} {v} ({v / max(total, 1):.0%})" for k, v in c.most_common())


def cmd_report(args):
    rows = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = ["# Reservoir / cap / heat-shield label audit (P1, Jev on expert review notes)", "",
             f"{len(rows)} expert notes on boxes labelled or corrected to {', '.join(sorted(CLASSES))} "
             f"(review rounds: Qwen, DeepSeek hybrid, Phase 2). Jev model: pinned in src/jev/client.py. "
             "Jev reads the reviewer's text only; the pixel decision is the reviewer's.", ""]
    for cls in ["coolant_reservoir", "brake_fluid_reservoir", "washer_fluid_reservoir", "radiator_cap",
                "power_steering_reservoir", "exhaust_manifold_heat_shield"]:
        rs = [r for r in rows if r["claimed_class"] == cls]
        if not rs:
            continue
        good = [r for r in rs if r["decision"] in ("correct", "bad_geometry", "missing_added")]
        bad = [r for r in rs if r["decision"] in ("wrong_class", "not_a_component", "duplicate")]
        lines += [f"## {cls} (VLM claimed / reviewer added: {len(rs)})", "",
                  f"- decisions: {pct(collections.Counter(r['decision'] for r in rs), len(rs))}",
                  f"- accepted boxes cover: {pct(collections.Counter(r['covers'] for r in good), len(good))}",
                  f"- evidence used on accepted boxes: {pct(collections.Counter(r['evidence'] for r in good), len(good))}",
                  f"- rejected boxes were actually: {pct(collections.Counter(r['identity'] for r in bad), len(bad))}",
                  f"- reviewer doubt (Jev noul ≥ 0.5): {sum(r['uncertain'] >= 0.5 for r in rs)} / {len(rs)}", ""]
        doubt = sorted((r for r in good if r["uncertain"] >= 0.5), key=lambda r: -r["uncertain"])[:5]
        if doubt:
            lines.append("Accepted but doubtful (re-check first):")
            lines += [f"  - `{r['key']}` ({r['uncertain']:.2f}): {r['note'][:140]}" for r in doubt]
            lines.append("")
    by_evid = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["decision"] in ("correct", "bad_geometry", "missing_added"):
            by_evid[r["claimed_class"]][r["evidence"]] += 1
    report = "\n".join(lines) + "\n"
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    # re-review queue: accepted coolant / brake / washer labels that are cap-only or doubtful
    queue = [r for r in rows if r["claimed_class"] in ("coolant_reservoir", "brake_fluid_reservoir", "washer_fluid_reservoir")
             and r["decision"] in ("correct", "bad_geometry", "missing_added") and (r["covers"] == "cap_only" or r["uncertain"] >= 0.5)]
    qp = args.out.with_name("reservoir_rereview_queue.jsonl")
    qp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in queue), encoding="utf-8")
    print(f"re-review queue: {len(queue)} boxes -> {qp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "reports" / "reservoir_label_audit.md")
    args = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
