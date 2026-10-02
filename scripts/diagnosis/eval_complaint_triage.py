"""English evaluation set for complaint -> vehicle system triage (the offline app's future text model).

Data: configs/complaint_triage_eval_en.jsonl, one case per line:
    id, complaint, dtc            input (dtc may be empty; the complaint text is what is being tested)
    system, also_ok               expected system from configs/diagnosis_knowledge.yaml; also_ok = other defensible systems
    action, action_ok             stop_driving | ask_more_info | inspect (same policy as src/jev/diagnosis_triage.py)
    hazards                       active hazards (keys of HAZARDS); non-empty <=> action stop_driving, unless action_ok allows
    inspect                       components (knowledge ids) a good triage MUST list (recall check, not exhaustive)
    style, difficulty             customer | technician | terse | noisy;  easy | medium | hard
    label_status                  draft (expert-authored, not yet checked by a technician) | reviewed
The set is a held-out TEST set: never use it to train or prompt-tune the student or the teacher.

Usage:
    python scripts/diagnosis/eval_complaint_triage.py validate
    python scripts/diagnosis/eval_complaint_triage.py laya [--state dict|text] [--components] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jev.diagnosis_triage import (HAZARD_STOP, HAZARDS, SUSPECT_MIN, SYSTEM_CONF_MIN, VAGUE_MIN,  # noqa: E402
                                  build_state, load_knowledge, lookup_dtc)

EVAL_PATH = ROOT / "configs" / "complaint_triage_eval_en.jsonl"
SMOKE_PATH = ROOT / "configs" / "jev_triage_smoke_cases.jsonl"
OUT_DIR = ROOT / "data" / "complaint_triage_eval"
ACTIONS = {"stop_driving", "ask_more_info", "inspect"}
STYLES = {"customer", "technician", "terse", "noisy"}
LEVELS = {"easy", "medium", "hard"}
FIELDS = ["id", "complaint", "dtc", "system", "also_ok", "action", "action_ok", "hazards", "inspect", "style",
          "difficulty", "label_status"]


def load_cases(path: Path = EVAL_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(cases: list[dict], knowledge: dict) -> list[str]:
    """Return every problem found; an empty list means the set is consistent with the knowledge layer."""
    errors = []
    systems, components = set(knowledge["systems"]), set(knowledge["components"])
    seen_ids, seen_text = set(), set()
    smoke = {json.loads(l)["complaint"].strip().lower() for l in SMOKE_PATH.read_text(encoding="utf-8").splitlines() if l.strip()}
    for c in cases:
        cid = c.get("id", "?")
        if list(c) != FIELDS:
            errors.append(f"{cid}: fields {list(c)} != {FIELDS}")
            continue
        text = c["complaint"].strip().lower()
        if cid in seen_ids:
            errors.append(f"{cid}: duplicate id")
        if text in seen_text or text in smoke:
            errors.append(f"{cid}: duplicate complaint (or copied from the smoke cases)")
        seen_ids.add(cid); seen_text.add(text)
        for s in [c["system"], *c["also_ok"]]:
            if s not in systems:
                errors.append(f"{cid}: unknown system {s}")
        if c["system"] in c["also_ok"]:
            errors.append(f"{cid}: also_ok repeats the main system")
        for a in [c["action"], *c["action_ok"]]:
            if a not in ACTIONS:
                errors.append(f"{cid}: unknown action {a}")
        for h in c["hazards"]:
            if h not in HAZARDS:
                errors.append(f"{cid}: unknown hazard {h}")
        if bool(c["hazards"]) and c["action"] != "stop_driving":
            errors.append(f"{cid}: hazards {c['hazards']} but action {c['action']}")
        if (c["system"] == "unclear") != (c["action"] == "ask_more_info"):
            errors.append(f"{cid}: 'unclear' system and 'ask_more_info' action must go together")
        if c["system"] == "unclear" and (c["dtc"] or c["inspect"] or c["also_ok"]):
            errors.append(f"{cid}: an unclear case must have no dtc, inspect list or alternative system")
        for comp in c["inspect"]:
            if comp not in components:
                errors.append(f"{cid}: unknown component {comp}")
        if c["dtc"]:
            matched, unknown = lookup_dtc(c["dtc"], knowledge)
            listed = {comp for m in matched for comp in m["components"]}
            # Components implied by the DTC table must not contradict the expected list for code-driven cases.
            if matched and c["inspect"] and not listed & set(c["inspect"]):
                errors.append(f"{cid}: none of {c['inspect']} is listed for {c['dtc']} in the DTC table")
        if c["style"] not in STYLES or c["difficulty"] not in LEVELS or c["label_status"] not in {"draft", "reviewed"}:
            errors.append(f"{cid}: bad style/difficulty/label_status")
    return errors


def summary(cases: list[dict]) -> str:
    def row(name, counter):
        return f"{name}: " + ", ".join(f"{k} {v}" for k, v in sorted(counter.items(), key=lambda kv: -kv[1]))
    return "\n".join([
        f"{len(cases)} cases",
        row("system", Counter(c["system"] for c in cases)),
        row("action", Counter(c["action"] for c in cases)),
        row("hazards", Counter(h for c in cases for h in c["hazards"])),
        row("difficulty", Counter(c["difficulty"] for c in cases)),
        row("style", Counter(c["style"] for c in cases)),
        f"with DTC {sum(bool(c['dtc']) for c in cases)}, with also_ok {sum(bool(c['also_ok']) for c in cases)}, "
        f"with action_ok {sum(bool(c['action_ok']) for c in cases)}, reviewed {sum(c['label_status'] == 'reviewed' for c in cases)}",
    ])


def laya_questions(knowledge: dict, components: bool) -> dict:
    """The same questions as the Jev triage, minus urgency (the action policy below uses the hazards)."""
    q = {
        "system": {"type": "choice",
                   "instructions": "Which vehicle system is most likely at fault, based on `complaint` and `dtc_codes`?",
                   "criteria": dict(knowledge["systems"])},
        "too_vague": {"type": "noul", "instructions": "Is `complaint` too vague to point to any specific vehicle system "
                                                      "(for example only 'my car has a problem')?"},
    }
    q.update({f"hazard:{h}": {"type": "noul", "instructions": text} for h, text in HAZARDS.items()})
    if components:
        for cid, comp in knowledge["components"].items():
            q[f"component:{cid}"] = {"type": "noul", "instructions":
                f"Is the {comp['name']} ({comp['description']}) a plausible cause of the problem in `complaint` and "
                f"`dtc_codes`, so a technician should inspect it?"}
    return q


def text_state(complaint: str, matched: list[dict], unknown: list[str]) -> str:
    codes = [f"{d['code']} ({d['meaning']})" for d in matched] + unknown
    return complaint + (f" Trouble codes: {', '.join(codes)}." if codes else "")


def run_laya(cases: list[dict], knowledge: dict, components: bool, state: str = "dict") -> list[dict]:
    import laya
    agent = laya.load("convaiinnovations/laya")
    questions = laya_questions(knowledge, components)
    preds = []
    t0 = time.time()
    for i, c in enumerate(cases):
        matched, unknown = lookup_dtc(c["dtc"], knowledge)
        s = build_state(c["complaint"], matched, unknown, None) if state == "dict" else text_state(c["complaint"], matched, unknown)
        ans = agent.predict(s, questions)["answers"]
        hazards = {h: float(ans[f"hazard:{h}"]["noul"]) for h in HAZARDS}
        system, conf, vague = ans["system"]["choice"], float(ans["system"]["confidence"]), float(ans["too_vague"]["noul"])
        if max(hazards.values()) >= HAZARD_STOP:
            action = "stop_driving"
        elif not matched and (vague >= VAGUE_MIN or system == "unclear" or conf < SYSTEM_CONF_MIN):
            action = "ask_more_info"
        else:
            action = "inspect"
        p = {"id": c["id"], "system": system, "confidence": round(conf, 3), "too_vague": round(vague, 3),
             "hazards": {h: round(v, 3) for h, v in hazards.items()}, "action": action}
        if components:
            comp_p = {cid: float(ans[f"component:{cid}"]["noul"]) for cid in knowledge["components"]}
            dtc_listed = {comp for m in matched for comp in m["components"]}
            p["suspects"] = sorted(cid for cid, v in comp_p.items() if v >= SUSPECT_MIN or cid in dtc_listed)
        preds.append(p)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cases)} ({time.time() - t0:.0f}s)", flush=True)
    return preds


def score(cases: list[dict], preds: list[dict]) -> str:
    by_id = {p["id"]: p for p in preds}
    rows = [(c, by_id[c["id"]]) for c in cases if c["id"] in by_id]
    n = len(rows)
    strict = sum(p["system"] == c["system"] for c, p in rows)
    lenient = sum(p["system"] in [c["system"], *c["also_ok"]] for c, p in rows)
    act = sum(p["action"] == c["action"] for c, p in rows)
    act_ok = sum(p["action"] in [c["action"], *c["action_ok"]] for c, p in rows)
    stop_true = [p["action"] == "stop_driving" for c, p in rows if c["action"] == "stop_driving"]
    stop_pred = [c["action"] in ["stop_driving", *c["action_ok"]] for c, p in rows if p["action"] == "stop_driving"]
    lines = ["# Complaint triage eval (English)", "",
             f"{n} cases. Labels: {Counter(c['label_status'] for c, _ in rows)}.", "",
             "| metric | value |", "|---|---|",
             f"| system accuracy (strict) | {strict}/{n} = {strict / n:.1%} |",
             f"| system accuracy (also_ok accepted) | {lenient}/{n} = {lenient / n:.1%} |",
             f"| action accuracy (strict) | {act}/{n} = {act / n:.1%} |",
             f"| action accuracy (action_ok accepted) | {act_ok}/{n} = {act_ok / n:.1%} |",
             f"| stop_driving recall | {sum(stop_true)}/{len(stop_true)} |",
             f"| stop_driving precision (action_ok accepted) | {sum(stop_pred)}/{len(stop_pred)} |"]
    if all("suspects" in p for _, p in rows):
        need = [(set(c["inspect"]), set(p["suspects"])) for c, p in rows if c["inspect"]]
        rec = sum(len(a & b) for a, b in need) / max(1, sum(len(a) for a, _ in need))
        full = sum(a <= b for a, b in need)
        size = sum(len(p["suspects"]) for _, p in rows) / n
        lines += [f"| required components recalled | {rec:.1%} (all required in {full}/{len(need)} cases) |",
                  f"| suspects listed per case (mean) | {size:.1f} |"]
    # Selective accuracy: what the app gets if it asks for more information below a confidence threshold.
    lines += ["", "## System accuracy vs coverage (ask for more info below the threshold)", "",
              "| confidence >= | coverage | accuracy (also_ok accepted) |", "|---|---|---|"]
    for t in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7):
        kept = [(c, p) for c, p in rows if p["confidence"] >= t and c["system"] != "unclear"]
        total = sum(c["system"] != "unclear" for c, _ in rows)
        ok = sum(p["system"] in [c["system"], *c["also_ok"]] for c, p in kept)
        lines.append(f"| {t:.1f} | {len(kept)}/{total} | {ok / max(1, len(kept)):.1%} |")
    for key in ("difficulty", "style"):
        lines += ["", f"## By {key}", "", "| group | n | system (also_ok) | action (action_ok) |", "|---|---|---|---|"]
        groups = defaultdict(list)
        for c, p in rows:
            groups[c[key]].append((c, p))
        for g, items in sorted(groups.items()):
            s = sum(p["system"] in [c["system"], *c["also_ok"]] for c, p in items)
            a = sum(p["action"] in [c["action"], *c["action_ok"]] for c, p in items)
            lines.append(f"| {g} | {len(items)} | {s / len(items):.0%} | {a / len(items):.0%} |")
    lines += ["", "## Per system (recall, also_ok accepted) and most common wrong answers", "",
              "| expected | n | recall | wrong answers |", "|---|---|---|---|"]
    per = defaultdict(list)
    for c, p in rows:
        per[c["system"]].append((c, p))
    for s, items in sorted(per.items()):
        ok = sum(p["system"] in [c["system"], *c["also_ok"]] for c, p in items)
        wrong = Counter(p["system"] for c, p in items if p["system"] not in [c["system"], *c["also_ok"]])
        lines.append(f"| {s} | {len(items)} | {ok / len(items):.0%} | {', '.join(f'{k} {v}' for k, v in wrong.most_common(3))} |")
    lines += ["", "## Wrong system or action", ""]
    for c, p in rows:
        bad_s = p["system"] not in [c["system"], *c["also_ok"]]
        bad_a = p["action"] not in [c["action"], *c["action_ok"]]
        if bad_s or bad_a:
            lines.append(f"- `{c['id']}` want {c['system']}/{c['action']}, got {p['system']} ({p['confidence']:.2f})/{p['action']}: "
                         f"{c['complaint']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["validate", "laya"])
    ap.add_argument("--components", action="store_true", help="also ask the 33 component questions (about 5x slower)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--state", choices=["dict", "text"], default="dict",
                    help="dict = the Jev triage state {complaint, dtc_codes, vehicle}; text = complaint plus code meanings")
    args = ap.parse_args()
    knowledge, cases = load_knowledge(), load_cases()
    errors = validate(cases, knowledge)
    print(summary(cases))
    if errors:
        print("\n".join(["", f"{len(errors)} problems:", *errors]))
        sys.exit(1)
    print("valid")
    if args.mode == "laya":
        cases = cases[: args.limit]
        preds = run_laya(cases, knowledge, args.components, args.state)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"laya_en_{args.state}" + ("_components" if args.components else "")
        (OUT_DIR / f"{tag}_predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
        report = score(cases, preds)
        (OUT_DIR / f"{tag}_report.md").write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
