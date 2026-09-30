"""Diagnosis triage with Jev: complaint text + DTC codes -> system, urgency, hazards, components to inspect.

Division of work (docs/plans/diagnosis_guidance_architecture_2026-09-29.md, layer [K]):
- code: DTC lookup in configs/diagnosis_knowledge.yaml, combining evidence, all thresholds and actions;
- Jev (one request, all questions in parallel): semantic judgments on the free-text complaint.
The `detector_targets` of the result are the L1 classes to locate and crop in the photo.

Thresholds below are starting defaults, not tuned: evaluate them on real Innova complaints before relying on them.
Jev is trained mainly on English; Vietnamese complaints work but with lower accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

KNOWLEDGE_PATH = Path(__file__).resolve().parents[2] / "configs" / "diagnosis_knowledge.yaml"

DTC_PRIOR = 0.8          # evidence weight of a component listed for a reported DTC
SUSPECT_MIN = 0.5        # combined probability to put a component on the inspection list
HAZARD_STOP = 0.7        # any single hazard above this -> stop driving
URGENT_STOP = 0.6        # probability of the "stop driving" urgency level
VAGUE_MIN = 0.6          # too-vague probability that triggers a follow-up question
SYSTEM_CONF_MIN = 0.35   # system choice confidence below this (and no DTC) -> ask for more information

URGENCY_LEVELS = [
    "The problem can wait for the next scheduled service; the car drives normally and no warning light is on",
    "The car still drives, but the fault should be inspected within a few days (for example a steady check-engine light, rough idle, a new noise)",
    "Driving should stop now because continuing risks engine damage or a safety problem (for example overheating with steam, a flashing check-engine light, brakes not working, fuel leaking, smoke)",
]

HAZARDS = {
    "overheating": "Does `complaint` say the engine is overheating right now, such as the temperature gauge in the red, steam from under the hood, or boiling coolant?",
    "brakes_unsafe": "Does `complaint` say the brakes are not working properly, such as the pedal sinking to the floor, a soft or spongy pedal, or the car not stopping well?",
    "fuel_leak": "Does `complaint` mention a strong raw fuel smell or fuel leaking from the car?",
    "smoke_fire": "Does `complaint` mention smoke, a burning smell, or fire coming from the engine bay?",
    "flashing_mil": "Does `complaint` say the check-engine light is flashing or blinking (not just staying on)?",
    "oil_pressure": "Does `complaint` say the red oil pressure warning light is on while the engine is running?",
}


@dataclass
class Suspect:
    component: str
    name: str
    detector_class: str | None
    probability: float
    jev: float
    from_dtc: list[str] = field(default_factory=list)


@dataclass
class TriageResult:
    action: str                       # stop_driving | ask_more_info | inspect
    system: str
    system_confidence: float
    urgency: float                    # 0 routine .. 2 stop driving
    hazards: dict[str, float]
    too_vague: float
    suspects: list[Suspect]
    detector_targets: list[str]       # L1 classes to locate, most likely first
    unknown_dtc: list[str]
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_knowledge(path: Path = KNOWLEDGE_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _code_value(code: str) -> tuple[str, int]:
    return code[0], int(code[1:], 16)


def lookup_dtc(codes: list[str], knowledge: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Exact/range lookup of DTC codes. Returns (matched entries with the code, unknown codes)."""
    matched, unknown = [], []
    for raw in codes:
        code = raw.strip().upper()
        if not code:
            continue
        hit = None
        for entry in knowledge["dtc"]:
            for spec in entry["codes"]:
                lo, _, hi = spec.partition("-")
                hi = hi or lo
                try:
                    (p, v), (plo, vlo), (phi, vhi) = _code_value(code), _code_value(lo), _code_value(hi)
                except ValueError:
                    continue
                if p == plo == phi and vlo <= v <= vhi:
                    hit = entry
                    break
            if hit:
                break
        if hit:
            matched.append({"code": code, "meaning": hit["meaning"], "components": hit["components"]})
        else:
            unknown.append(code)
    return matched, unknown


def build_state(complaint: str, dtc_matched: list[dict[str, Any]], unknown: list[str], vehicle: str | None) -> dict[str, Any]:
    dtc_codes = [{"code": d["code"], "meaning": d["meaning"]} for d in dtc_matched]
    dtc_codes += [{"code": c, "meaning": "not in the local code table (possibly manufacturer-specific)"} for c in unknown]
    return {"complaint": complaint or "(no description given)", "dtc_codes": dtc_codes, "vehicle": vehicle or "unknown"}


def build_questions(knowledge: dict[str, Any]) -> dict[str, Any]:
    from typesafe_sdk import Choice, Noul, NoulCriteria, Score

    q: dict[str, Any] = {
        "system": Choice(
            instructions="Which vehicle system is most likely at fault, based on `complaint` and `dtc_codes`?",
            criteria=dict(knowledge["systems"]),
        ),
        "urgency": Score(
            instructions="How urgently must the driver act on the problem described in `complaint` and `dtc_codes`?",
            criteria=URGENCY_LEVELS,
        ),
        "too_vague": Noul(
            instructions="Is `complaint` too vague to point to any specific vehicle system (for example only 'my car has a problem')?",
        ),
    }
    for hid, text in HAZARDS.items():
        q[f"hazard:{hid}"] = Noul(instructions=text)
    for cid, comp in knowledge["components"].items():
        q[f"component:{cid}"] = Noul(
            instructions=f"Is the {comp['name']} ({comp['description']}) a plausible cause of the problem in `complaint` and `dtc_codes`, so a technician should inspect it?",
            criteria=NoulCriteria(
                true="The described symptoms or codes are commonly caused by this component",
                false="The described symptoms and codes are not typically related to this component",
            ),
        )
    return q


def _prob_at(probabilities: dict[Any, float], level: int) -> float:
    return float(probabilities.get(level, probabilities.get(str(level), 0.0)))


def assemble(answers: dict[str, Any], knowledge: dict[str, Any], dtc_matched: list[dict[str, Any]],
             unknown: list[str], model: str | None = None) -> TriageResult:
    """Combine Jev answers with the DTC lookup and apply the action policy (pure code, testable offline)."""
    dtc_by_comp: dict[str, list[str]] = {}
    for d in dtc_matched:
        for c in d["components"]:
            dtc_by_comp.setdefault(c, []).append(d["code"])

    suspects = []
    for cid, comp in knowledge["components"].items():
        p_jev = float(answers[f"component:{cid}"].noul)
        codes = dtc_by_comp.get(cid, [])
        p = 1 - (1 - p_jev) * (1 - (DTC_PRIOR if codes else 0.0))  # noisy-OR of text and code evidence
        if p >= SUSPECT_MIN:
            suspects.append(Suspect(cid, comp["name"], comp.get("detector_class"), round(p, 3), round(p_jev, 3), codes))
    suspects.sort(key=lambda s: s.probability, reverse=True)

    system, urgency = answers["system"], answers["urgency"]
    hazards = {h: round(float(answers[f"hazard:{h}"].noul), 3) for h in HAZARDS}
    too_vague = float(answers["too_vague"].noul)

    if max(hazards.values()) >= HAZARD_STOP or _prob_at(urgency.probabilities, 2) >= URGENT_STOP:
        action = "stop_driving"
    elif not dtc_matched and (too_vague >= VAGUE_MIN or system.choice == "unclear"
                              or system.confidence < SYSTEM_CONF_MIN or not suspects):
        action = "ask_more_info"
    else:
        action = "inspect"

    targets = [s.detector_class for s in suspects if s.detector_class]
    return TriageResult(
        action=action, system=system.choice, system_confidence=round(float(system.confidence), 3),
        urgency=round(float(urgency.score), 3), hazards=hazards, too_vague=round(too_vague, 3),
        suspects=suspects, detector_targets=list(dict.fromkeys(targets)), unknown_dtc=unknown, model=model,
    )


def triage(complaint: str, dtc: list[str] | None = None, vehicle: str | None = None,
           client=None, knowledge: dict[str, Any] | None = None) -> TriageResult:
    """One Jev request per complaint. `client` is a TypeSafeClient (see jev.client.make_client)."""
    knowledge = knowledge or load_knowledge()
    matched, unknown = lookup_dtc(dtc or [], knowledge)
    if client is None:
        from .client import make_client

        with make_client() as c:
            resp = c.system_one(state=build_state(complaint, matched, unknown, vehicle), questions=build_questions(knowledge))
    else:
        resp = client.system_one(state=build_state(complaint, matched, unknown, vehicle), questions=build_questions(knowledge))
    return assemble(resp.answers, knowledge, matched, unknown, model=getattr(resp, "model", None))
