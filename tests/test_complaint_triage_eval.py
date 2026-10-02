"""The English triage eval set must stay consistent with configs/diagnosis_knowledge.yaml (offline, no model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "diagnosis"))

from eval_complaint_triage import load_cases, score, validate  # noqa: E402
from jev.diagnosis_triage import load_knowledge  # noqa: E402


def test_eval_set_matches_knowledge_layer():
    assert validate(load_cases(), load_knowledge()) == []


def test_score_accepts_alternatives_and_counts_stop_recall():
    cases = [c for c in load_cases() if c["also_ok"] and c["action"] == "inspect"][:1]
    cases += [c for c in load_cases() if c["action"] == "stop_driving"][:1]
    preds = [{"id": cases[0]["id"], "system": cases[0]["also_ok"][0], "confidence": 0.9, "action": "inspect"},
             {"id": cases[1]["id"], "system": cases[1]["system"], "confidence": 0.9, "action": "inspect"}]
    report = score(cases, preds)
    assert "| system accuracy (also_ok accepted) | 2/2" in report
    assert "| system accuracy (strict) | 1/2" in report
    assert "| stop_driving recall | 0/1 |" in report
