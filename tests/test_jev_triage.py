"""Offline tests for the Jev diagnosis triage (no network: answers are faked)."""
from types import SimpleNamespace

import pytest

pytest.importorskip("typesafe_sdk")

from jev.client import resolve_api_key
from jev.diagnosis_triage import HAZARDS, assemble, build_questions, build_state, load_knowledge, lookup_dtc

KNOW = load_knowledge()


def fake_answers(components=None, hazards=None, system="cooling", conf=0.9, urgency_probs=(0.1, 0.8, 0.1), vague=0.05):
    components = components or {}
    hazards = hazards or {}
    a = {f"component:{c}": SimpleNamespace(noul=components.get(c, 0.05)) for c in KNOW["components"]}
    a.update({f"hazard:{h}": SimpleNamespace(noul=hazards.get(h, 0.02)) for h in HAZARDS})
    a["system"] = SimpleNamespace(choice=system, confidence=conf)
    probs = dict(enumerate(urgency_probs))
    a["urgency"] = SimpleNamespace(score=sum(k * v for k, v in probs.items()), probabilities=probs)
    a["too_vague"] = SimpleNamespace(noul=vague)
    return a


def test_dtc_lookup_ranges_and_unknown():
    matched, unknown = lookup_dtc(["p0302", "P0171", "P1234", "U0100"], KNOW)
    assert [m["code"] for m in matched] == ["P0302", "P0171", "U0100"]
    assert "ignition_coil" in matched[0]["components"]
    assert unknown == ["P1234"]


def test_every_dtc_component_and_class_exists():
    classes = set(__import__("yaml").safe_load(open("configs/engine_bay_train_classes.yaml", encoding="utf-8"))["names"].values())
    for entry in KNOW["dtc"]:
        for c in entry["components"]:
            assert c in KNOW["components"], c
    for comp in KNOW["components"].values():
        assert comp["detector_class"] is None or comp["detector_class"] in classes


def test_questions_cover_components_and_hazards():
    q = build_questions(KNOW)
    assert {"system", "urgency", "too_vague"} <= set(q)
    assert sum(k.startswith("component:") for k in q) == len(KNOW["components"])
    assert sum(k.startswith("hazard:") for k in q) == len(HAZARDS)


def test_state_marks_unknown_codes():
    m, u = lookup_dtc(["P0302", "P1234"], KNOW)
    s = build_state("shakes", m, u, None)
    assert s["dtc_codes"][1]["code"] == "P1234" and "not in the local code table" in s["dtc_codes"][1]["meaning"]


def test_dtc_evidence_puts_component_on_list_and_orders_targets():
    m, u = lookup_dtc(["P0302"], KNOW)
    r = assemble(fake_answers(components={"ignition_coil": 0.7, "spark_plug": 0.3}, system="ignition_misfire"), KNOW, m, u)
    assert r.action == "inspect"
    assert r.suspects[0].component == "ignition_coil" and r.suspects[0].from_dtc == ["P0302"]
    assert r.suspects[0].probability == pytest.approx(1 - 0.3 * 0.2)  # noisy-OR of text 0.7 and code 0.8
    assert r.detector_targets[0] == "ignition_coil"
    assert "spark_plug" in [s.component for s in r.suspects]  # not detectable, still listed
    assert None not in r.detector_targets


def test_hazard_forces_stop_driving():
    r = assemble(fake_answers(components={"coolant_reservoir": 0.9}, hazards={"overheating": 0.95}), KNOW, [], [])
    assert r.action == "stop_driving"


def test_vague_complaint_asks_for_more_info():
    r = assemble(fake_answers(system="unclear", conf=0.3, vague=0.9), KNOW, [], [])
    assert r.action == "ask_more_info"


def test_resolve_key_from_alternate_env_name(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nAI_Vision_TypeSafe_API_Key=abc\n", encoding="utf-8")
    assert resolve_api_key(env) == "abc"
    env.write_text("OTHER=1\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        resolve_api_key(env)
