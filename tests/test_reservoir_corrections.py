from data_pipeline.apply_reservoir_corrections import correct_instance, correct_missing

ONTO = {"coolant_reservoir", "brake_fluid_reservoir", "washer_fluid_reservoir", "power_steering_reservoir", "radiator_cap"}


def _r(identity, label="coolant_reservoir", conf=0.9, policy="whole_body_ok", box=None):
    return {"identity": identity, "current_label": label, "confidence": conf, "box_policy": policy, "fixed_box_norm": box}


def test_rename_and_rebox():
    inst = {"id": 0, "decision": "correct"}
    assert correct_instance(inst, _r("brake_fluid_reservoir"), ONTO, 0.6) == "rename"
    assert inst["decision"] == "wrong_class" and inst["new_class"] == "brake_fluid_reservoir"
    inst = {"id": 1, "decision": "correct"}
    assert correct_instance(inst, _r("power_steering_reservoir", policy="needs_rebox", box=[0.1, 0.1, 0.2, 0.2]), ONTO, 0.6) == "rebox+rename"
    assert inst["decision"] == "bad_geometry" and inst["new_class"] == "power_steering_reservoir" and inst["fixed_box_norm"]


def test_uncertain_or_outside_ontology_is_dropped():
    for r in (_r("brake_fluid_reservoir", conf=0.45), _r("clutch_reservoir"), _r("other")):
        inst = {"id": 0, "decision": "correct"}
        assert correct_instance(inst, r, ONTO, 0.6) == "dropped"
        assert inst["decision"] == "not_a_component" and "new_class" not in inst
    # same identity at low confidence is kept
    inst = {"id": 0, "decision": "correct"}
    assert correct_instance(inst, _r("coolant_reservoir", conf=0.4), ONTO, 0.6) == "kept"


def test_missing_entries():
    m = {"class_name": "coolant_reservoir", "box_norm": [0, 0, 1, 1]}
    assert correct_missing(m, _r("washer_fluid_reservoir"), ONTO, 0.6) == "rename" and m["class_name"] == "washer_fluid_reservoir"
    assert correct_missing({"class_name": "coolant_reservoir"}, _r("other"), ONTO, 0.6) is None


def test_true_current_label_wins_over_batch_label():
    # earlier review already renamed this box (bad_geometry + new_class): identity equal to it is NOT a change
    inst = {"id": 0, "decision": "bad_geometry", "new_class": "brake_fluid_reservoir", "fixed_box_norm": [0, 0, 1, 1]}
    r = _r("brake_fluid_reservoir", label="coolant_reservoir")
    assert correct_instance(inst, r, ONTO, 0.6, current="brake_fluid_reservoir") == "kept"
    assert inst["decision"] == "bad_geometry" and inst["new_class"] == "brake_fluid_reservoir"
