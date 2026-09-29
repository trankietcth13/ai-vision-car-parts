import copy

import pytest
import yaml

from data_pipeline.taxonomy import DEFAULT_PATH, Taxonomy, load_taxonomy

ANNOTATION_36 = [
    "battery", "battery_terminal", "fuse_relay_box", "coolant_reservoir", "radiator_cap",
    "brake_fluid_reservoir", "washer_fluid_reservoir", "engine_cover", "oil_filler_cap", "oil_dipstick",
    "air_filter_box", "air_intake_duct", "maf_sensor", "throttle_body", "alternator", "ignition_coil",
    "radiator_hose_upper", "serpentine_belt", "ecu_module", "multimeter_diagnostic_tool", "radiator",
    "radiator_cooling_fan", "radiator_hose_lower", "intake_manifold", "abs_modulator_unit", "brake_booster",
    "power_steering_reservoir", "transmission_oil_dipstick", "ac_compressor", "exhaust_manifold_heat_shield",
    "turbocharger", "intercooler_piping", "strut_tower_brace", "hood_latch_mechanism",
    "windshield_wiper_motor", "oil_filter",
]


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def test_every_annotation_class_is_known(tax):
    unknown = [n for n in ANNOTATION_36 if tax.component_for(n) is None]
    assert unknown == []


def test_current_training_classes_stay_trained(tax):
    # the 20 classes trained since v6 must keep their own class in v2
    current = ["battery", "battery_terminal", "fuse_relay_box", "coolant_reservoir", "radiator_cap",
               "brake_fluid_reservoir", "washer_fluid_reservoir", "engine_cover", "oil_filler_cap",
               "oil_dipstick", "air_filter_box", "air_intake_duct", "maf_sensor", "throttle_body",
               "alternator", "ignition_coil", "radiator_hose", "ecu_module", "multimeter_diagnostic_tool",
               "intake_manifold"]
    assert all(tax.train_class_for(n) == n for n in current)


def test_mapping_rules(tax):
    assert tax.train_class_for("radiator_hose_upper") == "radiator_hose"
    assert tax.train_class_for("radiator_hose_lower") == "radiator_hose"
    assert tax.train_class_for("exhaust_manifold_heat_shield") == "exhaust_manifold_heat_shield"
    assert tax.train_class_for("power_steering_reservoir") == "other_reservoir"
    assert tax.train_class_for("serpentine_belt") is None  # tier B without fallback -> ignored
    assert tax.train_class_for("not_a_class") is None
    assert tax.system_of("maf_sensor") == "air_intake"


def test_training_ids_are_contiguous(tax):
    names = tax.training_names
    assert len(names) == len(set(names))
    assert [tax.train_id[n] for n in names] == list(range(len(names)))
    # tier A first, generic classes last
    first_generic = min(i for i, n in enumerate(names) if n in tax.generic)
    assert all(n in tax.generic for n in names[first_generic:])


def test_validation_rejects_bad_config():
    cfg = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    bad = copy.deepcopy(cfg)
    bad["components"]["battery"]["fallback"] = "other_nonexistent"
    with pytest.raises(ValueError):
        Taxonomy(bad)
    bad = copy.deepcopy(cfg)
    bad["components"]["battery"]["system"] = "spaceship"
    with pytest.raises(ValueError):
        Taxonomy(bad)


def test_build_full_dataset_remaps(tax):
    from data_pipeline.build_full_dataset import ROOT, build_remaps

    ann = {int(k): v for k, v in yaml.safe_load((ROOT / "configs" / "data_engine_bay.yaml").read_text(encoding="utf-8"))["names"].items()}
    tc = yaml.safe_load((ROOT / "configs" / "engine_bay_train_classes.yaml").read_text(encoding="utf-8"))
    base = {int(k): v for k, v in tc["names"].items()}

    # v1 (no taxonomy): identity on base ids except oil_filter, 36-class ids through the v1 map
    names, ann_remap, train_remap = build_remaps(base, ann, tc)
    oil = next(k for k, v in base.items() if v == "oil_filter")
    assert names == base and train_remap[oil] is None
    assert all(train_remap[k] == k for k in base if k != oil)
    inv_ann = {v: k for k, v in ann.items()}
    assert names[ann_remap[inv_ann["radiator_hose_lower"]]] == "radiator_hose"

    # v2: everything by name through the taxonomy
    names2, ann2, train2 = build_remaps(base, ann, tc, DEFAULT_PATH)
    assert list(names2.values()) == tax.training_names
    assert names2[ann2[inv_ann["exhaust_manifold_heat_shield"]]] == "exhaust_manifold_heat_shield"
    assert names2[ann2[inv_ann["power_steering_reservoir"]]] == "other_reservoir"
    assert ann2[inv_ann["serpentine_belt"]] is None
    assert train2[oil] is None
    for k, n in base.items():
        if n != "oil_filter":
            assert names2[train2[k]] == n  # every v1 training class survives under its own name
