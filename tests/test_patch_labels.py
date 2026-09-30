from data_pipeline.patch_labels import apply

NAMES = {"battery": 0, "battery_terminal": 1, "coolant_reservoir": 3, "brake_fluid_reservoir": 5}


def _line(cid, box):
    x1, y1, x2, y2 = box
    return f"{cid} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"


def test_replace_matched_instance_and_keep_others():
    lines = [_line(0, (0.1, 0.1, 0.5, 0.5)), _line(3, (0.6, 0.6, 0.8, 0.8))]
    patch = {"match": {"class": "coolant_reservoir", "box": [0.6, 0.6, 0.8, 0.81]},
             "replace": [{"class": "brake_fluid_reservoir", "polygon": [[0.6, 0.6], [0.8, 0.6], [0.8, 0.8]]}]}
    out, st = apply(lines, patch, NAMES, lambda n: n, 0.8)
    assert st == "replaced" and out[0] == lines[0] and out[1].startswith("5 ")


def test_unmatched_and_removal():
    lines = [_line(0, (0.1, 0.1, 0.5, 0.5))]
    far = {"match": {"class": "battery", "box": [0.6, 0.6, 0.9, 0.9]}, "replace": []}
    assert apply(lines, far, NAMES, lambda n: n, 0.8) == (lines, "unmatched")
    near = {"match": {"class": "battery", "box": [0.1, 0.1, 0.5, 0.5]}, "replace": []}
    assert apply(lines, near, NAMES, lambda n: n, 0.8) == ([], "removed")
    # a class the build does not train (mapper -> None) is reported, not guessed
    assert apply(lines, near, NAMES, lambda n: None, 0.8)[1] == "class_not_in_build"
