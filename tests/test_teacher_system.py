from inference.teacher_system import TeacherSystem


class _Priors:
    data = {"classes": {"battery_terminal": {"count_max": 1}, "ignition_coil": {"count_max": 4}}}


def _system(agnostic=0.85, priors=True):
    ts = TeacherSystem.__new__(TeacherSystem)  # no model needed for post-processing
    ts.conf, ts.iou, ts.class_conf, ts.cap_extra = 0.25, 0.5, {}, 1
    ts.agnostic_iou = agnostic
    ts.priors = _Priors() if priors else None
    return ts


def _det(cls, score, box):
    return {"cls": cls, "score": score, "box": box, "poly": None}


def test_same_class_duplicates_are_merged():
    ts = _system()
    dets = [_det("battery_terminal", 0.75, [0.28, 0.24, 0.36, 0.32]),
            _det("battery_terminal", 0.74, [0.281, 0.241, 0.361, 0.321]),
            _det("engine_cover", 0.9, [0.1, 0.1, 0.8, 0.6])]
    out = ts.postprocess(dets, use_cap=False, use_agnostic=False)
    assert sorted(d["cls"] for d in out) == ["battery_terminal", "engine_cover"]
    assert max(d["score"] for d in out if d["cls"] == "battery_terminal") == 0.75


def test_agnostic_merge_keeps_stronger_class():
    ts = _system()
    dets = [_det("brake_fluid_reservoir", 0.89, [0.05, 0.33, 0.125, 0.52]),
            _det("washer_fluid_reservoir", 0.63, [0.051, 0.33, 0.125, 0.523])]
    assert [d["cls"] for d in ts.postprocess(dets, use_cap=False, use_agnostic=True)] == ["brake_fluid_reservoir"]
    assert len(ts.postprocess(dets, use_cap=False, use_agnostic=False)) == 2


def test_count_cap_and_threshold():
    ts = _system()
    coils = [_det("ignition_coil", 0.9 - i * 0.05, [0.1 * i, 0.1, 0.1 * i + 0.05, 0.2]) for i in range(8)]
    out = ts.postprocess(coils + [_det("battery_terminal", 0.2, [0.5, 0.5, 0.6, 0.6])], use_cap=True, use_agnostic=True)
    assert sum(d["cls"] == "ignition_coil" for d in out) == 5  # count_max 4 + cap_extra 1
    assert all(d["cls"] != "battery_terminal" for d in out)  # below conf 0.25


def test_run_moves_tensors_to_cpu():
    # regression: on GPU the Ultralytics result tensors live on cuda; _run must call .cpu() before .numpy()
    import inspect
    from inference.teacher_system import TeacherSystem
    src = inspect.getsource(TeacherSystem._run)
    assert ".xyxy.cpu().numpy()" in src and ".conf.cpu().numpy()" in src and ".cls.cpu().numpy()" in src


def test_open_upright_applies_exif_orientation(tmp_path):
    from PIL import Image
    from inference.teacher_system import open_upright
    im = Image.new("RGB", (60, 20), (255, 0, 0))
    im.paste((0, 0, 255), (0, 0, 10, 20))  # blue strip on the left
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 CW when displayed
    p = tmp_path / "rot.jpg"
    im.save(p, exif=exif)
    up = open_upright(p)
    assert up.size == (20, 60)  # portrait after applying the tag
    assert up.getpixel((10, 2))[2] > 200  # the left strip is now at the top
