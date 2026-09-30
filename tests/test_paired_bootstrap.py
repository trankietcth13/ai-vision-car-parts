import argparse

import numpy as np

from evaluation.paired_bootstrap import compare


def _save(path, names, per_image, qa=0.0):
    """per_image: list of (im_name, [(cls, conf, is_tp)], [target cls])"""
    arr = {"im_names": np.array([p[0] for p in per_image]), "names": np.array(names), "qa_mask_map50_95": np.array(qa)}
    cols = {"tp_m": [], "tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    for _, preds, targets in per_image:
        tp = np.array([[t] * 10 for _, _, t in preds], bool).reshape(-1, 10)
        cols["tp_m"].append(tp)
        cols["tp"].append(tp)
        cols["conf"].append(np.array([c for _, c, _ in preds], float))
        cols["pred_cls"].append(np.array([k for k, _, _ in preds], float))
        cols["target_cls"].append(np.array(targets, float))
    for k, v in cols.items():
        arr[f"{k}_len"] = np.array([len(x) for x in v])
        arr[k] = np.concatenate(v, 0)
    np.savez_compressed(path, **arr)


def _args(tmp_path, pair, **kw):
    d = dict(stats=str(tmp_path), file=None, group=None, pair=[pair], classes=None, n_boot=200, margin=0.0, seed=0,
             out=str(tmp_path / "r.md"))
    d.update(kw)
    return argparse.Namespace(**d)


def test_better_model_detected_and_classes_matched_by_name(tmp_path):
    imgs = [f"im{i}.jpg" for i in range(40)]
    # model A finds every object; model B misses half. B has a different class order and an extra class.
    _save(tmp_path / "A.npz", ["battery", "radiator_cap"], [(im, [(0, 0.9, True), (1, 0.8, True)], [0, 1]) for im in imgs])
    _save(tmp_path / "B.npz", ["radiator_cap", "other_reservoir", "battery"],
          [(im, [(2, 0.9, True), (0, 0.8, i % 2 == 0)], [2, 0]) for i, im in enumerate(imgs)])
    compare(_args(tmp_path, "A:B"))
    import json
    r = json.loads((tmp_path / "r.json").read_text())
    assert r["classes"] == ["battery", "radiator_cap"] and r["gt_diff"] == {"B": 0}
    assert r["pairs"][0]["verdict"] == "better" and r["pairs"][0]["ci"][0] > 0


def test_identical_models_no_difference(tmp_path):
    imgs = [(f"im{i}.jpg", [(0, 0.9, i % 3 != 0)], [0]) for i in range(30)]
    _save(tmp_path / "A.npz", ["battery"], imgs)
    _save(tmp_path / "B.npz", ["battery"], imgs)
    compare(_args(tmp_path, "A:B"))
    import json
    p = json.loads((tmp_path / "r.json").read_text())["pairs"][0]
    assert p["diff"] == 0 and p["verdict"] == "no detectable difference"
