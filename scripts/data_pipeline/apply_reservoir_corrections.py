"""Turn pixel re-review verdicts of reservoir boxes into corrected COPIES of the original review verdicts.

Input : data/engine_bay_review_reservoir/verdicts/*.json  (lists of {key, current_label, identity, box_policy,
        fixed_box_norm, confidence, cues}; key = "<review_root>/<split>/<stem>#<instance id | missingK>")
Output: data/<review_root>_rr/verdicts/<split>/<stem>.json  every verdict of that root, corrections applied
        data/<review_root>_rr/packets/<split>/<stem>.json   packet JSONs copied (apply_review reads them)
        data/engine_bay_review_reservoir/CHANGES.json       what changed, per key
The originals are never modified. Rebuild the reviewed labels with (DGX, SAM2 for re-segmented boxes):
    python scripts/data_pipeline/apply_review.py --review data/<root>_rr --seg <same seg dir as before> --out data/<reviewed>_rr

Rules (docs/plans/label_policy_reservoirs_heat_shield.md):
  confidence >= --min-conf and identity in the 36-class ontology -> rename (wrong_class / bad_geometry + new_class)
  needs_rebox                                                    -> bad_geometry with fixed_box_norm
  identity other / not_a_component, or a name outside the ontology (clutch, inverter coolant)
                                                                 -> not_a_component (dropped from training)
  identity differs but confidence < --min-conf (two reviewers disagree) -> not_a_component (dropped)
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_corrections(paths):
    out = {}
    for p in paths:
        for r in json.loads(Path(p).read_text(encoding="utf-8")):
            out[r["key"]] = r  # later rounds override earlier ones
    return out


def correct_instance(inst: dict, r: dict, onto: set, min_conf: float, current: str | None = None) -> str:
    """Mutate one verdict instance; return the action taken. `current` = the label the verdict really produces
    (new_class when the earlier reviewer set one, also with bad_geometry; else the packet class)."""
    ident, conf = r["identity"], float(r.get("confidence") or 0)
    changed = ident != (current or r["current_label"])
    rebox = r.get("box_policy") == "needs_rebox" and r.get("fixed_box_norm")
    if ident in ("other", "not_a_component") or (changed and ident not in onto) or (changed and conf < min_conf):
        inst.update(decision="not_a_component", note=f"[rr] dropped: {ident} ({conf:.2f}) {r.get('cues', '')}"[:300])
        inst.pop("new_class", None)
        return "dropped"
    if rebox:
        inst.update(decision="bad_geometry", fixed_box_norm=[round(float(v), 4) for v in r["fixed_box_norm"]])
        if changed:
            inst["new_class"] = ident
        inst["note"] = f"[rr] rebox{' + rename to ' + ident if changed else ''}"
        return "rebox+rename" if changed else "rebox"
    if changed:
        if inst.get("decision") == "bad_geometry":
            inst["new_class"] = ident
        else:
            inst.update(decision="wrong_class", new_class=ident)
        inst["note"] = f"[rr] rename to {ident}"
        return "rename"
    return "kept"


def correct_missing(m: dict, r: dict, onto: set, min_conf: float) -> str | None:
    """Mutate a reviewer-added (missing) entry; None means remove it."""
    ident, conf = r["identity"], float(r.get("confidence") or 0)
    changed = ident != r["current_label"]
    if ident in ("other", "not_a_component") or (changed and ident not in onto) or (changed and conf < min_conf):
        return None
    if changed:
        m["class_name"] = ident
    if r.get("box_policy") == "needs_rebox" and r.get("fixed_box_norm"):
        m["box_norm"] = [round(float(v), 4) for v in r["fixed_box_norm"]]
    return "rename" if changed else ("rebox" if r.get("box_policy") == "needs_rebox" else "kept")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", nargs="+", type=Path,
                    default=sorted((ROOT / "data" / "engine_bay_review_reservoir" / "verdicts").glob("*.json")))
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "data_engine_bay.yaml")
    args = ap.parse_args()

    onto = set(yaml.safe_load(args.config.read_text(encoding="utf-8"))["names"].values())
    corr = load_corrections(args.verdicts)
    by_root = {}
    for key in corr:
        by_root.setdefault(key.split("/")[0], []).append(key)
    changes, actions = [], Counter()
    for root, keys in by_root.items():
        src, dst = ROOT / "data" / root, ROOT / "data" / f"{root}_rr"
        for sub in ("verdicts", "packets"):
            for f in (src / sub).glob("*/*.json"):
                t = dst / sub / f.parent.name / f.name
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
        by_image = {}
        for key in keys:
            path, inst_id = key.split("#")
            by_image.setdefault(path, []).append((inst_id, key))
        for path, items in by_image.items():
            _, split, stem = path.split("/")
            vp = dst / "verdicts" / split / f"{stem}.json"
            v = json.loads(vp.read_text(encoding="utf-8"))
            for inst_id, key in items:  # indices refer to the ORIGINAL missing list: remove only at the end
                r = corr[key]
                if inst_id.startswith("missing"):
                    k = int(inst_id[len("missing"):])
                    r = {**r, "current_label": v["missing"][k]["class_name"]}
                    res = correct_missing(v["missing"][k], r, onto, args.min_conf)
                    if res is None:
                        v["missing"][k]["_remove"] = True
                        res = "removed_missing"
                else:
                    inst = next(i for i in v["instances"] if i["id"] == int(inst_id))
                    packet = json.loads((dst / "packets" / split / f"{stem}.json").read_text(encoding="utf-8"))
                    pcls = next(i["class_name"] for i in packet["instances"] if i["id"] == int(inst_id))
                    current = inst.get("new_class") or pcls
                    r = {**r, "current_label": current}
                    res = correct_instance(inst, r, onto, args.min_conf, current)
                actions[res] += 1
                if res != "kept":
                    changes.append({"key": key, "action": res, "from": r["current_label"], "to": r["identity"],
                                    "confidence": r.get("confidence")})
            v["missing"] = [m for m in v.get("missing", []) if not m.get("_remove")]
            vp.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    out = ROOT / "data" / "engine_bay_review_reservoir" / "CHANGES.json"
    out.write_text(json.dumps({"actions": dict(actions), "changes": changes}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"roots": {r: len(k) for r, k in by_root.items()}, "actions": dict(actions)}, indent=1))


if __name__ == "__main__":
    main()
