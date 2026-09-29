"""Vehicle-level train / val / test split of a build_full_dataset.py pool (goal B protocol).

The v1 reference (v6..v8, p5 teachers) keeps 6 val vehicles and 3 test vehicles out of training; a v2
(taxonomy) model must be judged on exactly the same vehicles to compare with teacher p5_reg (test mask
mAP50-95 0.354). EXT__* images always go to train. Train uses the same repeat-factor sampling as the pool.

    python scripts/data_pipeline/vehicle_split.py --pool data/engine_bay_full_v2
    # -> <pool>/vehicle_{train,val,test}.txt and <pool>/data_vehicle.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.build_full_dataset import repeat_list  # noqa: E402

VAL_VEHICLES = ["Request_ID_08", "Request_ID_36", "Request_ID_49", "Request_ID_50", "Request_ID_52", "Request_ID_58"]
TEST_VEHICLES = ["Request_ID_13", "Request_ID_23", "Request_ID_29"]


def split_stems(stems, val_vehicles, test_vehicles):
    out = {"train": [], "val": [], "test": []}
    for s in stems:
        v = s.split("__")[0]
        out["test" if v in test_vehicles else "val" if v in val_vehicles else "train"].append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True, help="build_full_dataset.py output (images/all, labels/all, data_full.yaml)")
    ap.add_argument("--val-vehicles", nargs="+", default=VAL_VEHICLES)
    ap.add_argument("--test-vehicles", nargs="+", default=TEST_VEHICLES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rfs-t", type=float, default=0.10)
    ap.add_argument("--max-repeat", type=int, default=4)
    a = ap.parse_args()

    pool = a.pool.resolve()
    stems = sorted(p.stem for p in (pool / "labels" / "all").glob("*.txt"))
    parts = split_stems(stems, set(a.val_vehicles), set(a.test_vehicles))
    labels = {s: (pool / "labels" / "all" / f"{s}.txt").read_text().splitlines() for s in stems}
    for name, ss in parts.items():
        lines = (repeat_list(ss, labels, random.Random(a.seed), a.rfs_t, a.max_repeat) if name == "train"
                 else [f"./images/all/{s}.jpg" for s in ss])
        (pool / f"vehicle_{name}.txt").write_text("\n".join(lines) + "\n")
    full = yaml.safe_load((pool / "data_full.yaml").read_text(encoding="utf-8"))
    cfg = {"path": str(pool), "nc": full["nc"], "names": full["names"],
           "train": "vehicle_train.txt", "val": "vehicle_val.txt", "test": "vehicle_test.txt"}
    (pool / "data_vehicle.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    summary = {k: {"images": len(v), "vehicles": len({s.split('__')[0] for s in v}),
                   "instances": sum(len([l for l in labels[s] if l.strip()]) for s in v)} for k, v in parts.items()}
    missing = [v for v in a.val_vehicles + a.test_vehicles if not any(s.startswith(v + "__") for s in stems)]
    summary["missing_heldout_vehicles"] = missing
    (pool / "VEHICLE_SPLIT.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
