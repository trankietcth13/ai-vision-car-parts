"""Instance / vehicle counts per taxonomy-v2 component and training class, with recommended tiers.

Reads YOLO-seg label folders whose class names come from their own data yaml (e.g. the 36-class
reviewed sets), maps every instance through configs/taxonomy_v2.yaml and reports:
  * per fine component: instances, vehicles, declared tier, recommended tier (promotion thresholds)
  * per training class: instances and vehicles (what a v2 build would train on)
  * per system: instances
Vehicle id = stem prefix before the first "__" (Request_ID_xx or EXT source). An image that appears in
several sources is counted once (first source wins).

    python scripts/data_pipeline/taxonomy_stats.py \
        --src data/engine_bay_reviewed data/engine_bay_reviewed_hybrid --out docs/reports/taxonomy_v2_stats.md
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.taxonomy import load_taxonomy  # noqa: E402


def read_names(src: Path) -> dict:
    ymls = sorted(src.glob("*.yaml"))
    if not ymls:
        raise FileNotFoundError(f"no data yaml in {src}")
    names = yaml.safe_load(ymls[0].read_text(encoding="utf-8"))["names"]
    return {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}


def collect(srcs, splits):
    seen = set()
    rows = []  # (ann_name, vehicle, split)
    for src in srcs:
        names = read_names(src)
        for split in splits:
            for lp in sorted((src / "labels" / split).glob("*.txt")):
                if lp.stem in seen:
                    continue
                seen.add(lp.stem)
                vehicle = lp.stem.split("__")[0]
                for line in lp.read_text().splitlines():
                    if line.strip():
                        rows.append((names[int(line.split()[0])], vehicle, split))
    return rows, len(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--taxonomy", default=None)
    ap.add_argument("--out", type=Path, default=None, help="markdown report (json next to it)")
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy) if args.taxonomy else load_taxonomy()
    rows, n_images = collect(args.src, args.splits)
    min_n, min_v = tax.promotion.get("min_instances", 60), tax.promotion.get("min_vehicles", 8)
    keep_n = tax.promotion.get("keep_min_instances", 40)

    comp_n, comp_v = collections.Counter(), collections.defaultdict(set)
    train_n, train_v = collections.Counter(), collections.defaultdict(set)
    sys_n, unknown = collections.Counter(), collections.Counter()
    for ann, veh, _ in rows:
        comp = tax.component_for(ann)
        if comp is None:
            unknown[ann] += 1
            continue
        comp_n[comp.name] += 1
        comp_v[comp.name].add(veh)
        sys_n[comp.system] += 1
        tc = tax.train_class_for(ann)
        if tc:
            train_n[tc] += 1
            train_v[tc].add(veh)

    lines = [f"# Taxonomy v2 statistics", "",
             f"Sources: {', '.join(str(s) for s in args.src)} · splits {', '.join(args.splits)} · "
             f"{n_images} images · {len(rows)} instances · promotion ≥ {min_n} instances from ≥ {min_v} vehicles, "
             f"tier A kept while ≥ {keep_n} instances", ""]
    lines += ["## Components", "", "| system | component | instances | vehicles | declared | recommended | trained as |",
              "|---|---|---|---|---|---|---|"]
    changes = []
    for sysname, comps in tax.by_system().items():
        for name in comps:
            c = tax.components[name]
            n, v = comp_n[name], len(comp_v[name])
            promote = n >= min_n and v >= min_v
            keep = c.tier == "A" and n >= keep_n
            rec = "A" if (promote or keep) else "B"
            if rec != c.tier:
                changes.append((name, c.tier, rec, n, v))
            flag = " ⚠" if rec != c.tier else ""
            lines.append(f"| {sysname} | {name} | {n} | {v} | {c.tier} | {rec}{flag} | {tax.train_class_for(name) or '—'} |")
    lines += ["", "## Training classes (v2 build)", "", "| id | class | instances | vehicles |", "|---|---|---|---|"]
    for i, n in enumerate(tax.training_names):
        lines.append(f"| {i} | {n} | {train_n[n]} | {len(train_v[n])} |")
    lines += ["", "## Systems", "", "| system | instances |", "|---|---|"]
    lines += [f"| {s} | {sys_n[s]} |" for s in tax.systems]
    if changes:
        lines += ["", "## Tier mismatches (declared vs recommended)", ""]
        lines += [f"- {n}: declared {d}, recommended {r} ({k} instances, {v} vehicles)" for n, d, r, k, v in changes]
    if unknown:
        lines += ["", "## Unknown annotation names", "", ", ".join(f"{k} ({v})" for k, v in unknown.most_common())]
    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps({
            "images": n_images, "instances": len(rows),
            "components": {k: {"instances": comp_n[k], "vehicles": len(comp_v[k])} for k in tax.components},
            "training_classes": {k: {"instances": train_n[k], "vehicles": len(train_v[k])} for k in tax.training_names},
            "tier_changes": [dict(zip(("component", "declared", "recommended", "instances", "vehicles"), c)) for c in changes],
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
