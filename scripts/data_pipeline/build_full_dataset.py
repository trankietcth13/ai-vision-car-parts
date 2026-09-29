"""
Phase 3 of the engine-bay roadmap (goal A: detect accurately on the images of `dataset/`).

Builds ONE pool with every reviewed image of all 28 vehicles (train + val + test vehicles) and
5 image-level cross-validation folds over it, plus a config that trains on everything.

Label sources (training ontology, configs/engine_bay_train_classes.yaml), highest precedence first:
    --reviewed DIR:SPLIT   apply_review.py outputs (36-class ids, remapped), in the order given; default
                           data/engine_bay_reviewed_p2:p2 then data/engine_bay_reviewed_hybrid:train. These also
                           cover reviewed images of val/test vehicles that v8/v9 keep out of train.
    --base     latest train dataset (v8/v9): train/val/test labels, used only for stems whose status is
               "reviewed" in --candidates (expert / hybrid / Codex); pseudo-labelled stems are skipped
Image lookup: the source folder, then the base dataset, then the original under dataset/ (CANDIDATES path).
Images larger than --max-side are resized (labels are normalised, so they stay valid).
Images whose reviewer set exclude_from_training (EXCLUDED.txt) are dropped. EXT__* images of the base
train list are added to the train side of every fold (never validated on), unless --no-ext.

Folds: stems are shuffled per vehicle (seed) and dealt round-robin, so every fold holds ~1/5 of every
vehicle. This is deliberately an IMAGE split (goal A = the same vehicles); it says nothing about new
vehicles (goal B keeps the vehicle-level test split of v8/v9).

Train lists use LVIS repeat-factor sampling (same rule as build_v6.py, computed per fold on its train part).

Output (--out, default data/engine_bay_full):
    images/all, labels/all                 hard links / copies
    fold{k}_train.txt, fold{k}_val.txt     k = 0..4
    data_fold{k}.yaml                      train = fold{k}_train.txt, val = fold{k}_val.txt
    data_full.yaml                         train = full_train.txt (all), val = fold0_val.txt (NOT held out;
                                           only for monitoring), extra split `all: images/all` for test_cases.py
    FULL_SUMMARY.json

Usage (DGX, after the Phase 2 labels are applied):
    .venv/bin/python scripts/data_pipeline/build_full_dataset.py --base data/engine_bay_train_v9 \
        --reviewed data/engine_bay_reviewed_p2:p2 data/engine_bay_reviewed_hybrid:train --candidates data/engine_bay_phase2/CANDIDATES.jsonl --out data/engine_bay_full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link(src: Path, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def find_image(folder: Path, stem: str) -> Path | None:
    for suf in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = folder / f"{stem}{suf}"
        if p.exists():
            return p
    return None


def repeat_list(stems: list[str], labels: dict[str, list[str]], rng: random.Random, t: float, max_repeat: int):
    img_classes = {s: {int(l.split()[0]) for l in labels[s]} for s in stems}
    n = max(1, len(stems))
    freq = Counter(c for cs in img_classes.values() for c in cs)
    rc = {c: max(1.0, math.sqrt(t / (freq[c] / n))) for c in freq}
    out = []
    for s in stems:
        r = max([rc[c] for c in img_classes[s]] + [1.0])
        k = min(max_repeat, int(math.floor(r)) + (1 if rng.random() < r - math.floor(r) else 0))
        out += [f"./images/all/{s}.jpg"] * max(1, k)
    return out


def build_remaps(base_names: dict[int, str], ann: dict[int, str], tc: dict, taxonomy_path: Path | None = None):
    """Class-id maps for the pool.

    Returns (out_names, ann_remap, train_remap):
        out_names    id -> name of the output ontology
        ann_remap    36-class annotation id -> output id (None = drop)
        train_remap  base training id (v1 ontology of --base) -> output id (None = drop)
    Without a taxonomy the output ontology is the v1 training ontology (oil_filter dropped, as since v6).
    With configs/taxonomy_v2.yaml every name goes through Taxonomy.train_id_for (tier A itself, tier B its
    generic fallback, unknown/ignored -> None).
    """
    if taxonomy_path is not None:
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from data_pipeline.taxonomy import load_taxonomy
        tax = load_taxonomy(taxonomy_path)
        ann_remap = {k: tax.train_id_for(n) for k, n in ann.items()}
        train_remap = {k: tax.train_id_for(n) for k, n in base_names.items()}
        return tax.yolo_names(), ann_remap, train_remap
    t_id = {v: int(k) for k, v in tc["names"].items()}
    oil = t_id.get("oil_filter")
    ann_remap = {k: (t_id[tc["map"][n]] if tc["map"].get(n) else None) for k, n in ann.items()}
    ann_remap = {k: (None if v == oil else v) for k, v in ann_remap.items()}
    train_remap = {k: (None if k == oil else k) for k in base_names}
    return dict(base_names), ann_remap, train_remap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="v8/v9 train dataset (training class ids)")
    ap.add_argument("--reviewed", nargs="*", default=["data/engine_bay_reviewed_p2:p2", "data/engine_bay_reviewed_hybrid:train"],
                    help="DIR:SPLIT apply_review outputs with 36-class ids, highest precedence first (missing dirs are skipped)")
    ap.add_argument("--reviewed-train", nargs="*", default=[],
                    help="DIR:SPLIT label folders already in TRAINING ids (e.g. data/engine_bay_p2_verified:train, the "
                         "verifier-corrected Phase 2 labels); highest precedence of all sources")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--candidates", type=Path, default=ROOT / "data" / "engine_bay_phase2" / "CANDIDATES.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "engine_bay_full")
    ap.add_argument("--ann-config", type=Path, default=ROOT / "configs" / "data_engine_bay.yaml")
    ap.add_argument("--train-classes", type=Path, default=ROOT / "configs" / "engine_bay_train_classes.yaml")
    ap.add_argument("--taxonomy", type=Path, default=None,
                    help="configs/taxonomy_v2.yaml: output the v2 training classes (tier A + generic fallbacks) "
                         "instead of the v1 training ontology; base and reviewed labels are remapped by name")
    ap.add_argument("--extra-from", nargs="*", default=[],
                    help="with --taxonomy: DIR:SPLIT 36-class reviewed label folders (e.g. data/engine_bay_reviewed:train). "
                         "For stems labelled from --base (v1 ids, which dropped non-v1 classes) the instances of classes "
                         "that v1 dropped but v2 trains (heat shield, generic fallbacks) are added back from here")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--dup-groups", type=Path, default=ROOT / "data" / "engine_bay_phase2" / "DUPLICATES.json",
                    help="find_duplicates.py output; each group of identical/near-identical photos stays in one fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rfs-t", type=float, default=0.10)
    ap.add_argument("--max-repeat", type=int, default=4)
    ap.add_argument("--no-ext", action="store_true", help="leave the external (EXT__*) images out")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="also keep base labels of stems that are still pseudo-labelled (not recommended)")
    a = ap.parse_args()

    base, out = a.base.resolve(), a.out.resolve()
    cfg = yaml.safe_load((base / "data_engine_bay_train.yaml").read_text(encoding="utf-8"))
    base_names = {int(k): v for k, v in cfg["names"].items()}
    ann = {int(k): v for k, v in yaml.safe_load(a.ann_config.read_text(encoding="utf-8"))["names"].items()}
    tc = yaml.safe_load(a.train_classes.read_text(encoding="utf-8"))
    names, remap, train_remap = build_remaps(base_names, ann, tc, a.taxonomy)

    status, orig_path = {}, {}
    if a.candidates.exists():
        for line in a.candidates.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            status[r["stem"]] = r["status"]
            orig_path[r["stem"]] = ROOT / r["path"]

    labels: dict[str, list[str]] = {}
    image_of: dict[str, Path] = {}
    source: dict[str, str] = {}
    stats = Counter()

    def locate(stem: str, first: Path) -> Path | None:
        for folder in (first, *(base / "images" / sp for sp in ("train", "val", "test"))):
            img = find_image(folder, stem)
            if img is not None:
                return img
        p = orig_path.get(stem)
        return p if p is not None and p.exists() else None

    # 1) reviewed sources, highest precedence first: training-id folders, then apply_review outputs (36-class ids)
    rev_excluded = set()
    specs = [(x, True) for x in a.reviewed_train] + [(x, False) for x in a.reviewed]
    for spec, train_ids in specs:
        d, _, split = spec.rpartition(":")
        d = Path(d) if Path(d).is_absolute() else ROOT / d
        if not (d / "labels" / split).is_dir():
            print(f"note: reviewed source {d}/labels/{split} not found, skipped")
            continue
        ex = d / "EXCLUDED.txt"
        if ex.exists():
            rev_excluded |= {x.split("/")[-1] for x in ex.read_text().split()}
        for lp in sorted((d / "labels" / split).glob("*.txt")):
            if lp.stem in rev_excluded or lp.stem in labels:
                continue
            img = locate(lp.stem, d / "images" / split)
            if img is None:
                stats[f"missing_image_{d.name}"] += 1
                continue
            lines = []
            for line in lp.read_text().splitlines():
                s = line.split()
                if len(s) < 7:
                    continue
                new = train_remap.get(int(s[0])) if train_ids else remap.get(int(s[0]))
                if new is None:
                    stats["reviewed_instances_dropped_non_training_class"] += 1
                    continue
                lines.append(" ".join([str(new), *s[1:]]))
            labels[lp.stem], image_of[lp.stem], source[lp.stem] = lines, img, d.name
            stats[f"from_{d.name}"] += 1
    p2_excluded = rev_excluded

    # 2) base labels of already-reviewed stems (all splits)
    ext_stems = []
    for split in ("train", "val", "test"):
        for lp in sorted((base / "labels" / split).glob("*.txt")):
            stem = lp.stem
            if stem in labels or stem in p2_excluded:
                continue
            if stem.startswith("EXT__"):
                if not a.no_ext and split == "train":
                    img = find_image(base / "images" / split, stem)
                    if img is not None:
                        labels[stem], image_of[stem], source[stem] = lp.read_text().splitlines(), img, "external"
                        ext_stems.append(stem)
                continue
            st = status.get(stem)
            if st == "excluded":
                stats["skipped_excluded_base"] += 1
                continue
            if st != "reviewed" and not (split in ("val", "test")) and not a.allow_unreviewed:
                stats["skipped_unreviewed_base"] += 1
                continue
            img = find_image(base / "images" / split, stem)
            if img is None:
                stats["base_missing_image"] += 1
                continue
            kept = []
            for l in lp.read_text().splitlines():
                s = l.split()
                if not s:
                    continue
                new = train_remap.get(int(s[0]))
                if new is None:
                    stats["base_instances_dropped_non_training_class"] += 1
                    continue
                kept.append(" ".join([str(new), *s[1:]]))
            labels[stem] = kept
            image_of[stem], source[stem] = img, f"base_{split}"
            stats[f"from_base_{split}"] += 1

    # 3) v2 only: add back instances of classes the v1 base build dropped
    if a.extra_from:
        if a.taxonomy is None:
            raise SystemExit("--extra-from needs --taxonomy")
        _, v1_ann_remap, _ = build_remaps(base_names, ann, tc)
        dropped_by_v1 = {k for k, v in v1_ann_remap.items() if v is None and remap.get(k) is not None}
        for spec in a.extra_from:
            d, _, split = spec.rpartition(":")
            d = Path(d) if Path(d).is_absolute() else ROOT / d
            if not (d / "labels" / split).is_dir():
                print(f"note: extra source {d}/labels/{split} not found, skipped")
                continue
            for lp in sorted((d / "labels" / split).glob("*.txt")):
                s = lp.stem
                if s not in labels or not source[s].startswith("base_"):
                    continue
                for line in lp.read_text().splitlines():
                    p = line.split()
                    if len(p) >= 7 and int(p[0]) in dropped_by_v1:
                        labels[s].append(" ".join([str(remap[int(p[0])]), *p[1:]]))
                        stats["extra_instances_added_v2"] += 1

    dataset_stems = sorted(s for s in labels if not s.startswith("EXT__"))
    missing = sorted(s for s, st in status.items() if st in ("unreviewed_in_train", "unreviewed_not_in_train")
                     and s not in labels)
    stats["dataset_images_still_unlabelled"] = len(missing)

    # write pool
    for s in labels:
        dst_img = out / "images" / "all" / f"{s}.jpg"
        if not dst_img.exists():
            import cv2
            src = image_of[s]
            im = cv2.imread(str(src)) if src.stat().st_size else None
            if im is None:  # unreadable / placeholder: link as is
                link(src, dst_img)
            elif src.suffix.lower() == ".jpg" and max(im.shape[:2]) <= a.max_side:
                link(src, dst_img)
            else:
                h, w = im.shape[:2]
                f = min(1.0, a.max_side / max(h, w))
                if f < 1.0:
                    im = cv2.resize(im, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dst_img), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                stats["images_resized"] += 1
        lp = out / "labels" / "all" / f"{s}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        if lp.exists():
            lp.unlink()
        lp.write_text("\n".join(labels[s]) + ("\n" if labels[s] else ""))

    # folds: per vehicle shuffle + round robin with a per-vehicle offset
    rng = random.Random(a.seed)
    by_vehicle = defaultdict(list)
    for s in dataset_stems:
        by_vehicle[s.split("__")[0]].append(s)
    # duplicate / near-duplicate photos (find_duplicates.py) are dealt as one unit so they never straddle train and val
    rep = {s: s for s in dataset_stems}
    if a.dup_groups and a.dup_groups.exists():
        for g in json.loads(a.dup_groups.read_text(encoding="utf-8")).get("groups", []):
            g = [s for s in g if s in rep]
            for s in g[1:]:
                rep[s] = g[0]
        stats["duplicate_images_grouped"] = sum(1 for s, r in rep.items() if s != r)
    fold_of = {}
    for i, v in enumerate(sorted(by_vehicle)):
        units = sorted({rep[s] for s in by_vehicle[v]})
        rng.shuffle(units)
        unit_fold = {u: (j + i) % a.folds for j, u in enumerate(units)}
        for s in by_vehicle[v]:
            fold_of[s] = unit_fold.get(rep[s], unit_fold.get(s, 0))
    for s in dataset_stems:  # a cross-vehicle group follows its representative
        if rep[s] != s and rep[s] in fold_of:
            fold_of[s] = fold_of[rep[s]]

    base_cfg = {"path": str(out), "nc": len(names), "names": names}
    fold_stats = []
    for k in range(a.folds):
        val = sorted(s for s in dataset_stems if fold_of[s] == k)
        trn = sorted(s for s in dataset_stems if fold_of[s] != k) + ext_stems
        (out / f"fold{k}_val.txt").write_text("".join(f"./images/all/{s}.jpg\n" for s in val))
        (out / f"fold{k}_train.txt").write_text("\n".join(repeat_list(trn, labels, random.Random(a.seed + k),
                                                                     a.rfs_t, a.max_repeat)) + "\n")
        (out / f"data_fold{k}.yaml").write_text(yaml.safe_dump(
            {**base_cfg, "train": f"fold{k}_train.txt", "val": f"fold{k}_val.txt", "test": f"fold{k}_val.txt"},
            sort_keys=False, allow_unicode=True))
        fold_stats.append({"fold": k, "val_images": len(val), "train_images": len(trn),
                           "val_instances": sum(len(labels[s]) for s in val)})
    full = dataset_stems + ext_stems
    (out / "full_train.txt").write_text("\n".join(repeat_list(full, labels, random.Random(a.seed + 99),
                                                              a.rfs_t, a.max_repeat)) + "\n")
    (out / "data_full.yaml").write_text(yaml.safe_dump(
        {**base_cfg, "train": "full_train.txt", "val": "fold0_val.txt", "test": "fold0_val.txt", "all": "images/all"},
        sort_keys=False, allow_unicode=True))
    # test_cases.py over every dataset image (EXT excluded): a folder of links without EXT images
    for s in dataset_stems:
        link(out / "images" / "all" / f"{s}.jpg", out / "images" / "dataset" / f"{s}.jpg")
        link(out / "labels" / "all" / f"{s}.txt", out / "labels" / "dataset" / f"{s}.txt")
    cfg_eval = yaml.safe_load((out / "data_full.yaml").read_text(encoding="utf-8"))
    cfg_eval["dataset"] = "images/dataset"
    (out / "data_full.yaml").write_text(yaml.safe_dump(cfg_eval, sort_keys=False, allow_unicode=True))

    cls = Counter(names[int(l.split()[0])] for s in dataset_stems for l in labels[s])
    summary = {**stats, "dataset_images": len(dataset_stems), "vehicles": len(by_vehicle),
               "ext_images": len(ext_stems), "instances_per_class": dict(cls.most_common()),
               "sources": dict(Counter(source[s] for s in dataset_stems)), "folds": fold_stats,
               "still_unlabelled_examples": missing[:10]}
    (out / "FULL_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if missing:
        print(f"WARNING: {len(missing)} dataset images still have no reviewed label (finish Phase 2 first)")


if __name__ == "__main__":
    main()
