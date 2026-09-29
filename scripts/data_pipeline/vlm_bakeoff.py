"""
Bake-off: an alternative VLM (e.g. DeepSeek) vs the existing Qwen3-VL box labels, scored against
the expert-reviewed engine-bay labels.

Ground truth  data/engine_bay_reviewed/labels/<split>/*.txt   (YOLO-seg polygons -> boxes)
Qwen          data/engine_bay_labeled/raw_annotations/<split>/*.json   (already computed, no requests)
Candidate     OpenAI-compatible chat endpoint, same prompt/classes/parser as qwen_grounding_annotator.py

Caveat: the reviewed labels were produced by correcting Qwen's boxes, so box geometry of "correct"
instances is anchored to Qwen. Treat a small Qwen lead in precision as possibly inflated; a candidate
lead is conservative.

Credentials come from .env (loaded at runtime, never printed):
    --key-env       name of the API-key variable        (default DEEPSEEK_API_KEY)
    --base-url-env  name of the base-URL variable       (default DEEPSEEK_BASE_URL; falls back to --base-url)

Usage:
    python scripts/data_pipeline/vlm_bakeoff.py --qwen-only                      # sanity check, no API calls
    python scripts/data_pipeline/vlm_bakeoff.py --model deepseek-v4-flash-vision-exp --limit 5    # smoke test
    python scripts/data_pipeline/vlm_bakeoff.py --model deepseek-v4-flash-vision-exp --limit 60
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data_pipeline"))
from qwen_grounding_annotator import (  # noqa: E402
    build_prompt,
    encode_image,
    parse_detections,
)

GROUNDING_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>", re.S)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reviewed", type=Path, default=PROJECT_ROOT / "data/engine_bay_reviewed")
    ap.add_argument("--labeled", type=Path, default=PROJECT_ROOT / "data/engine_bay_labeled")
    ap.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/vlm_bakeoff")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--limit", type=int, default=60, help="images, round-robin across vehicles")
    ap.add_argument("--model", default="deepseek-v4-flash-vision-exp")
    ap.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled",
                    help="DeepSeek thinking mode (Qwen ran without reasoning)")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--base-url-env", default="DEEPSEEK_BASE_URL")
    ap.add_argument("--max-side", type=int, default=1400, help="same as the Qwen run")
    ap.add_argument("--jpeg-quality", type=int, default=88)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--no-json-mode", action="store_true", help="omit response_format (some servers reject it)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--qwen-only", action="store_true", help="score Qwen only; no API calls")
    return ap.parse_args()


# ---------------------------------------------------------------- data

def pick_images(reviewed: Path, labeled: Path, splits: list[str], limit: int) -> list[tuple[str, str, Path]]:
    by_vehicle: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)
    for split in splits:
        for img in sorted((reviewed / "images" / split).iterdir()):
            if (labeled / "raw_annotations" / split / f"{img.stem}.json").exists():
                by_vehicle[img.stem.split("__")[0]].append((split, img.stem, img))
    picked, queues = [], [list(v) for _, v in sorted(by_vehicle.items())]
    while len(picked) < limit and any(queues):
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
    return picked


def load_gt(reviewed: Path, split: str, stem: str, names: dict[int, str]) -> list[dict]:
    path = reviewed / "labels" / split / f"{stem}.txt"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        parts = line.split()
        if len(parts) < 5:
            continue
        xy = list(map(float, parts[1:]))
        xs, ys = xy[0::2], xy[1::2]
        out.append({"class_name": names[int(parts[0])], "box": [min(xs), min(ys), max(xs), max(ys)]})
    return out


def load_qwen(labeled: Path, split: str, stem: str) -> list[dict]:
    ann = json.loads((labeled / "raw_annotations" / split / f"{stem}.json").read_text(encoding="utf-8"))
    return [{"class_name": d["class_name"], "box": d["bbox_norm_xyxy"], "confidence": d["confidence"]}
            for d in ann["detections"]]


# ---------------------------------------------------------------- candidate VLM

def grounding_to_json(text: str) -> str:
    """DeepSeek-VL2 style <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2],...]<|/det|> (0-999) -> our JSON."""
    dets = []
    for label, boxes in GROUNDING_RE.findall(text):
        for box in re.findall(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", boxes):
            dets.append({"class": label.strip(), "bbox_2d": [float(v) * 1000 / 999 for v in box]})
    return json.dumps({"detections": dets})


def call_candidate(client, model: str, image_uri: str, prompt: str, json_mode: bool,
                   thinking: str) -> tuple[str, float, dict]:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    if thinking == "disabled":
        kwargs["temperature"] = 0.0  # DeepSeek thinking mode rejects temperature
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": prompt},
        ]}],
        # reasoning tokens count against max_tokens; 2500 left thinking runs with empty content
        max_tokens=2500 if thinking == "disabled" else 16000,
        extra_body={"thinking": {"type": thinking}},
        **kwargs,
    )
    choice = resp.choices[0]
    info = {"finish_reason": choice.finish_reason,
            "reasoning_chars": len(getattr(choice.message, "reasoning_content", None) or ""),
            "usage": resp.usage.model_dump() if resp.usage else None}
    return choice.message.content or "", time.time() - t0, info


def run_candidate(args, images, class_to_id, prompt) -> dict[str, list[dict]]:
    from dotenv import dotenv_values, load_dotenv
    from openai import OpenAI

    load_dotenv(PROJECT_ROOT / ".env")
    key_env = args.key_env
    if not os.environ.get(key_env):
        # fall back to a single .env variable named like *DEEPSEEK*KEY* (only the name is printed)
        found = [k for k in dotenv_values(PROJECT_ROOT / ".env") if re.search(r"deepseek.*key|key.*deepseek", k, re.I)]
        if len(found) != 1:
            sys.exit(f"{args.key_env} is not set and {len(found)} *DEEPSEEK*KEY* variables found in .env; "
                     "pass --key-env <VARIABLE_NAME>")
        key_env = found[0]
        print(f"using API key from .env variable {key_env}", flush=True)
    key = os.environ[key_env]
    base_url = os.environ.get(args.base_url_env) or args.base_url
    client = OpenAI(api_key=key, base_url=base_url, timeout=args.timeout, max_retries=2)

    slug = re.sub(r"[^\w.-]", "_", args.model)
    raw_dir = args.output / "raw" / f"{slug}__thinking_{args.thinking}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def one(item):
        split, stem, img = item
        cache = raw_dir / f"{stem}.json"
        if cache.exists():
            rec = json.loads(cache.read_text(encoding="utf-8"))
        else:
            uri, width, height = encode_image(img, args.max_side, args.jpeg_quality)
            try:
                text, secs, info = call_candidate(client, args.model, uri, prompt, not args.no_json_mode,
                                                  args.thinking)
                rec = {"response": text, "seconds": round(secs, 2), "width": width, "height": height, **info}
                if not text.strip():
                    rec["error"] = f"empty content (finish_reason={info['finish_reason']})"
            except Exception as e:  # keep going; recorded as a failed image
                rec = {"error": f"{type(e).__name__}: {e}"[:500], "width": width, "height": height}
            if "error" not in rec:  # only cache successful responses so failures are retried
                cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        if "error" in rec:
            return stem, None, rec
        text = rec["response"]
        if "<|det|>" in text:
            text = grounding_to_json(text)
        try:
            dets, warnings = parse_detections(text, class_to_id, rec["width"], rec["height"], 0.0, 0.0)
        except (json.JSONDecodeError, ValueError) as e:
            rec["error"] = f"unparseable: {e}"
            return stem, None, rec
        rec["warnings"] = warnings
        return stem, [{"class_name": d["class_name"], "box": d["bbox_norm_xyxy"],
                       "confidence": d["confidence"]} for d in dets], rec

    preds, meta = {}, {}
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (stem, dets, rec) in enumerate(pool.map(one, images), 1):
            meta[stem] = rec
            if dets is not None:
                preds[stem] = dets
            status = "ERR " + rec["error"][:120] if "error" in rec else f"{len(dets)} dets"
            print(f"[{i}/{len(images)}] {stem}: {status}", flush=True)
    args._candidate_meta = meta
    return preds


# ---------------------------------------------------------------- scoring

def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds: list[dict], gts: list[dict], thr: float, class_aware: bool) -> tuple[int, int, int, Counter]:
    """Greedy one-to-one matching by IoU. Returns tp, fp, fn and per-class tp/fp/fn counters."""
    pairs = sorted(
        ((iou(p["box"], g["box"]), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)
         if not class_aware or p["class_name"] == g["class_name"]),
        reverse=True,
    )
    used_p, used_g = set(), set()
    for v, pi, gi in pairs:
        if v < thr:
            break
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    per = Counter()
    for pi, p in enumerate(preds):
        per[(p["class_name"], "tp" if pi in used_p else "fp")] += 1
    for gi, g in enumerate(gts):
        if gi not in used_g:
            per[(g["class_name"], "fn")] += 1
    return len(used_p), len(preds) - len(used_p), len(gts) - len(used_g), per


def prf(tp, fp, fn) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(system_preds: dict[str, list[dict]], gt: dict[str, list[dict]], stems: list[str], thr: float) -> dict:
    tot = Counter()
    per_cls = Counter()
    for stem in stems:
        preds = system_preds.get(stem, [])
        tp, fp, fn, per = match(preds, gt[stem], thr, class_aware=True)
        tot.update(tp=tp, fp=fp, fn=fn)
        per_cls.update(per)
        atp, afp, afn, _ = match(preds, gt[stem], thr, class_aware=False)
        tot.update(atp=atp, afp=afp, afn=afn)
    return {"tot": tot, "per_cls": per_cls}


def main() -> None:
    args = parse_args()
    names = yaml.safe_load((args.reviewed / "data_engine_bay_reviewed.yaml").read_text(encoding="utf-8"))["names"]
    names = {int(k): v for k, v in names.items()}
    class_to_id = {v: k for k, v in names.items()}
    prompt = build_prompt(names)

    images = pick_images(args.reviewed, args.labeled, args.splits, args.limit)
    vehicles = Counter(stem.split("__")[0] for _, stem, _ in images)
    print(f"{len(images)} images from {len(vehicles)} vehicles ({', '.join(args.splits)})", flush=True)
    gt = {stem: load_gt(args.reviewed, split, stem, names) for split, stem, _ in images}
    systems = {"qwen3-vl-30b (existing)": {stem: load_qwen(args.labeled, split, stem) for split, stem, _ in images}}

    if not args.qwen_only:
        args.output.mkdir(parents=True, exist_ok=True)
        cand = run_candidate(args, images, class_to_id, prompt)
        systems[args.model] = cand
        failed = [s for _, s, _ in images if s not in cand]
        if failed:
            print(f"{len(failed)} images failed for {args.model}; all systems scored on the {len(cand)} common images")
        stems = [s for _, s, _ in images if s in cand]
    else:
        stems = [s for _, s, _ in images]
    if not stems:
        sys.exit("no images scored")

    n_gt = sum(len(gt[s]) for s in stems)
    lines = [
        "# VLM bake-off vs expert-reviewed engine-bay labels",
        "",
        f"{len(stems)} images, {len({s.split('__')[0] for s in stems})} vehicles, {n_gt} reviewed objects; "
        f"box match IoU >= {args.iou}. Qwen boxes are the pre-review originals.",
        "",
        "| system | pred/img | precision | recall | F1 | class-agnostic P / R |",
        "|---|---|---|---|---|---|",
    ]
    results = {}
    for name, preds in systems.items():
        s = score(preds, gt, stems, args.iou)
        results[name] = s
        t = s["tot"]
        p, r, f = prf(t["tp"], t["fp"], t["fn"])
        ap, ar, _ = prf(t["atp"], t["afp"], t["afn"])
        lines.append(f"| {name} | {(t['tp'] + t['fp']) / len(stems):.1f} | {p:.1%} | {r:.1%} | {f:.3f} | {ap:.1%} / {ar:.1%} |")

    if not args.qwen_only:
        secs = [m["seconds"] for s, m in args._candidate_meta.items() if "seconds" in m]
        if secs:
            lines.append(f"\n{args.model}: median {sorted(secs)[len(secs) // 2]:.1f}s/image")

    gt_cls = Counter(g["class_name"] for s in stems for g in gt[s])
    lines += ["", "## Per class F1 (classes with >= 5 reviewed objects)", "",
              "| class | GT | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
    for cls, n in gt_cls.most_common():
        if n < 5:
            continue
        cells = []
        for name in systems:
            pc = results[name]["per_cls"]
            cells.append(f"{prf(pc[(cls, 'tp')], pc[(cls, 'fp')], pc[(cls, 'fn')])[2]:.2f}")
        lines.append(f"| {cls} | {n} | " + " | ".join(cells) + " |")

    report = "\n".join(lines) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.-]", "_", args.model)
    out = args.output / ("REPORT_qwen_only.md" if args.qwen_only else f"REPORT_{slug}.md")
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
