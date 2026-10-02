"""Package trained models into a versioned release folder: models/<name>_<version>/.

Copies each checkpoint under a standard file name (<role>_<arch>_<version>.pt), optionally exports ONNX, and writes
  manifest.json      md5, size, architecture, source run and test metrics of every file
  BENCHMARK.md       latency tables from benchmark_models.py JSON files (GPU / CPU ...)
  RELEASE_NOTES.md   --summary text (hand written) + generated accuracy, per-class and file tables
Weights are git-ignored (*.pt, *.onnx); the three text files are meant to be committed.

    python scripts/deployment/package_release.py --name engine_bay_yolo26-seg --version v3.0 \
        --model teacher=.../avg5.pt --model student_kd=.../avg5.pt --qa QA_REPORT.json \
        --qa-key teacher=teacher_v26l_avg5 --qa-key student_kd=kd_26s_v26_s0_avg5 \
        --ref teacher_yolo11l_v10=teacher_v10_avg5 --bench "GPU (DGX)=bench_gpu.json" --summary notes.md --onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arch_of(path: Path) -> tuple[str, float, bool]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # Ultralytics checkpoints are pickled modules
    m = (ckpt.get("ema") or ckpt.get("model")).float()
    arch, nms_free = Path(m.yaml.get("yaml_file", "model")).stem, bool(getattr(m.model[-1], "end2end", False))
    m.fuse(verbose=False)  # deployed size, as qa_test.py reports it: BN folded, YOLO26 one-to-many head dropped
    return arch, round(sum(p.numel() for p in m.parameters()) / 1e6, 2), nms_free


def pairs(items: list[str]) -> dict[str, str]:
    return dict(s.split("=", 1) for s in items or [])


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def accuracy_table(rows: dict[str, dict]) -> list[str]:
    out = ["| Model | Params (M) | mask mAP50-95 | mask mAP50 | box mAP50-95 | Precision | Recall | GPU latency (ms, fp16, network only) |",
           "|---|---|---|---|---|---|---|---|"]
    for label, m in rows.items():
        out.append(f"| {label} | {fmt(m.get('params_M'), 1)} | **{fmt(m.get('mask_map50_95'))}** | {fmt(m.get('mask_map50'))} | "
                   f"{fmt(m.get('box_map50_95'))} | {fmt(m.get('mask_precision'))} | {fmt(m.get('mask_recall'))} | "
                   f"{fmt(m.get('latency_ms_bs1'), 1)} |")
    return out


def per_class_table(rows: dict[str, dict]) -> list[str]:
    classes = list(next(iter(rows.values())).get("per_class", {}))
    out = ["| Class | " + " | ".join(rows) + " |", "|---|" + "---|" * len(rows)]
    for c in classes:
        out.append(f"| {c} | " + " | ".join(fmt(m.get("per_class", {}).get(c, {}).get("mask_ap50_95")) for m in rows.values()) + " |")
    return out


def bench_tables(benches: dict[str, dict]) -> list[str]:
    out = []
    for label, b in benches.items():
        env = b["environment"]
        out += [f"### {label}", "",
                f"{env['device']} · torch {env['torch']} · ultralytics {env['ultralytics']} · {len(b['images'])} photos × "
                f"{b['iters_per_image']} runs · batch 1 · fp32 · conf {b['conf']} · imgsz {b['imgsz']}", "",
                "| Model | Format | Median (ms) | Mean (ms) | p90 (ms) | Pre / inference / post (ms) | FPS | Detections |",
                "|---|---|---|---|---|---|---|---|"]
        for name, m in b["models"].items():
            fps = 1000.0 / m["median_ms"] if m["median_ms"] else 0
            out.append(f"| {name} | {m['format']} | **{m['median_ms']}** | {m['mean_ms']} | {m['p90_ms']} | "
                       f"{m['preprocess_ms']} / {m['inference_ms']} / {m['postprocess_ms']} | {fps:.1f} | {m['detections']} |")
        out.append("")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--model", action="append", required=True, help="role=checkpoint.pt (role: teacher, student_kd, ...)")
    ap.add_argument("--source", action="append", help="role=run path on the training machine (manifest only)")
    ap.add_argument("--qa", help="QA_REPORT.json of qa_test.py (test split)")
    ap.add_argument("--qa-key", action="append", help="role=model name in the QA report")
    ap.add_argument("--ref", action="append", help="label=model name in the QA report, shown as a reference row")
    ap.add_argument("--bench", action="append", help="label=benchmark_models.py JSON")
    ap.add_argument("--summary", help="markdown file placed at the top of RELEASE_NOTES.md")
    ap.add_argument("--n-test", type=int, help="test photos (qa_test.py counts 0 when the split is a .txt list)")
    ap.add_argument("--onnx", action="store_true", help="also export every model to ONNX (opset 18, imgsz 640)")
    ap.add_argument("--out-root", default=str(ROOT / "models"))
    a = ap.parse_args()

    out = Path(a.out_root) / f"{a.name}_{a.version}"
    out.mkdir(parents=True, exist_ok=True)
    qa = json.loads(Path(a.qa).read_text(encoding="utf-8")) if a.qa else {"models": {}}
    qa_keys, sources = pairs(a.qa_key), pairs(a.source)
    manifest = {"name": a.name, "version": a.version, "date": date.today().isoformat(), "files": []}
    acc_rows: dict[str, dict] = {}

    for role, src in pairs(a.model).items():
        src = Path(src)
        arch, params, nms_free = arch_of(src)
        dst = out / f"{role}_{arch}_{a.version}.pt"
        shutil.copyfile(src, dst)
        metrics = qa["models"].get(qa_keys.get(role, ""), {})
        entry = {"role": role, "file": dst.name, "arch": arch, "params_M": params, "nms_free": nms_free,
                 "md5": md5(dst), "size_mb": round(dst.stat().st_size / 2**20, 1), "source": sources.get(role, str(src)),
                 "qa_name": qa_keys.get(role), "test_metrics": {k: v for k, v in metrics.items() if k != "per_class"}}
        manifest["files"].append(entry)
        if a.onnx:
            from ultralytics import YOLO

            onnx = Path(YOLO(str(dst)).export(format="onnx", imgsz=640, opset=18, simplify=True, device="cpu"))
            manifest["files"].append({**{k: entry[k] for k in ("role", "arch", "params_M", "nms_free")},
                                      "file": onnx.name, "md5": md5(onnx), "size_mb": round(onnx.stat().st_size / 2**20, 1),
                                      "source": dst.name})
        if metrics:
            acc_rows[f"{role} ({arch})"] = metrics
    for label, key in pairs(a.ref).items():
        if key in qa["models"]:
            acc_rows[f"{label} (reference)"] = qa["models"][key]

    benches = {label: json.loads(Path(p).read_text(encoding="utf-8")) for label, p in pairs(a.bench).items()}
    n_test = a.n_test or qa.get("n_images")
    manifest["qa"] = {"report": a.qa, "split": qa.get("split"), "n_images": n_test, "data": qa.get("data")}
    manifest["benchmarks"] = {label: {"environment": b["environment"], "models": b["models"]} for label, b in benches.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    bench_md = [f"# Benchmark · {a.name} {a.version}", "",
                "End-to-end `model.predict` per photo, measured like the web app benchmark (`scripts/deployment/benchmark_models.py`):",
                "photo downscaled to 1600 px, 2 warm-up runs per photo, median over all timed runs. Post = NMS + masks; YOLO26",
                "heads are end-to-end, so they skip NMS.", "", *bench_tables(benches)]
    (out / "BENCHMARK.md").write_text("\n".join(bench_md), encoding="utf-8")

    notes = [Path(a.summary).read_text(encoding="utf-8").rstrip(), ""] if a.summary else [f"# {a.name} {a.version}", ""]
    if acc_rows:
        notes += [f"## Accuracy on the held-out test split ({n_test} photos, vehicles never seen in training)", "",
                  *accuracy_table(acc_rows), "", "### Per-class mask AP50-95", "", *per_class_table(acc_rows), ""]
    if benches:
        notes += ["## Speed", "", "Full tables: [BENCHMARK.md](BENCHMARK.md).", "", *bench_tables(benches)]
    notes += ["## Files", "", "| File | Role | Architecture | Params (M) | NMS | Size (MB) | md5 |", "|---|---|---|---|---|---|---|"]
    notes += [f"| `{f['file']}` | {f['role']} | {f['arch']} | {fmt(f['params_M'], 1)} | {'none (end-to-end)' if f['nms_free'] else 'yes'} | "
              f"{f['size_mb']} | `{f['md5']}` |" for f in manifest["files"]]
    notes += ["", "Sources and full metrics: [manifest.json](manifest.json). Weights are not in git (`*.pt`, `*.onnx` are ignored)."]
    (out / "RELEASE_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(f"[release] {out}: {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
