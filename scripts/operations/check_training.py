"""
One-shot status of the engine-bay training runs (local GPU + DGX).

Usage:
    python scripts/operations/check_training.py                 # default runs
    python scripts/operations/check_training.py --dgx engine_teacher_v3_dgx engine_teacher_v4_dgx
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DGX = os.environ.get("DGX_SSH_HOST", "admin@dgx-host")
DGX_REMOTE = os.environ.get("DGX_REMOTE_DIR", "/srv/distillation_workspace")
DGX_RUNS = f"{DGX_REMOTE}/runs/segment"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def summarize(rows, name, where):
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    if not rows:
        print(f"[{where}] {name}: started, no finished epoch yet")
        return
    key = "metrics/mAP50-95(M)"
    best = max(rows, key=lambda r: float(r[key]))
    last = rows[-1]
    print(f"[{where}] {name}: epoch {last['epoch']} | last mask mAP50-95 {float(last[key]):.3f} "
          f"| best {float(best[key]):.3f} @ epoch {best['epoch']}")


def local(name):
    f = ROOT / "runs" / "segment" / name / "results.csv"
    if not f.exists():
        print(f"[local] {name}: no results.csv yet")
        return
    with open(f, newline="") as fh:
        summarize(list(csv.DictReader(fh)), name, "local")
    log = ROOT / "runs" / "logs" / f"{name}.log"
    if log.exists():
        tail = log.read_text(encoding="utf-8", errors="ignore")[-3000:].replace("\r", "\n").splitlines()
        tail = [l for l in tail if l.strip()]
        if any("finished" in l for l in tail[-5:]):
            print(f"[local] {name}: FINISHED")
        elif any("Traceback" in l or "Error" in l for l in tail[-20:]):
            print(f"[local] {name}: ERROR -> see {log}")
        elif tail:
            print(f"[local] now: {tail[-1].strip()[:150]}")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15).stdout.strip()
        print(f"[local] GPU: {gpu}")
    except Exception:
        pass


def dgx(name):
    cmd = (f"cat {DGX_RUNS}/{name}/results.csv 2>/dev/null; echo '@@'; "
           f"tail -c 1500 {DGX_REMOTE}/runs/logs/{name}.log 2>/dev/null | tr '\\r' '\\n' | grep -v '^$' | tail -1; "
           "echo '@@'; free -g | awk '/Mem:/{print $3\" GB used / \"$2\" GB, available \"$7\" GB\"}'; "
           f"pgrep -f {name}.py >/dev/null && echo running || echo stopped")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", DGX, cmd],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if r.returncode != 0 and not r.stdout:
        print(f"[dgx] ssh failed: {r.stderr.strip()[:200]}")
        return
    csv_part, now, rest = (r.stdout.split("@@") + ["", ""])[:3]
    lines = [l for l in csv_part.strip().splitlines() if l.strip()]
    if lines:
        summarize(list(csv.DictReader(lines)), name, "dgx")
    else:
        print(f"[dgx] {name}: no results.csv yet")
    if now.strip():
        print(f"[dgx] now: {now.strip()[:150]}")
    for l in rest.strip().splitlines():
        print(f"[dgx] {l}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default="", help="Local run name (training is DGX-only; leave empty)")
    ap.add_argument("--dgx", nargs="+", default=["engine_teacher_v4_dgx", "engine_teacher_v4_dgx_resume"])
    ap.add_argument("--watch", type=int, default=0, help="Refresh every N seconds (Ctrl+C to stop)")
    a = ap.parse_args()
    if a.watch:
        import os, time
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                print(time.strftime("%Y-%m-%d %H:%M:%S"), f"(refresh every {a.watch}s, Ctrl+C to stop)\n")
                run_once(a)
                time.sleep(a.watch)
        except KeyboardInterrupt:
            return
    run_once(a)


def run_once(a):
    if a.local:
        local(a.local)
        print()
    for n in a.dgx:
        dgx(n)
        print()


if __name__ == "__main__":
    main()
