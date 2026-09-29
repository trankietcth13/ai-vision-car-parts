"""
Run engine-bay training jobs on the DGX over plain ssh/scp (key-based auth).

Steps (each is idempotent):
    python scripts/operations/dgx_train.py check
    python scripts/operations/dgx_train.py setup
    python scripts/operations/dgx_train.py push --dataset data/engine_bay_train_v3
    python scripts/operations/dgx_train.py push-code
    python scripts/operations/dgx_train.py teacher --dataset engine_bay_train_v3 --name engine_teacher_v2_dgx
    python scripts/operations/dgx_train.py kd --teacher runs/segment/engine_teacher_v2_dgx/weights/best.pt --imgsz 640
    python scripts/operations/dgx_train.py status --name engine_teacher_v2_dgx
    python scripts/operations/dgx_train.py pull --name engine_teacher_v2_dgx

Jobs run under `nohup` on the DGX and survive disconnects. The DGX also serves the
Qwen3-VL vLLM endpoint used for labeling, so `check` reports its GPU memory share and
training jobs default to a conservative batch; raise it only if memory allows.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOST = os.environ.get("DGX_SSH_HOST", "admin@dgx-host")
REMOTE = os.environ.get("DGX_REMOTE_DIR", "/srv/distillation_workspace")
VENV = f"{REMOTE}/.venv"
PY = f"{VENV}/bin/python"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new"]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ssh(cmd: str, check=True, capture=False):
    r = subprocess.run(["ssh", *SSH_OPTS, HOST, cmd], text=True, capture_output=capture)
    if check and r.returncode != 0:
        if capture:
            print(r.stdout, r.stderr)
        raise SystemExit(f"[dgx] remote command failed ({r.returncode}): {cmd[:120]}")
    return r


def scp(local: Path, remote: str):
    r = subprocess.run(["scp", *SSH_OPTS, str(local), f"{HOST}:{remote}"])
    if r.returncode != 0:
        raise SystemExit(f"[dgx] scp failed: {local}")


def scp_back(remote: str, local: Path):
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["scp", "-r", *SSH_OPTS, f"{HOST}:{remote}", str(local)])
    if r.returncode != 0:
        raise SystemExit(f"[dgx] scp back failed: {remote}")


def push_tree(paths: list[Path], remote_dir: str, arc_root: Path):
    with tempfile.TemporaryDirectory() as td:
        tar_path = Path(td) / "bundle.tar"
        with tarfile.open(tar_path, "w") as tar:
            for p in paths:
                tar.add(p, arcname=str(p.relative_to(arc_root)).replace("\\", "/"),
                        filter=lambda ti: None if "__pycache__" in ti.name else ti)
        size = tar_path.stat().st_size / 1e6
        print(f"[dgx] uploading {size:.0f} MB -> {remote_dir}")
        ssh(f"mkdir -p {shlex.quote(remote_dir)}")
        scp(tar_path, f"{remote_dir}/_bundle.tar")
        ssh(f"cd {shlex.quote(remote_dir)} && tar -xf _bundle.tar && rm _bundle.tar")


def cmd_check(_):
    r = ssh("hostname; uname -m; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader; "
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; free -g | sed -n 2p; "
            f"df -h {REMOTE} 2>/dev/null | tail -1 || df -h ~ | tail -1; "
            f"test -x {PY} && {PY} -c 'import torch,ultralytics;print(\"venv torch\",torch.__version__,torch.cuda.is_available(),\"ultralytics\",ultralytics.__version__)' "
            "|| echo 'venv missing (run: python scripts/operations/dgx_train.py setup)'; "
            "curl -s -m 5 http://127.0.0.1:8000/v1/models | head -c 120; echo", check=False, capture=True)
    print(r.stdout or r.stderr)
    if r.returncode != 0:
        print("[dgx] SSH failed. Add this machine's public key to admin@dgx-host:~/.ssh/authorized_keys "
              "(see Note.md), then rerun `python scripts/operations/dgx_train.py check`.")


def cmd_setup(_):
    ssh(f"mkdir -p {REMOTE} && cd {REMOTE} && (test -x {PY} || python3 -m venv --system-site-packages .venv) && "
        f"{PY} -m pip install -q --upgrade pip && {PY} -m pip install -q 'ultralytics==8.4.75' pyyaml opencv-python-headless && "
        f"{PY} -c 'import torch,ultralytics;print(torch.__version__,torch.cuda.is_available(),ultralytics.__version__)'")


def cmd_push(a):
    ds = (ROOT / a.dataset).resolve()
    if not (ds / "data_engine_bay_train.yaml").exists():
        raise SystemExit(f"{ds} has no data_engine_bay_train.yaml")
    push_tree([ds], f"{REMOTE}/data", ds.parent)
    # rewrite the dataset path for the DGX
    ssh(f"cd {REMOTE}/data/{ds.name} && sed -i 's|^path:.*|path: {REMOTE}/data/{ds.name}|' data_engine_bay_train.yaml && head -2 data_engine_bay_train.yaml")


def cmd_push_code(_):
    paths = [
        ROOT / "src" / "distillation",
        ROOT / "scripts" / "data_pipeline",
        ROOT / "scripts" / "training" / "train_kd.py",
        ROOT / "scripts" / "training" / "train_teacher.py",
        ROOT / "configs",
    ]
    push_tree([p for p in paths if p.exists()], REMOTE, ROOT)


def launch(name: str, pycode: str):
    log = f"{REMOTE}/runs/logs/{name}.log"
    script = f"{REMOTE}/runs/logs/{name}.py"
    ssh(f"mkdir -p {REMOTE}/runs/logs && cat > {script} <<'PYEOF'\n{pycode}\nPYEOF\n"
        f"cd {REMOTE} && nohup {PY} -u {script} > {log} 2>&1 & echo launched pid $!")
    print(f"[dgx] follow with: python scripts/operations/dgx_train.py status --name {name}")


def cmd_teacher(a):
    launch(a.name, f"""from ultralytics import YOLO
YOLO({a.model!r}).train(data='{REMOTE}/data/{a.dataset}/data_engine_bay_train.yaml', imgsz={a.imgsz}, batch={a.batch},
    epochs={a.epochs}, patience={a.patience}, cache='ram', workers={a.workers}, device=0, project='{REMOTE}/runs/segment',
    name={a.name!r}, exist_ok=True, seed=0, close_mosaic=10, plots=True)""")


def cmd_resume(a):
    """Continue an interrupted run from its last.pt (same args, same run folder)."""
    ssh(f"ls {REMOTE}/runs/*/{a.name}/weights/last.pt >/dev/null")
    launch(a.name + "_resume", f"""import glob
from ultralytics import YOLO
last = glob.glob('{REMOTE}/runs/*/{a.name}/weights/last.pt')[0]
print('resuming from', last, flush=True)
YOLO(last).train(resume=True)""")
    print(f"[dgx] log: runs/logs/{a.name}_resume.log  (status: python scripts/operations/dgx_train.py status --name {a.name}_resume)")


def cmd_stop(a):
    """Stop a running job gracefully (SIGINT); last.pt keeps the last finished epoch."""
    pat = "".join(f"[{c}]" if i == 0 else c for i, c in enumerate(a.name))  # avoid matching this shell
    ssh(f"pids=$(ps -eo pid,cmd | grep '{pat}' | grep -v grep | awk '{{print $1}}'); "
        f"[ -z \"$pids\" ] && echo 'nothing running' || {{ kill -INT $pids; sleep 20; "
        f"ps -eo pid,cmd | grep '{pat}' | grep -v grep | awk '{{print $1}}' | xargs -r kill -TERM; echo stopped; }}", check=False)


def cmd_pull_qa(_):
    """Copy qa_results/ (QA report, overlays, ERRORS.json) from the DGX."""
    scp_back(f"{REMOTE}/qa_results", ROOT)
    print("[dgx] pulled qa_results/")


def cmd_kd(a):
    launch(a.name, f"""import subprocess, sys
sys.exit(subprocess.call([sys.executable, 'scripts/training/train_kd.py', '--teacher', '{REMOTE}/{a.teacher}', '--student', {a.student!r},
    '--data', '{REMOTE}/data/{a.dataset}/data_engine_bay_train.yaml', '--epochs', '{a.epochs}', '--batch', '{a.batch}',
    '--imgsz', '{a.imgsz}', '--device', '0', '--workers', '{a.workers}', '--project', '{REMOTE}/runs/train_kd', '--name', {a.name!r}]
    + (['--no-kd'] if {a.no_kd!r} else [])))""")


def cmd_status(a):
    ssh(f"tail -c 2000 {REMOTE}/runs/logs/{a.name}.log | tr '\\r' '\\n' | grep -v '^$' | tail -6; "
        f"f=$(ls {REMOTE}/runs/*/{a.name}/results.csv 2>/dev/null | head -1); test -n \"$f\" && tail -1 $f; "
        "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader", check=False)


def cmd_pull(a):
    r = ssh(f"ls -d {REMOTE}/runs/*/{a.name}", capture=True)
    remote_run = r.stdout.strip().splitlines()[0]
    kind = remote_run.rstrip("/").split("/")[-2]
    scp_back(remote_run, ROOT / "runs" / kind)
    print(f"[dgx] pulled to runs/{kind}/{a.name}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check"); sub.add_parser("setup"); sub.add_parser("push-code"); sub.add_parser("pull-qa")
    p = sub.add_parser("push"); p.add_argument("--dataset", default="data/engine_bay_train_v3")
    for n in ("teacher", "kd"):
        p = sub.add_parser(n)
        p.add_argument("--dataset", default="engine_bay_train_v3")
        p.add_argument("--name", required=True)
        p.add_argument("--epochs", type=int, default=100 if n == "teacher" else 150)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--imgsz", type=int, default=1024)
        p.add_argument("--workers", type=int, default=8)
        if n == "teacher":
            p.add_argument("--model", default="yolo11l-seg.pt")
            p.add_argument("--patience", type=int, default=40)
        else:
            p.add_argument("--teacher", required=False, default="")
            p.add_argument("--student", default="yolo11n-seg.pt")
            p.add_argument("--no-kd", action="store_true")
    for n in ("status", "pull", "resume", "stop"):
        p = sub.add_parser(n); p.add_argument("--name", required=True)
    a = ap.parse_args()
    {"check": cmd_check, "setup": cmd_setup, "push": cmd_push, "push-code": cmd_push_code, "teacher": cmd_teacher,
     "kd": cmd_kd, "status": cmd_status, "pull": cmd_pull,
     "resume": cmd_resume, "stop": cmd_stop, "pull-qa": cmd_pull_qa}[a.cmd](a)


if __name__ == "__main__":
    main()
