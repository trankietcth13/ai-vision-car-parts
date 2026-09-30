"""Jev diagnosis triage CLI.

  python scripts/diagnosis/jev_triage.py check
      verify the API key and list the models the account can use
  python scripts/diagnosis/jev_triage.py run --complaint "engine shakes at idle, check engine light on" --dtc P0302
      triage one complaint (prints JSON)
  python scripts/diagnosis/jev_triage.py cases --cases configs/jev_triage_smoke_cases.jsonl --out qa_results/jev_triage
      triage every JSONL case ({"complaint", "dtc"?, "vehicle"?, "expected_system"?, "expected_action"?,
      "expected_targets"?}); writes results.jsonl + prints agreement with the expected fields

The API key is read from TYPESAFE_API_KEY or the project .env (never printed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from jev import DEFAULT_MODEL, load_knowledge, make_client, triage  # noqa: E402


def cmd_check(_: argparse.Namespace) -> None:
    with make_client() as client:
        for m in client.models.list().models:
            print(m.name, m.release_date, "-", m.description)
    print(f"OK, default model for this project: {DEFAULT_MODEL}")


def cmd_run(a: argparse.Namespace) -> None:
    res = triage(a.complaint, a.dtc, a.vehicle)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))


def cmd_cases(a: argparse.Namespace) -> None:
    knowledge = load_knowledge()
    cases = [json.loads(line) for line in Path(a.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hits = {"system": [0, 0], "action": [0, 0], "targets": [0, 0]}
    with make_client() as client, open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for case in cases:
            r = triage(case["complaint"], case.get("dtc"), case.get("vehicle"), client=client, knowledge=knowledge)
            row = {"case": case, "result": r.to_dict()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if "expected_system" in case:
                hits["system"][0] += r.system == case["expected_system"]; hits["system"][1] += 1
            if "expected_action" in case:
                hits["action"][0] += r.action == case["expected_action"]; hits["action"][1] += 1
            for t in case.get("expected_targets", []):
                hits["targets"][0] += t in r.detector_targets; hits["targets"][1] += 1
            print(f"{r.action:14s} {r.system:22s} conf={r.system_confidence:.2f} targets={r.detector_targets[:4]} | {case['complaint'][:60]}")
    for k, (ok, n) in hits.items():
        if n:
            print(f"{k:8s} {ok}/{n} = {ok / n:.0%}")
    print(f"results: {out / 'results.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    r = sub.add_parser("run")
    r.add_argument("--complaint", default="")
    r.add_argument("--dtc", nargs="*", default=[])
    r.add_argument("--vehicle", default=None)
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("cases")
    c.add_argument("--cases", default="configs/jev_triage_smoke_cases.jsonl")
    c.add_argument("--out", default="qa_results/jev_triage")
    c.set_defaults(fn=cmd_cases)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
