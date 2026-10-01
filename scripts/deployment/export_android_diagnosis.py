"""Export the DTC knowledge table to the Android app, for offline "error code -> components to inspect" lookup.

Reads configs/diagnosis_knowledge.yaml (the same table src/jev/diagnosis_triage.py uses for its exact DTC lookup),
its Vietnamese companion configs/diagnosis_knowledge_vi.yaml and the app's assets/config.json (class names), and writes:
  app/src/main/assets/diagnosis.json                       DTC table + component names, descriptions, detector class or null;
                                                           every text is {"en": ..., "vi": ...}
  app/src/test/resources/fixtures/dtc_lookup.json          Python lookup_dtc() results for a set of codes; the Kotlin
                                                           unit test checks that the app's lookup gives the same answer
The DTC lookup needs no network and no Jev: Jev is only needed for free-text complaints.
Urgency and safety notes below are generic gasoline-ICE guidance (draft, to be reviewed by an expert).

    python scripts/deployment/export_android_diagnosis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.jev.diagnosis_triage import load_knowledge, lookup_dtc  # noqa: E402

APP = ROOT / "apps" / "engine_bay_android" / "app"

VI_PATH = ROOT / "configs" / "diagnosis_knowledge_vi.yaml"  # shared EN/VI display names + Vietnamese texts

# Keyed by the first code spec of a dtc entry. urgency: stop (stop driving / stop the engine) | soon | check (default).
HOT_COOLANT = {"en": "Open the radiator or coolant reservoir cap only when the engine is cold: hot coolant under pressure can spray out and burn.",
               "vi": "Chỉ mở nắp két nước hoặc bình nước phụ khi máy đã nguội: nước làm mát nóng có áp suất có thể phun ra gây bỏng."}
BATTERY = {"en": "When disconnecting the battery: remove the negative (−) terminal first and reconnect it last; never let a metal tool touch both terminals.",
           "vi": "Khi tháo ắc quy: tháo cọc âm (−) trước, lắp cọc âm sau cùng; không để dụng cụ kim loại chạm cùng lúc hai cọc."}
FLASHING_MIL = {"en": "If the check-engine light is flashing: reduce load and stop soon; unburnt fuel can damage the catalytic converter.",
                "vi": "Nếu đèn check engine nhấp nháy: giảm tải và dừng xe sớm, nhiên liệu chưa cháy có thể làm hỏng bộ xúc tác."}
IGNITION_HV = {"en": "Switch the engine off before removing ignition coils or spark plugs: the ignition high voltage can give a shock.",
               "vi": "Tắt máy trước khi tháo bô-bin hoặc bugi: điện cao áp đánh lửa có thể gây giật."}
SAFETY = {
    "P0115-P0119": ("check", [HOT_COOLANT]),
    "P0128": ("check", [HOT_COOLANT]),
    "P0201-P0208": ("soon", [{"en": "The fuel line is under pressure: do not open it before relieving the pressure; if you smell or see leaking fuel, do not start the engine and keep sparks and flames away.",
                              "vi": "Đường nhiên liệu có áp suất: không tháo ống khi chưa xả áp; thấy mùi xăng hoặc rò xăng thì không nổ máy, tránh lửa và tia điện."}]),
    "P0217": ("stop", [{"en": "Engine overheating: stop in a safe place and switch the engine off.",
                        "vi": "Máy quá nhiệt: dừng xe ở chỗ an toàn và tắt máy."}, HOT_COOLANT]),
    "P0300": ("soon", [FLASHING_MIL]),
    "P0301-P0308": ("soon", [FLASHING_MIL, IGNITION_HV]),
    "P0351-P0358": ("soon", [IGNITION_HV]),
    "P0520-P0524": ("stop", [{"en": "If the red oil-pressure light is on with the engine running: stop the engine at once and check the oil level; driving on can destroy the engine.",
                              "vi": "Nếu đèn áp suất dầu (màu đỏ) sáng khi máy chạy: tắt máy ngay và kiểm tra mức dầu, chạy tiếp có thể phá hỏng động cơ."}]),
    "P0562": ("check", [BATTERY]),
    "P0600-P0606": ("check", [BATTERY]),
    "U0100": ("check", [BATTERY]),
}

# Codes for the Kotlin parity test: single codes, range ends, codes just outside ranges, hex digits, other letters, junk.
TEST_CODES = ["P0301", "p0171", "U0100", "U0101", "P0420", "P0430", "P0455", "P0216", "P0217", "P0A80", "B1234",
              "P0100", "P0103", "P0104", "P0130", "P0167", "P0168", "P0128", "P2135", "P0441", "P0450", "P0606",
              "P0", "X", "P03O1"]


def build(knowledge: dict, vi: dict, config: dict) -> dict:
    """Detectable components take the short EN/VI labels used on the photo (display_names); the others take the
    yaml's English name and the Vietnamese name from the companion file. DTC meanings: English from the yaml,
    Vietnamese keyed by the entry's codes joined with "/"."""
    classes = set(config["names"])
    components = {}
    for key, c in knowledge["components"].items():
        det = c.get("detector_class")
        if det and det not in classes:
            print(f"  note: {key}: detector_class {det} is not a class of {config['model_name']}, exported as null")
            det = None
        if det:
            name = dict(vi["display_names"][det])
        else:
            name = {"en": c["name"][:1].upper() + c["name"][1:], "vi": vi["components"][key]["name"]}
        desc = {"en": c["description"][:1].upper() + c["description"][1:], "vi": vi["components"][key]["description"]}
        components[key] = {"name": name, "description": desc, "detector_class": det}
    entries = []
    for d in knowledge["dtc"]:
        missing = [c for c in d["components"] if c not in components]
        if missing:
            raise SystemExit(f"dtc {d['codes']}: unknown components {missing}")
        vi_meaning = vi["dtc"].get("/".join(d["codes"]))
        if not vi_meaning:
            raise SystemExit(f"dtc {d['codes']}: no Vietnamese meaning in {VI_PATH.name}")
        urgency, safety = SAFETY.get(d["codes"][0], ("check", []))
        entries.append({"codes": d["codes"], "meaning": {"en": d["meaning"], "vi": vi_meaning},
                        "components": d["components"], "urgency": urgency, "safety": safety})
    unused = set(SAFETY) - {d["codes"][0] for d in knowledge["dtc"]}
    if unused:
        raise SystemExit(f"SAFETY keys match no dtc entry: {sorted(unused)}")
    return {
        "schema_version": 2,
        "source": "configs/diagnosis_knowledge.yaml via scripts/deployment/export_android_diagnosis.py",
        "draft": True,
        "note": "Generic OBD-II meanings for gasoline engines; urgency and safety notes are drafts to be reviewed.",
        "components": components,
        "dtc": entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", type=Path, default=APP)
    args = ap.parse_args()
    knowledge = load_knowledge()
    config = json.loads((args.app / "src/main/assets/config.json").read_text(encoding="utf-8"))
    vi = yaml.safe_load(VI_PATH.read_text(encoding="utf-8"))
    table = build(knowledge, vi, config)
    out = args.app / "src/main/assets/diagnosis.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_det = sum(1 for c in table["components"].values() if c["detector_class"])
    print(f"wrote {out.relative_to(ROOT)}: {len(table['dtc'])} dtc entries, {len(table['components'])} components "
          f"({n_det} detectable)")

    cases = []
    for code in TEST_CODES:
        matched, unknown = lookup_dtc([code], knowledge)
        cases.append({"input": code, "matched": [{"code": m["code"], "components": m["components"]} for m in matched],
                      "unknown": unknown})
    fx = args.app / "src/test/resources/fixtures/dtc_lookup.json"
    fx.parent.mkdir(parents=True, exist_ok=True)
    fx.write_text(json.dumps({"cases": cases}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {fx.relative_to(ROOT)}: {len(cases)} cases")


if __name__ == "__main__":
    main()
