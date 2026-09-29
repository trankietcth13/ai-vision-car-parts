#!/usr/bin/env python3
"""Fetch curated engine-bay datasets and preserve image-level provenance.

Examples:
    python scripts/data_pipeline/fetch_engine_bay_datasets.py --list
    python scripts/data_pipeline/fetch_engine_bay_datasets.py --datasets arabeitak_toyota_corolla
    python scripts/data_pipeline/fetch_engine_bay_datasets.py --datasets engine_bay_parts_roboflow
"""

from __future__ import annotations

import sys
from pathlib import Path

from fetch_vehicle_datasets import main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "configs" / "engine_bay_datasets.json"
OUTPUT = PROJECT_ROOT / "datasets" / "engine_bay_sources"


def has_option(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    defaults: list[str] = []
    if not has_option(arguments, "--manifest"):
        defaults.extend(["--manifest", str(MANIFEST)])
    if not has_option(arguments, "--output"):
        defaults.extend(["--output", str(OUTPUT)])
    raise SystemExit(main(defaults + arguments))
