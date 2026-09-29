#!/usr/bin/env python3
"""Download and verify the Engine Assembly Dataset from Zenodo.

By default, data is stored at H:/AI_Datasets/engine_assembly_zenodo.
Use --output to select another dataset root.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fetch_vehicle_datasets import main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "configs" / "engine_assembly_zenodo.json"
DEFAULT_OUTPUT = Path("H:/AI_Datasets")
DATASET_ID = "engine_assembly_zenodo"


def has_option(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(f"{option}=") for value in arguments)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    defaults: list[str] = []
    if not has_option(arguments, "--manifest"):
        defaults.extend(["--manifest", str(MANIFEST)])
    if not has_option(arguments, "--output"):
        defaults.extend(["--output", str(DEFAULT_OUTPUT)])
    if not has_option(arguments, "--datasets") and "--list" not in arguments and "--notes-only" not in arguments:
        defaults.extend(["--datasets", DATASET_ID])
    raise SystemExit(main(defaults + arguments))
