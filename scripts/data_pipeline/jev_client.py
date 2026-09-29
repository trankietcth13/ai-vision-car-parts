"""Jev client for the data-pipeline scripts: uses src/jev (pinned model, shared key lookup) when present,
otherwise builds a TypeSafeClient itself. The key comes from TYPESAFE_API_KEY or the project .env; never printed."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-1.13.0")


def make_client(timeout: float = 30.0, model: str | None = None):
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from jev import make_client as shared
        return shared(model=model, timeout=timeout)
    except ImportError:
        pass
    from typesafe_sdk import TypeSafeClient
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and (ROOT / ".env").exists():
        from dotenv import dotenv_values
        values = dotenv_values(ROOT / ".env")
        found = [k for k, v in values.items() if v and re.search(r"(typesafe|jev).*key|key.*(typesafe|jev)", k, re.I)]
        key = values.get("TYPESAFE_API_KEY") or (values[found[0]] if len(found) == 1 else None)
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is not set (environment or project .env)")
    return TypeSafeClient(api_key=key, model=model or DEFAULT_MODEL, timeout=timeout)
