"""TypeSafe client factory. Credentials come from the environment or the project .env and are never printed."""
from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEY_ENV = "TYPESAFE_API_KEY"
# Pinned so tuned thresholds don't move when the `jev-latest` alias changes; override with TYPESAFE_DEFAULT_MODEL.
DEFAULT_MODEL = os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-1.13.0")


def resolve_api_key(env_file: Path | None = None) -> str:
    """Return the API key from TYPESAFE_API_KEY, else from a single *TYPESAFE*KEY* / *JEV*KEY* variable in .env."""
    if os.environ.get(KEY_ENV):
        return os.environ[KEY_ENV]
    env_file = env_file or PROJECT_ROOT / ".env"
    if env_file.exists():
        from dotenv import dotenv_values

        values = dotenv_values(env_file)
        if values.get(KEY_ENV):
            return values[KEY_ENV]
        found = [k for k, v in values.items() if v and re.search(r"(typesafe|jev).*key|key.*(typesafe|jev)", k, re.I)]
        if len(found) == 1:
            return values[found[0]]
        if len(found) > 1:
            raise RuntimeError(f"{len(found)} *TYPESAFE*KEY* variables in {env_file.name}: {found}; set {KEY_ENV}")
    raise RuntimeError(f"{KEY_ENV} is not set. Create a key at https://console.typesafe.ai/ and add "
                       f"{KEY_ENV}=... to {env_file}")


def make_client(model: str | None = None, timeout: float = 20.0):
    """Synchronous TypeSafeClient with the resolved key and the pinned model."""
    from typesafe_sdk import TypeSafeClient

    return TypeSafeClient(api_key=resolve_api_key(), model=model or DEFAULT_MODEL, timeout=timeout)
