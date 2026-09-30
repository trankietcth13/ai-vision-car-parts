"""TypeSafe Jev (System One) integration: text-only judgments for the diagnosis layer.

Jev takes text state + typed questions (Choice / Score / Noul) and returns probabilities.
It cannot see images: use it on complaints, DTC descriptions and procedure text, never on
pixels or on a VLM's own description of pixels (that audit had AUROC ~0.5).
"""
from .client import DEFAULT_MODEL, make_client, resolve_api_key
from .diagnosis_triage import TriageResult, load_knowledge, triage

__all__ = ["DEFAULT_MODEL", "make_client", "resolve_api_key", "TriageResult", "load_knowledge", "triage"]
