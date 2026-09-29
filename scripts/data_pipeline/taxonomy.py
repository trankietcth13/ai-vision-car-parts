"""Engine-bay taxonomy v2 (configs/taxonomy_v2.yaml): system -> fine component -> generic fallback.

Training classes are derived, never hand-listed:
    tier A components (in YAML order), then the generic classes that at least one tier-B component
    falls back to (in `generic_classes` order).
A tier-B component is trained under its fallback generic class, or ignored when the fallback is null.
Annotation names (36-class ontology, review verdicts) resolve to a fine component through `ann_names`
(default: the component's own name).

    from data_pipeline.taxonomy import load_taxonomy
    tax = load_taxonomy()
    tax.training_names            # ['battery', ..., 'other_reservoir', ...]
    tax.train_class_for("radiator_hose_upper")   # 'radiator_hose'
    tax.train_class_for("power_steering_reservoir")  # 'other_reservoir'
    tax.system_of("maf_sensor")   # 'air_intake'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "configs" / "taxonomy_v2.yaml"


@dataclass
class Component:
    name: str
    system: str
    tier: str
    fallback: Optional[str]
    vi: str = ""
    en: str = ""
    cues: List[str] = field(default_factory=list)
    confusers: Dict[str, str] = field(default_factory=dict)
    ann_names: List[str] = field(default_factory=list)


class Taxonomy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.version = cfg.get("version")
        self.promotion = cfg.get("promotion", {})
        self.systems: Dict[str, dict] = cfg["systems"]
        self.generic: Dict[str, dict] = cfg.get("generic_classes", {})
        self.detail_classes: List[str] = list(cfg.get("detail_classes", []))
        self.components: Dict[str, Component] = {}
        for name, c in cfg["components"].items():
            self.components[name] = Component(
                name=name,
                system=c["system"],
                tier=str(c["tier"]).upper(),
                fallback=c.get("fallback"),
                vi=c.get("vi", ""),
                en=c.get("en", ""),
                cues=list(c.get("cues", [])),
                confusers=dict(c.get("confusers", {}) or {}),
                ann_names=list(c.get("ann_names", [name])),
            )
        self._validate()
        self._ann_index = {a: comp.name for comp in self.components.values() for a in comp.ann_names}
        tier_a = [n for n, c in self.components.items() if c.tier == "A"]
        used_generic = {c.fallback for c in self.components.values() if c.tier == "B" and c.fallback}
        self.training_names: List[str] = tier_a + [g for g in self.generic if g in used_generic]
        self.train_id = {n: i for i, n in enumerate(self.training_names)}

    # ------------------------------------------------------------------ checks
    def _validate(self) -> None:
        errors = []
        seen_ann: Dict[str, str] = {}
        for comp in self.components.values():
            if comp.system not in self.systems:
                errors.append(f"{comp.name}: unknown system '{comp.system}'")
            if comp.tier not in ("A", "B"):
                errors.append(f"{comp.name}: tier must be A or B, got '{comp.tier}'")
            if comp.fallback is not None and comp.fallback not in self.generic:
                errors.append(f"{comp.name}: unknown fallback '{comp.fallback}'")
            if comp.name in self.generic:
                errors.append(f"{comp.name}: component name collides with a generic class")
            for a in comp.ann_names:
                if a in seen_ann:
                    errors.append(f"ann_name '{a}' used by both {seen_ann[a]} and {comp.name}")
                seen_ann[a] = comp.name
        if errors:
            raise ValueError("taxonomy_v2 invalid:\n  " + "\n  ".join(errors))

    # ------------------------------------------------------------------ lookups
    def component_for(self, ann_name: str) -> Optional[Component]:
        """Fine component for an annotation / verdict class name (None when unknown)."""
        key = self._ann_index.get(ann_name, ann_name)
        return self.components.get(key)

    def train_class_for(self, ann_name: str) -> Optional[str]:
        """Training class for an annotation name: itself (tier A), its fallback (tier B) or None."""
        comp = self.component_for(ann_name)
        if comp is None:
            return ann_name if ann_name in self.generic else None
        if comp.tier == "A":
            return comp.name
        return comp.fallback

    def train_id_for(self, ann_name: str) -> Optional[int]:
        cls = self.train_class_for(ann_name)
        return None if cls is None else self.train_id[cls]

    def system_of(self, name: str) -> Optional[str]:
        comp = self.component_for(name)
        return comp.system if comp else None

    def members_of_generic(self, generic: str) -> List[str]:
        """Tier-B components currently trained under a generic class."""
        return [c.name for c in self.components.values() if c.tier == "B" and c.fallback == generic]

    def by_system(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {s: [] for s in self.systems}
        for c in self.components.values():
            out[c.system].append(c.name)
        return out

    def yolo_names(self) -> Dict[int, str]:
        return dict(enumerate(self.training_names))


def load_taxonomy(path: Path | str = DEFAULT_PATH) -> Taxonomy:
    with open(path, encoding="utf-8") as f:
        return Taxonomy(yaml.safe_load(f))


if __name__ == "__main__":
    tax = load_taxonomy()
    print(f"taxonomy v{tax.version}: {len(tax.components)} components, {len(tax.systems)} systems")
    print(f"training classes ({len(tax.training_names)}):")
    for i, n in enumerate(tax.training_names):
        extra = f"  <- {', '.join(tax.members_of_generic(n))}" if n in tax.generic else ""
        print(f"  {i:2d} {n}{extra}")
    ignored = [c.name for c in tax.components.values() if c.tier == "B" and not c.fallback]
    print(f"ignored until promoted ({len(ignored)}): {', '.join(ignored)}")
