from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VehicleMetadata:
    year: str = "unknown-year"
    make: str = "unknown-make"
    model: str = "unknown-model"
    engine: str = "unknown-engine"


@dataclass(frozen=True)
class CandidateImage:
    url: str
    alt: str = ""
    gallery: str = ""
    source_page: str = ""
    source_name: str = ""
    license_code: str = ""
    license_url: str = ""
    creator: str = ""
    creator_url: str = ""
    attribution: str = ""
    component_hint: str = ""
    vehicle: VehicleMetadata | None = None


@dataclass(frozen=True)
class AuctionListing:
    url: str
    vehicle: VehicleMetadata
    images: list[CandidateImage] = field(default_factory=list)


@dataclass(frozen=True)
class ClassificationResult:
    accepted: bool
    category: str = ""
    component_name: str = ""
    confidence: float = 0.0
    labels: tuple[str, ...] = ()
