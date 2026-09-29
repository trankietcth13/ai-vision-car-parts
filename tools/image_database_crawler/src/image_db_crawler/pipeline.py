from __future__ import annotations

import io
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image

from .classifiers import ImageClassifier
from .models import AuctionListing
from .storage import DatasetStore


class CrawlPipeline:
    def __init__(
        self,
        classifier: ImageClassifier,
        output: Path,
        delay_seconds: float = 1.0,
        min_width: int = 640,
        min_height: int = 480,
        user_agent: str = "VehicleImageResearchCrawler/0.2 (local research)",
    ) -> None:
        self.classifier = classifier
        self.store = DatasetStore(output)
        self.delay_seconds = delay_seconds
        self.min_width = min_width
        self.min_height = min_height
        self.client = httpx.Client(
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": user_agent, "Accept": "image/*"},
        )

    def run(self, listings: list[AuctionListing], max_images_per_listing: int | None = None) -> dict[str, int]:
        stats = {"candidates": 0, "accepted": 0, "duplicates": 0, "rejected": 0, "errors": 0}
        try:
            for listing in listings:
                images = listing.images[:max_images_per_listing] if max_images_per_listing else listing.images
                for candidate in images:
                    stats["candidates"] += 1
                    try:
                        response = self.client.get(candidate.url, headers={"Referer": listing.url})
                        response.raise_for_status()
                        content = response.content
                        image = Image.open(io.BytesIO(content))
                        if image.width < self.min_width or image.height < self.min_height:
                            stats["rejected"] += 1
                            continue
                        result = self.classifier.classify(content)
                        if not result.accepted:
                            stats["rejected"] += 1
                            continue
                        extension = _extension(image.format, candidate.url)
                        vehicle = candidate.vehicle or listing.vehicle
                        saved = self.store.save(
                            content,
                            extension,
                            vehicle,
                            result,
                            candidate.url,
                            candidate.source_page or listing.url,
                            source_name=candidate.source_name,
                            license_code=candidate.license_code,
                            license_url=candidate.license_url,
                            creator=candidate.creator,
                            creator_url=candidate.creator_url,
                            attribution=candidate.attribution,
                            component_hint=candidate.component_hint,
                        )
                        stats["accepted" if saved else "duplicates"] += 1
                    except Exception as exc:
                        stats["errors"] += 1
                        print(f"[WARN] {candidate.url}: {exc}")
                    finally:
                        time.sleep(self.delay_seconds)
        finally:
            self.client.close()
            self.store.close()
        return stats


def _extension(image_format: str | None, url: str) -> str:
    mapping = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    if image_format in mapping:
        return mapping[image_format]
    suffix = Path(urlsplit(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
