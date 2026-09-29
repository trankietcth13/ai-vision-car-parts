from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote

import httpx

from .models import AuctionListing, CandidateImage, VehicleMetadata


DEFAULT_QUERIES = (
    "car engine bay",
    "automobile engine compartment",
    "car battery engine bay",
    "mass air flow sensor car",
    "car alternator engine bay",
)

IFIXIT_QUERIES = (
    "car battery replacement",
    "car alternator replacement",
    "car air filter replacement",
    "car coolant reservoir replacement",
    "car mass air flow sensor replacement",
    "car engine oil dipstick",
    "car fuse box",
)

KNOWN_MAKES = (
    "Alfa Romeo", "Aston Martin", "Mercedes-Benz", "Land Rover", "Rolls-Royce",
    "Chevrolet", "Volkswagen", "Mitsubishi", "Lamborghini", "Toyota", "Honda",
    "Hyundai", "Mazda", "Nissan", "Subaru", "Porsche", "Ferrari", "Pontiac",
    "Dodge", "Chrysler", "Cadillac", "Lexus", "Infiniti", "Acura", "Volvo",
    "Audi", "BMW", "Ford", "Kia", "Jeep", "GMC", "Fiat", "Mini", "Tesla",
    "Triumph",
)


def _plain(value: object) -> str:
    if isinstance(value, dict):
        value = value.get("value", "")
    text = html.unescape(str(value or ""))
    return re.sub(r"<[^>]+>", "", text).strip()


def infer_vehicle(*texts: str) -> VehicleMetadata:
    joined = " ".join(filter(None, texts)).replace("_", " ")
    year_match = re.search(r"\b(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)?\d{2})?\b", joined)
    year = re.sub(r"\s+", "", year_match.group(0)).replace("–", "-") if year_match else "unknown-year"
    words = re.findall(r"[A-Za-z0-9]+", joined)
    make = next((name for name in KNOWN_MAKES if re.search(rf"\b{re.escape(name)}\b", joined, re.I)), "unknown-make")
    compact_model = ""
    if make == "unknown-make":
        for name in KNOWN_MAKES:
            compact_make = re.sub(r"[^a-z0-9]", "", name.lower())
            tagged = next((word for word in words if word.lower().startswith(compact_make) and len(word) > len(compact_make)), "")
            if tagged:
                make = name
                compact_model = tagged[len(compact_make) :]
                break
    model = "unknown-model"
    if compact_model:
        model = compact_model
    elif make != "unknown-make":
        after = re.split(re.escape(make), joined, maxsplit=1, flags=re.I)[-1]
        after = re.split(re.escape(make), after, maxsplit=1, flags=re.I)[0]
        after = re.sub(r"\b(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)?\d{2})?\b", "", after)
        after = re.sub(r"\b(?:car|automobile|engine|bay|hood|replacement|repair|guide|with|open)\b.*$", "", after, flags=re.I)
        candidate = re.sub(r"[^A-Za-z0-9.+-]+", " ", after).strip()
        if candidate:
            model = " ".join(candidate.split()[:4])
    return VehicleMetadata(year, make, model, "unknown-engine")


class ApiSource:
    def __init__(self, user_agent: str, timeout: float = 45.0) -> None:
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
        )

    def close(self) -> None:
        self.client.close()


class OpenverseSource(ApiSource):
    API = "https://api.openverse.org/v1/images/"

    def collect(self, queries: Iterable[str], limit: int = 100) -> list[AuctionListing]:
        query_list = list(queries)
        quota = max(1, math.ceil(limit / max(1, len(query_list))))
        images: list[CandidateImage] = []
        seen: set[str] = set()
        for query in query_list:
            page = 1
            added = 0
            while len(images) < limit and added < quota:
                page_size = min(20, quota - added, limit - len(images))
                response = self.client.get(
                    self.API,
                    params={
                        "q": query,
                        "license": "cc0,by,by-sa,pdm",
                        "page_size": page_size,
                        "page": page,
                    },
                )
                response.raise_for_status()
                data = response.json()
                rows = data.get("results", [])
                if not rows:
                    break
                for row in rows:
                    url = str(row.get("url") or "")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    title = str(row.get("title") or query)
                    tags = " ".join(str(tag.get("name", "")) for tag in row.get("tags", []))
                    images.append(
                        CandidateImage(
                            url=url,
                            alt=title,
                            source_page=str(row.get("foreign_landing_url") or ""),
                            source_name="openverse",
                            license_code=str(row.get("license") or "").lower(),
                            license_url=str(row.get("license_url") or ""),
                            creator=str(row.get("creator") or ""),
                            creator_url=str(row.get("creator_url") or ""),
                            attribution=str(row.get("attribution") or ""),
                            component_hint=query,
                            vehicle=infer_vehicle(title, tags),
                        )
                    )
                    added += 1
                    if len(images) >= limit:
                        break
                page += 1
                if page > int(data.get("page_count", page)):
                    break
            if len(images) >= limit:
                break
        return [AuctionListing("https://openverse.org/", VehicleMetadata(), images)]


class WikimediaSource(ApiSource):
    API = "https://commons.wikimedia.org/w/api.php"

    def collect(self, categories: Iterable[str], limit: int = 200) -> list[AuctionListing]:
        images: list[CandidateImage] = []
        seen: set[str] = set()
        for raw_category in categories:
            category = raw_category if raw_category.lower().startswith("category:") else f"Category:{raw_category}"
            continuation: dict[str, str] = {}
            while len(images) < limit:
                params = {
                    "action": "query",
                    "generator": "categorymembers",
                    "gcmtitle": category,
                    "gcmtype": "file",
                    "gcmlimit": "50",
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata",
                    "format": "json",
                    "formatversion": "2",
                    **continuation,
                }
                response = self.client.get(self.API, params=params)
                response.raise_for_status()
                data = response.json()
                pages = data.get("query", {}).get("pages", [])
                for page in pages:
                    info = (page.get("imageinfo") or [{}])[0]
                    meta = info.get("extmetadata") or {}
                    url = str(info.get("url") or "")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    title = str(page.get("title") or "").removeprefix("File:")
                    license_code = _plain(meta.get("LicenseShortName")).lower().replace(" ", "-")
                    if not _license_allowed(license_code):
                        continue
                    images.append(
                        CandidateImage(
                            url=url,
                            alt=_plain(meta.get("ImageDescription")) or title,
                            source_page=str(info.get("descriptionurl") or ""),
                            source_name="wikimedia-commons",
                            license_code=license_code,
                            license_url=_plain(meta.get("LicenseUrl")),
                            creator=_plain(meta.get("Artist")),
                            creator_url="",
                            attribution=_plain(meta.get("Credit")) or _plain(meta.get("Artist")),
                            component_hint=category.removeprefix("Category:"),
                            vehicle=infer_vehicle(title, category, _plain(meta.get("Categories"))),
                        )
                    )
                    if len(images) >= limit:
                        break
                if len(images) >= limit or "continue" not in data:
                    break
                continuation = {key: str(value) for key, value in data["continue"].items()}
        return [AuctionListing("https://commons.wikimedia.org/", VehicleMetadata(), images)]


class IFixitSource(ApiSource):
    API = "https://www.ifixit.com/api/2.0"
    LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/"

    def collect(self, queries: Iterable[str], limit: int = 100, guide_limit: int = 30) -> list[AuctionListing]:
        query_list = list(queries)
        per_query_guides = max(1, math.ceil(guide_limit / max(1, len(query_list))))
        guide_ids: list[int] = []
        for query in query_list:
            response = self.client.get(f"{self.API}/suggest/{quote(query, safe='')}")
            response.raise_for_status()
            added = 0
            for row in response.json().get("results", []):
                if row.get("dataType") == "guide" and int(row["guideid"]) not in guide_ids:
                    guide_ids.append(int(row["guideid"]))
                    added += 1
                    if len(guide_ids) >= guide_limit or added >= per_query_guides:
                        break
            if len(guide_ids) >= guide_limit:
                break

        images: list[CandidateImage] = []
        seen: set[str] = set()
        for guide_id in guide_ids:
            response = self.client.get(f"{self.API}/guides/{guide_id}")
            response.raise_for_status()
            guide = response.json()
            category = str(guide.get("category") or "")
            title = str(guide.get("title") or "")
            subject = str(guide.get("subject") or "")
            context = f"{category} {title} {subject}"
            if not _looks_automotive(context):
                continue
            author = guide.get("author") or {}
            creator = str(author.get("username") if isinstance(author, dict) else author)
            page_url = str(guide.get("url") or f"https://www.ifixit.com/Guide/{guide_id}")
            vehicle = infer_vehicle(category, title)
            for step in guide.get("steps", []):
                step_text = " ".join(str(line.get("text_raw") or "") for line in step.get("lines", []))
                media = step.get("media") or {}
                if media.get("type") != "image":
                    continue
                for item in media.get("data", []):
                    url = str(item.get("original") or item.get("huge") or item.get("large") or "")
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    images.append(
                        CandidateImage(
                            url=url,
                            alt=step_text or title,
                            source_page=page_url,
                            source_name="ifixit",
                            license_code="cc-by-nc-sa-3.0",
                            license_url=self.LICENSE_URL,
                            creator=creator,
                            attribution=f"{title} — {creator or 'iFixit contributors'} / iFixit",
                            component_hint=subject or title,
                            vehicle=vehicle,
                        )
                    )
                    if len(images) >= limit:
                        break
                if len(images) >= limit:
                    break
            if len(images) >= limit:
                break
        return [AuctionListing("https://www.ifixit.com/", VehicleMetadata(), images)]


def _looks_automotive(text: str) -> bool:
    automotive = re.search(r"\b(car|truck|automobile|vehicle|honda|toyota|ford|bmw|audi|mazda|nissan|subaru|kia|hyundai)\b", text, re.I)
    target = re.search(r"\b(battery|terminal|alternator|engine|coolant|radiator|sensor|air filter|oil|fuse)\b", text, re.I)
    return bool(automotive and target)


def _license_allowed(code: str) -> bool:
    normalized = code.lower().replace("_", "-")
    return normalized.startswith(("cc0", "cc-by-", "cc-by-sa-", "public-domain", "pd-")) or normalized in {
        "cc-by", "cc-by-sa", "pdm", "pd"
    }
