from __future__ import annotations

import asyncio
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

from .models import AuctionListing, CandidateImage, VehicleMetadata


AUCTION_URL_RE = re.compile(r"^https://(?:www\.)?carsandbids\.com/auctions/[A-Za-z0-9]+/[^?#]+")
MEDIA_HOST = "media.carsandbids.com"


def normalize_media_url(url: str) -> str | None:
    if url.startswith("//"):
        url = "https:" + url
    parts = urlsplit(url)
    if parts.hostname != MEDIA_HOST:
        return None
    path = unquote(parts.path)
    marker = "/cdn-cgi/image/"
    if marker in path:
        remainder = path.split(marker, 1)[1]
        slash = remainder.find("/")
        if slash >= 0:
            path = "/" + remainder[slash + 1 :].lstrip("/")
    if "/photos/" not in path:
        return None
    return urlunsplit(("https", MEDIA_HOST, path, parts.query, ""))


def parse_visible_metadata(text: str, title: str = "") -> VehicleMetadata:
    heading = next((line.strip() for line in text.splitlines() if re.match(r"^(19|20)\d{2}\s+", line.strip())), title)
    match = re.search(r"\b((?:19|20)\d{2})\s+(.+)", heading)
    year, make, model = "unknown-year", "unknown-make", "unknown-model"
    if match:
        year = match.group(1)
        remainder = re.sub(r"\s+(?:for Sale|auction).*$", "", match.group(2), flags=re.I).strip()
        fields = _label_values(text)
        make = fields.get("Make", "") or remainder.split(maxsplit=1)[0]
        model = fields.get("Model", "") or (remainder.split(maxsplit=1)[1] if " " in remainder else remainder)
    else:
        fields = _label_values(text)
        make, model = fields.get("Make", make), fields.get("Model", model)
    return VehicleMetadata(year, make, model, fields.get("Engine", "unknown-engine"))


def _label_values(text: str) -> dict[str, str]:
    wanted = {"Make", "Model", "Engine"}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    values: dict[str, str] = {}
    for index, line in enumerate(lines[:-1]):
        if line in wanted and lines[index + 1] not in wanted:
            values[line] = lines[index + 1]
    return values


class CarsAndBidsBrowser:
    def __init__(self, profile_dir: Path, headless: bool = False, challenge_timeout: int = 180) -> None:
        self.profile_dir = profile_dir
        self.headless = headless
        self.challenge_timeout = challenge_timeout

    async def collect(self, urls: list[str], limit: int | None = None) -> list[AuctionListing]:
        from playwright.async_api import async_playwright

        selected = [url for url in urls if AUCTION_URL_RE.match(url)]
        if limit is not None:
            selected = selected[:limit]
        listings: list[AuctionListing] = []
        async with async_playwright() as playwright:
            context = await playwright.chromium.launch_persistent_context(
                str(self.profile_dir),
                headless=self.headless,
                viewport={"width": 1440, "height": 1000},
                locale="en-US",
            )
            page = context.pages[0] if context.pages else await context.new_page()
            for url in selected:
                await page.goto(url, wait_until="domcontentloaded", timeout=90_000)
                await self._wait_for_site(page)
                await self._expand_gallery(page)
                listings.append(await self._parse_page(page, url))
            await context.close()
        return listings

    async def _wait_for_site(self, page) -> None:
        deadline = asyncio.get_running_loop().time() + self.challenge_timeout
        while "just a moment" in (await page.title()).lower():
            if self.headless or asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    "Cars & Bids Cloudflare challenge is active. Run without --headless and complete it in the opened browser."
                )
            await page.wait_for_timeout(1_000)
        await page.wait_for_load_state("networkidle", timeout=60_000)

    async def _expand_gallery(self, page) -> None:
        for pattern in (r"All Photos", r"View all photos", r"Show all photos"):
            locator = page.get_by_text(re.compile(pattern, re.I)).first
            if await locator.count():
                try:
                    await locator.click(timeout=3_000)
                    await page.wait_for_timeout(1_000)
                    break
                except Exception:
                    pass
        for _ in range(8):
            await page.mouse.wheel(0, 2400)
            await page.wait_for_timeout(250)

    async def _parse_page(self, page, url: str) -> AuctionListing:
        body = await page.locator("body").inner_text()
        vehicle = parse_visible_metadata(body, await page.title())
        raw = await page.evaluate(
            """
            () => {
              const rows = [];
              for (const img of document.images) {
                rows.push({url: img.currentSrc || img.src, alt: img.alt || '', gallery: img.closest('section')?.innerText?.slice(0, 80) || ''});
                for (const part of (img.srcset || '').split(',')) rows.push({url: part.trim().split(/\\s+/)[0], alt: img.alt || '', gallery: ''});
              }
              for (const a of document.querySelectorAll('a[href*="media.carsandbids.com"]')) rows.push({url: a.href, alt: '', gallery: ''});
              for (const item of performance.getEntriesByType('resource')) rows.push({url: item.name, alt: '', gallery: ''});
              return rows;
            }
            """
        )
        images: list[CandidateImage] = []
        seen: set[str] = set()
        for item in raw:
            normalized = normalize_media_url(str(item.get("url", "")))
            if normalized and normalized not in seen:
                seen.add(normalized)
                images.append(CandidateImage(normalized, str(item.get("alt", "")), str(item.get("gallery", ""))))
        return AuctionListing(url=url, vehicle=vehicle, images=images)

