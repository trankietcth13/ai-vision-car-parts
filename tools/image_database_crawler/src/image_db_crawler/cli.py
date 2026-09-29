from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from .carsandbids import CarsAndBidsBrowser
from .classifiers import JevApiClassifier, UltralyticsJevClassifier
from .pipeline import CrawlPipeline
from .sources import DEFAULT_QUERIES, IFIXIT_QUERIES, IFixitSource, OpenverseSource, WikimediaSource


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT.parent / "runs" / "segment" / "engine_teacher_v7" / "weights" / "best.pt"


def _urls(args: argparse.Namespace) -> list[str]:
    values = list(args.url or [])
    if args.url_file:
        values.extend(
            line.strip() for line in Path(args.url_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return list(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect license-aware vehicle images and classify them with Jev/YOLO")
    parser.add_argument(
        "--source",
        choices=("wikimedia", "openverse", "ifixit", "carsandbids"),
        default="wikimedia",
    )
    parser.add_argument("--query", action="append", help="Search query; repeat to use multiple queries")
    parser.add_argument("--category", action="append", help="Wikimedia Commons category; repeat as needed")
    parser.add_argument("--limit", type=int, default=100, help="Maximum candidate images discovered")
    parser.add_argument(
        "--contact",
        default=os.environ.get("CRAWLER_CONTACT"),
        help="Contact email or URL included in the API User-Agent (required by Wikimedia)",
    )
    parser.add_argument(
        "--allow-noncommercial",
        action="store_true",
        help="Acknowledge that iFixit CC BY-NC-SA data is restricted to noncommercial use",
    )
    parser.add_argument("--url", action="append", help="Auction URL; repeat for multiple listings")
    parser.add_argument("--url-file", type=Path, help="Text file containing one auction URL per line")
    parser.add_argument(
        "--authorization-file",
        type=Path,
        help="Written source authorization; required only for Cars & Bids",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--browser-profile", type=Path, default=PROJECT_ROOT / ".browser" / "carsandbids")
    parser.add_argument("--headless", action="store_true", help="Only use after the browser profile has passed Cloudflare")
    parser.add_argument("--challenge-timeout", type=int, default=180)
    parser.add_argument("--listing-limit", type=int)
    parser.add_argument("--max-images-per-listing", type=int)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between image requests")
    parser.add_argument("--classifier", choices=("yolo", "jev-api"), default="yolo")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Jev/YOLO checkpoint")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--jev-endpoint", help="Jev HTTP endpoint (or JEV_ENDPOINT)")
    parser.add_argument("--jev-api-key", help="Jev API key (or JEV_API_KEY)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    user_agent = f"VehicleImageResearchCrawler/0.2 ({args.contact or 'local research'})"
    listings = _collect(args, user_agent)
    candidate_count = sum(len(listing.images) for listing in listings)
    print(f"Discovered {candidate_count} candidate images from {args.source}")
    if args.classifier == "yolo":
        if not args.model.exists():
            raise SystemExit(f"Model not found: {args.model}")
        classifier = UltralyticsJevClassifier(args.model, args.confidence)
    else:
        endpoint = args.jev_endpoint or os.environ.get("JEV_ENDPOINT")
        if not endpoint:
            raise SystemExit("--jev-endpoint or JEV_ENDPOINT is required for jev-api")
        classifier = JevApiClassifier(endpoint, args.jev_api_key or os.environ.get("JEV_API_KEY"))
    stats = CrawlPipeline(classifier, args.output, args.delay, user_agent=user_agent).run(
        listings, args.max_images_per_listing
    )
    print("Done:", ", ".join(f"{key}={value}" for key, value in stats.items()))


def _collect(args: argparse.Namespace, user_agent: str):
    if args.source == "carsandbids":
        urls = _urls(args)
        if not urls:
            raise SystemExit("Cars & Bids requires at least one --url or --url-file")
        if not args.authorization_file or not args.authorization_file.is_file():
            raise SystemExit(
                "Cars & Bids Terms prohibit unauthorized automated extraction and AI/ML use. "
                "Obtain written permission, save it locally, then pass --authorization-file PATH."
            )
        browser = CarsAndBidsBrowser(args.browser_profile, args.headless, args.challenge_timeout)
        return asyncio.run(browser.collect(urls, args.listing_limit or args.limit))

    if args.source == "wikimedia":
        if not args.contact or not ("@" in args.contact or args.contact.startswith(("http://", "https://"))):
            raise SystemExit("Wikimedia requires --contact with a real email address or URL")
        source = WikimediaSource(user_agent)
        categories = args.category or ["Automobiles with open hoods"]
        try:
            return source.collect(categories, args.limit)
        finally:
            source.close()

    if args.source == "openverse":
        source = OpenverseSource(user_agent)
        try:
            return source.collect(args.query or DEFAULT_QUERIES, args.limit)
        finally:
            source.close()

    if not args.allow_noncommercial:
        raise SystemExit("iFixit is CC BY-NC-SA 3.0; pass --allow-noncommercial for noncommercial research")
    source = IFixitSource(user_agent)
    try:
        return source.collect(args.query or IFIXIT_QUERIES, args.limit)
    finally:
        source.close()


if __name__ == "__main__":
    main()
