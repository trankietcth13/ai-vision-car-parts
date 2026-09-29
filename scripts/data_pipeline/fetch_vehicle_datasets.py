#!/usr/bin/env python3
"""Download vehicle-component datasets while preserving provenance.

The script is intentionally conservative: sources that require a form, a
commercial agreement, or an explicit research-only acknowledgement are listed
in SOURCES_NOTES.md but are never scraped or downloaded automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "vehicle_datasets.json"
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "vehicle_components"
STATE_FILE = "FETCH_STATE.json"
NOTES_FILE = "SOURCES_NOTES.md"
USER_AGENT = "vehicle-component-dataset-fetcher/1.0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def human_size(value: int | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def replace_with_retry(source: Path, destination: Path, attempts: int = 15) -> None:
    """Rename a completed download, tolerating short antivirus locks on Windows."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(1)


def compact_file_records(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact large snapshots while retaining a reproducible manifest hash."""
    if len(files) <= 100:
        return files
    digest = hashlib.sha256()
    total_bytes = 0
    paths = []
    for item in sorted(files, key=lambda value: value["path"]):
        total_bytes += int(item.get("bytes", 0))
        paths.append(item["path"])
        record = f"{item['path']}\0{item.get('bytes', 0)}\0{item.get('sha256', '')}\n"
        digest.update(record.encode("utf-8"))
    return [
        {
            "path": os.path.commonpath(paths),
            "bytes": total_bytes,
            "file_count": len(files),
            "manifest_sha256": digest.hexdigest(),
        }
    ]


def load_state(output: Path) -> dict[str, Any]:
    path = output / STATE_FILE
    if not path.exists():
        return {"schema_version": 1, "updated_at": utc_now(), "datasets": {}}
    return load_json(path)


def save_state(output: Path, state: dict[str, Any]) -> None:
    for entry in state.get("datasets", {}).values():
        files = entry.get("files")
        if isinstance(files, list):
            entry["files"] = compact_file_records(files)
    state["updated_at"] = utc_now()
    save_json_atomic(output / STATE_FILE, state)


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_image_file(path: Path, include_extensionless: bool = False) -> bool:
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return True
    if path.suffix or not include_extensionless:
        return False
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return False
    return (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith((b"II*\x00", b"MM\x00*", b"BM"))
        or (len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP")
    )


def download_url(url: str, destination: Path) -> Path:
    """Download URL with best-effort HTTP range resume to a .part file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"  already downloaded: {destination}")
        return destination
    partial = destination.with_name(destination.name + ".part")

    # curl handles GitHub/Azure/Dropbox redirect chains more reliably than
    # urllib on some managed Windows networks. It also provides robust resume.
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        command = [
            curl,
            "--location",
            "--fail",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--continue-at",
            "-",
            "--user-agent",
            USER_AGENT,
            "--output",
            str(partial),
            url,
        ]
        subprocess.run(command, check=True)
        replace_with_retry(partial, destination)
        return destination

    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and partial.exists():
            replace_with_retry(partial, destination)
            return destination
        raise

    status = getattr(response, "status", response.getcode())
    append = existing > 0 and status == 206
    if existing and not append:
        existing = 0
    mode = "ab" if append else "wb"
    remaining = response.headers.get("Content-Length")
    total = existing + int(remaining) if remaining and remaining.isdigit() else None
    received = existing
    last_report = 0.0

    with response, partial.open(mode) as handle:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            received += len(chunk)
            now = time.monotonic()
            if now - last_report >= 1.0:
                suffix = f"/{human_size(total)}" if total else ""
                print(f"  {destination.name}: {human_size(received)}{suffix}", flush=True)
                last_report = now

    replace_with_retry(partial, destination)
    return destination


def validate_file(path: Path, dataset: dict[str, Any]) -> dict[str, Any]:
    actual_size = path.stat().st_size
    expected_size = dataset.get("expected_size_bytes")
    # Hosts sometimes report rounded sizes. A manifest size is advisory unless a
    # cryptographic checksum is also supplied.
    if expected_size and dataset.get("expected_sha256"):
        if actual_size != int(expected_size):
            print(
                f"  warning: size {human_size(actual_size)} differs from manifest "
                f"{human_size(int(expected_size))}"
            )

    sha256 = digest_file(path, "sha256")
    expected_sha256 = dataset.get("expected_sha256")
    if expected_sha256 and sha256.lower() != expected_sha256.lower():
        raise ValueError(f"SHA256 mismatch for {path.name}")

    expected_md5 = dataset.get("expected_md5")
    md5 = None
    if expected_md5:
        md5 = digest_file(path, "md5")
        if md5.lower() != expected_md5.lower():
            raise ValueError(f"MD5 mismatch for {path.name}")

    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": actual_size,
        "sha256": sha256,
    }
    if md5:
        result["md5"] = md5
    return result


def safe_member_path(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Unsafe archive member: {member_name}") from exc
    return target


def extract_archive(archive: Path, destination: Path) -> None:
    marker = destination / ".extract_complete.json"
    if marker.exists():
        print(f"  extracted already: {destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                safe_member_path(destination, member.filename)
            bundle.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            for member in members:
                safe_member_path(destination, member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive links are not allowed: {member.name}")
            bundle.extractall(destination, members=members)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")

    save_json_atomic(
        marker,
        {
            "archive": str(archive.resolve()),
            "archive_sha256": digest_file(archive),
            "extracted_at": utc_now(),
        },
    )


def fetch_url_dataset(
    dataset: dict[str, Any], output: Path, extract: bool
) -> list[dict[str, Any]]:
    target_dir = output / dataset["id"]
    archive = download_url(dataset["download_url"], target_dir / dataset["filename"])
    file_info = validate_file(archive, dataset)
    if extract and dataset.get("archive"):
        extract_archive(archive, target_dir / "extracted")
    return [file_info]


def fetch_huggingface_dataset(
    dataset: dict[str, Any], output: Path, extract: bool
) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face source requires: pip install huggingface_hub"
        ) from exc

    target_dir = output / dataset["id"]
    target_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": dataset["repo_id"],
        "repo_type": "dataset",
        "local_dir": str(target_dir),
    }
    if dataset.get("revision"):
        kwargs["revision"] = dataset["revision"]
    if dataset.get("allow_patterns"):
        kwargs["allow_patterns"] = dataset["allow_patterns"]
    snapshot_download(**kwargs)

    files: list[dict[str, Any]] = []
    candidates = [path for path in target_dir.rglob("*") if path.is_file()]
    candidates = [path for path in candidates if ".cache" not in path.parts]
    for path in sorted(candidates):
        info = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": digest_file(path),
        }
        files.append(info)

    archive_member = dataset.get("archive_member")
    if archive_member:
        archive = target_dir / archive_member
        if not archive.exists():
            raise FileNotFoundError(f"Expected Hugging Face file missing: {archive}")
        validate_file(archive, dataset)
        if extract:
            extract_archive(archive, target_dir / "extracted")
    elif extract:
        archives = [
            path
            for path in candidates
            if zipfile.is_zipfile(path) or tarfile.is_tarfile(path)
        ]
        for archive in archives:
            extract_archive(archive, target_dir / "extracted" / archive.stem)
    return files


def fetch_roboflow_dataset(
    dataset: dict[str, Any], output: Path, extract: bool
) -> list[dict[str, Any]]:
    """Download an authenticated public Roboflow version in a training format."""
    del extract  # Roboflow's SDK downloads and extracts the generated archive.
    api_key_env = dataset.get("api_key_env", "ROBOFLOW_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Roboflow requires an API key. Set {api_key_env} in the environment; "
            "the key is never stored in the provenance files."
        )
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            'Roboflow source requires the optional package: pip install "roboflow>=1.1"'
        ) from exc

    target_dir = output / dataset["id"]
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(dataset["workspace"]).project(dataset["project"])
    version = project.version(int(dataset["version"]))
    version.download(
        dataset.get("format", "yolov8"),
        location=str(target_dir),
        overwrite=bool(dataset.get("overwrite", False)),
    )

    candidates = [path for path in target_dir.rglob("*") if path.is_file()]
    if not any(path.suffix.lower() in IMAGE_EXTENSIONS for path in candidates):
        raise RuntimeError(
            f"Roboflow returned no images in {target_dir}. Check the project/version "
            "and remove an incomplete empty directory before retrying."
        )
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": digest_file(path),
        }
        for path in sorted(candidates)
    ]


def fetch_zenodo_dataset(
    dataset: dict[str, Any], output: Path, extract: bool
) -> list[dict[str, Any]]:
    api_url = f"https://zenodo.org/api/records/{dataset['record_id']}"
    request = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)

    target_dir = output / dataset["id"]
    target_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(target_dir / "zenodo_record.json", metadata)
    downloaded: list[dict[str, Any]] = []
    for item in metadata.get("files", []):
        name = item.get("key") or item.get("filename")
        links = item.get("links", {})
        url = links.get("download") or links.get("self")
        if not name or not url:
            continue
        path = download_url(url, target_dir / name)
        upstream_checksum = item.get("checksum") or ""
        if upstream_checksum.startswith("md5:"):
            expected_md5 = upstream_checksum.split(":", 1)[1]
            actual_md5 = digest_file(path, "md5")
            if actual_md5.lower() != expected_md5.lower():
                raise ValueError(f"Zenodo MD5 mismatch for {name}")
        info = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": digest_file(path),
            "upstream_checksum": upstream_checksum or None,
        }
        downloaded.append(info)
        if extract and (zipfile.is_zipfile(path) or tarfile.is_tarfile(path)):
            extract_archive(path, target_dir / "extracted" / path.stem)
    return downloaded


def find_yolo_annotation(image: Path) -> Path | None:
    """Find the common YOLO label path corresponding to an image."""
    candidates = [image.with_suffix(".txt")]
    if image.parent.name.lower() == "images":
        candidates.append(image.parent.parent / "labels" / f"{image.stem}.txt")
    parts = list(image.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            remapped = parts.copy()
            remapped[index] = "labels"
            candidates.append(Path(*remapped).with_suffix(".txt"))
    return next((path for path in candidates if path.is_file()), None)


def write_dataset_provenance(
    output: Path, dataset: dict[str, Any]
) -> dict[str, Any]:
    """Create dataset- and image-level source records next to downloaded data."""
    target_dir = output / dataset["id"]
    retrieved_at = utc_now()
    source_record = {
        "schema_version": 1,
        "dataset_id": dataset["id"],
        "name": dataset["name"],
        "source_page": dataset["source_page"],
        "license": dataset.get("license"),
        "citation": dataset.get("citation"),
        "terms_note": dataset.get("terms_note"),
        "classes": dataset.get("classes"),
        "upstream_revision": dataset.get("revision") or dataset.get("version"),
        "retrieved_at": retrieved_at,
    }
    source_path = target_dir / "DATASET_SOURCE.json"
    save_json_atomic(source_path, source_record)

    include_extensionless = bool(dataset.get("include_extensionless_images", False))
    images = sorted(
        path
        for path in target_dir.rglob("*")
        if path.is_file()
        and is_image_file(path, include_extensionless=include_extensionless)
        and ".cache" not in path.parts
    )
    manifest_path = target_dir / "IMAGE_PROVENANCE.jsonl"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target_dir, delete=False, suffix=".tmp"
    ) as handle:
        for image in images:
            annotation = find_yolo_annotation(image)
            record: dict[str, Any] = {
                "dataset_id": dataset["id"],
                "image_path": str(image.relative_to(target_dir)),
                "sha256": digest_file(image),
                "source_page": dataset["source_page"],
                "license": dataset.get("license"),
                "retrieved_at": retrieved_at,
            }
            if annotation:
                record["annotation_path"] = str(annotation.relative_to(target_dir))
            if dataset.get("folder_labels") and image.parent != target_dir:
                record["class_hint"] = image.parent.name
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temp_name = handle.name
    replace_with_retry(Path(temp_name), manifest_path)
    return {
        "dataset_source": str(source_path.resolve()),
        "image_manifest": str(manifest_path.resolve()),
        "image_count": len(images),
    }


def markdown_cell(value: Any) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ")


def write_notes(
    output: Path, manifest_path: Path, datasets: Iterable[dict[str, Any]], state: dict[str, Any]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    engine_bay_manifest = manifest_path.name == "engine_bay_datasets.json"
    engine_assembly_manifest = manifest_path.name == "engine_assembly_zenodo.json"
    generator_script = (
        "scripts/data_pipeline/fetch_engine_bay_datasets.py"
        if engine_bay_manifest
        else (
            "scripts/data_pipeline/fetch_engine_assembly_zenodo.py"
            if engine_assembly_manifest
            else "scripts/data_pipeline/fetch_vehicle_datasets.py"
        )
    )
    rows = []
    for dataset in datasets:
        entry = state.get("datasets", {}).get(dataset["id"], {})
        status = entry.get("status", "not_fetched")
        source = f"[{dataset['source_page']}]({dataset['source_page']})"
        rows.append(
            "| {id} | {group} | {status} | {scope} | {source} | {license} |".format(
                id=markdown_cell(dataset["id"]),
                group=markdown_cell(dataset.get("group")),
                status=markdown_cell(status),
                scope=markdown_cell(dataset.get("scope")),
                source=source,
                license=markdown_cell(dataset.get("license")),
            )
        )

    details = []
    for dataset in datasets:
        entry = state.get("datasets", {}).get(dataset["id"], {})
        details.extend(
            [
                f"### {dataset['name']} (`{dataset['id']}`)",
                "",
                f"- Source: {dataset['source_page']}",
                f"- Fetch mode: `{dataset['handler']}`",
                f"- License/terms: {dataset.get('license', 'Chua ro')}",
                f"- Important: {dataset.get('terms_note', '-')}",
                f"- Citation: {dataset.get('citation', '-')}",
                f"- Local status: `{entry.get('status', 'not_fetched')}`",
            ]
        )
        if entry.get("fetched_at"):
            details.append(f"- Fetched at (UTC): {entry['fetched_at']}")
        provenance = entry.get("provenance")
        if provenance:
            details.append(
                f"- Image provenance: `{provenance.get('image_manifest')}` "
                f"({provenance.get('image_count', 0)} images)"
            )
        for item in entry.get("files", []):
            if item.get("file_count"):
                details.append(
                    f"- Snapshot: `{item['path']}`; {item['file_count']} files; "
                    f"{human_size(item.get('bytes'))}; manifest SHA256 "
                    f"`{item.get('manifest_sha256', '-')}`"
                )
            else:
                details.append(
                    f"- File: `{item['path']}`; {human_size(item.get('bytes'))}; "
                    f"SHA256 `{item.get('sha256', '-')}`"
                )
        if entry.get("error"):
            details.append(f"- Last error: `{entry['error']}`")
        details.append("")

    if engine_bay_manifest:
        usage_notes = [
            "- Roboflow is the bounding-box source; audit its numeric placeholder labels before training.",
            "- The Toyota Corolla source contains folder-level classes, not bounding boxes or masks.",
            "- Use `configs/engine_bay_class_map.yaml`; ambiguous labels are intentionally not auto-mapped.",
            "- Google Images is not scraped because image-level training rights and provenance are not uniform.",
        ]
        command_lines = [
            "# Inventory only; no download",
            f"python {generator_script} --list",
            "",
            "# Hugging Face images (about 5.56 GB, no API key)",
            f"python {generator_script} --datasets arabeitak_toyota_corolla",
            "",
            "# Roboflow YOLO labels (requires ROBOFLOW_API_KEY)",
            '$env:ROBOFLOW_API_KEY="YOUR_PRIVATE_KEY"',
            f"python {generator_script} --datasets engine_bay_parts_roboflow",
        ]
    elif engine_assembly_manifest:
        usage_notes = [
            "- The Zenodo record contains 195 captured real images with boxes, keypoints and polygons.",
            "- Around 280,000 augmented images are reproducible outputs, not independent real samples.",
            "- Keep augmentation descendants in the same split as their source image to prevent leakage.",
            "- The downloader verifies Zenodo MD5 checksums and records per-image SHA-256 provenance.",
        ]
        command_lines = [
            "# Download, verify, safely extract and build provenance",
            f"python {generator_script} --extract",
            "",
            "# Inventory only; no download",
            f"python {generator_script} --list",
        ]
    else:
        usage_notes = [
            "- `starter` is the suggested first download set; `extended` is larger or mainly useful for pretraining.",
            "- `manual` sources are never scraped. Follow the upstream form, account, research-only, or commercial-license flow.",
            "- Do not assume that a code repository license automatically covers its image dataset.",
            "- Check duplicate images across Carparts-Seg and DSMLR before merging, then remap all labels into one versioned ontology.",
            "- Keep component labels separate from damage labels so one region can carry both part and damage information.",
        ]
        command_lines = [
            "# Inventory only; no download",
            f"python {generator_script} --list",
            "",
            "# Suggested first set (about 3.5+ GiB), with safe archive extraction",
            f"python {generator_script} --datasets starter --extract",
            "",
            "# Larger complementary/pretraining sources",
            f"python {generator_script} --datasets extended --extract --continue-on-error",
            "",
            "# One source only",
            f"python {generator_script} --datasets carparts_seg --extract",
        ]

    content = "\n".join(
        [
            "# Vehicle-component dataset sources",
            "",
            f"> Generated by `{generator_script}`. Do not remove this file when moving the dataset.",
            "> License labels below are discovery notes, not legal advice. Re-check the upstream terms before commercial use or redistribution.",
            "",
            f"- Generated at (UTC): {utc_now()}",
            f"- Manifest: `{manifest_path.resolve()}`",
            f"- Download root: `{output.resolve()}`",
            "",
            "## Source inventory",
            "",
            "| Dataset ID | Group | Local status | Scope | Upstream source | License/terms |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
            "## Usage notes",
            "",
            *usage_notes,
            "",
            "## Commands",
            "",
            "```powershell",
            *command_lines,
            "```",
            "",
            "## Per-source provenance",
            "",
            *details,
        ]
    )
    (output / NOTES_FILE).write_text(content.rstrip() + "\n", encoding="utf-8")


def select_datasets(
    datasets: list[dict[str, Any]], selections: list[str]
) -> list[dict[str, Any]]:
    by_id = {item["id"]: item for item in datasets}
    selected_ids: list[str] = []
    for selection in selections:
        for token in selection.split(","):
            token = token.strip()
            if not token:
                continue
            if token == "all":
                selected_ids.extend(by_id)
            elif token == "all-auto":
                selected_ids.extend(
                    item["id"] for item in datasets if item["handler"] != "manual"
                )
            elif token in {"starter", "extended", "manual"}:
                selected_ids.extend(
                    item["id"] for item in datasets if item.get("group") == token
                )
            elif token in by_id:
                selected_ids.append(token)
            else:
                raise ValueError(f"Unknown dataset/group: {token}")
    # Stable de-duplication in manifest order.
    wanted = set(selected_ids)
    return [item for item in datasets if item["id"] in wanted]


def print_inventory(datasets: list[dict[str, Any]], state: dict[str, Any]) -> None:
    print(f"{'ID':30} {'GROUP':10} {'MODE':12} {'SIZE':>11} STATUS")
    for dataset in datasets:
        status = state.get("datasets", {}).get(dataset["id"], {}).get(
            "status", "not_fetched"
        )
        print(
            f"{dataset['id']:30} {dataset.get('group', '-'):10} "
            f"{dataset['handler']:12} "
            f"{human_size(dataset.get('expected_size_bytes')):>11} {status}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch vehicle-component datasets and preserve source notes."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="List sources; download nothing")
    parser.add_argument(
        "--datasets",
        action="append",
        default=[],
        metavar="ID_OR_GROUP",
        help="Dataset ID or starter/extended/manual/all-auto/all; comma-separated is accepted",
    )
    parser.add_argument(
        "--extract", action="store_true", help="Safely extract downloaded archives"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue fetching other sources after a failure",
    )
    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="Regenerate SOURCES_NOTES.md without downloading",
    )
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help="Rebuild source and per-image provenance for already downloaded data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_json(args.manifest)
    datasets: list[dict[str, Any]] = manifest["datasets"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = load_state(output)
    write_notes(output, args.manifest, datasets, state)

    if args.list:
        print_inventory(datasets, state)
        print(f"\nSource notes: {output / NOTES_FILE}")
        return 0
    if args.notes_only:
        print(f"Updated: {output / NOTES_FILE}")
        return 0
    if not args.datasets:
        print_inventory(datasets, state)
        print("\nNothing downloaded. Pass --datasets starter (or an explicit dataset ID).")
        print(f"Source notes: {output / NOTES_FILE}")
        return 0

    try:
        selected = select_datasets(datasets, args.datasets)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.provenance_only:
        for dataset in selected:
            target = output / dataset["id"]
            if not target.is_dir():
                print(f"error: dataset folder not found: {target}", file=sys.stderr)
                return 1
            provenance = write_dataset_provenance(output, dataset)
            entry = state["datasets"].setdefault(dataset["id"], {})
            entry.update(
                {
                    "source_page": dataset["source_page"],
                    "provenance": provenance,
                    "provenance_updated_at": utc_now(),
                }
            )
            print(
                f"[{dataset['id']}] provenance rebuilt: "
                f"{provenance['image_count']} image(s)"
            )
        save_state(output, state)
        write_notes(output, args.manifest, datasets, state)
        return 0

    failures = 0
    handlers = {
        "url": fetch_url_dataset,
        "huggingface": fetch_huggingface_dataset,
        "roboflow": fetch_roboflow_dataset,
        "zenodo": fetch_zenodo_dataset,
    }
    for dataset in selected:
        print(f"\n[{dataset['id']}] {dataset['name']}")
        if dataset["handler"] == "manual":
            print(f"  manual action required: {dataset['source_page']}")
            print(f"  {dataset.get('terms_note', '')}")
            state["datasets"][dataset["id"]] = {
                "status": "manual_required",
                "checked_at": utc_now(),
                "source_page": dataset["source_page"],
            }
            save_state(output, state)
            write_notes(output, args.manifest, datasets, state)
            continue

        try:
            files = handlers[dataset["handler"]](dataset, output, args.extract)
            provenance = write_dataset_provenance(output, dataset)
            state["datasets"][dataset["id"]] = {
                "status": "fetched",
                "fetched_at": utc_now(),
                "source_page": dataset["source_page"],
                "files": files,
                "provenance": provenance,
            }
            print(f"  complete: {len(files)} file(s)")
        except Exception as exc:  # keep provenance even for interrupted sources
            failures += 1
            state["datasets"][dataset["id"]] = {
                "status": "failed",
                "checked_at": utc_now(),
                "source_page": dataset["source_page"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                save_state(output, state)
                write_notes(output, args.manifest, datasets, state)
                return 1
        finally:
            save_state(output, state)
            write_notes(output, args.manifest, datasets, state)

    print(f"\nSource notes: {output / NOTES_FILE}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
