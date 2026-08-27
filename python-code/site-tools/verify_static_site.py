"""Verify local website links, identifiers, and generated dataset contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


HTML_FILES = (
    "index.html",
    "about.html",
    "website/index.html",
    "website/about.html",
)


class ReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.identifiers: list[str] = []
        self.references: list[tuple[str, str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        identifier = attributes.get("id")
        if identifier:
            self.identifiers.append(identifier)
        for attribute in ("href", "src"):
            value = attributes.get(attribute)
            if value:
                self.references.append((tag, attribute, value))


def find_repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "website" / "index.html").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_html(root: Path, relative: str) -> list[str]:
    path = root / relative
    parser = ReferenceParser()
    parser.feed(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    duplicate_ids = sorted(
        identifier
        for identifier, count in Counter(parser.identifiers).items()
        if count > 1
    )
    if duplicate_ids:
        failures.append(f"{relative}: duplicate IDs {', '.join(duplicate_ids)}")

    for tag, attribute, value in parser.references:
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        decoded_path = unquote(parsed.path)
        if decoded_path.startswith("/"):
            target = root / decoded_path.lstrip("/")
        else:
            target = path.parent / decoded_path
        if decoded_path.endswith("/"):
            target /= "index.html"
        if not target.resolve().exists():
            failures.append(
                f"{relative}: missing {tag}[{attribute}] target {value}"
            )
    return failures


def parse_assignment(path: Path, name: str) -> object:
    text = path.read_text(encoding="utf-8-sig")
    prefix = f"window.{name} = "
    if not text.startswith(prefix) or not text.endswith(";\n"):
        raise RuntimeError(f"Unexpected JavaScript assignment format: {path}")
    return json.loads(text[len(prefix) : -2])


def verify_full_catalog_asset(root: Path) -> list[str]:
    asset_path = root / "website" / "assets" / "full-catalog-generalization-data.js"
    data = parse_assignment(asset_path, "GWTC5_FULL_CATALOG_DATA")
    if not isinstance(data, dict):
        return ["Full-catalog asset is not a JSON object."]
    expected_counts = {
        "primary_events": 78,
        "strict_events": 93,
        "census_events": 104,
        "generalization_tests": 6,
        "population_summaries": 2,
    }
    failures = [
        f"Full-catalog asset {name}: expected {expected}, found {len(data.get(name, []))}"
        for name, expected in expected_counts.items()
        if len(data.get(name, [])) != expected
    ]
    primary_positive = sum(
        float(row["delta_nw1"]) > 0 for row in data.get("primary_events", [])
    )
    census_positive = sum(
        row["delta_positive"] == "True" for row in data.get("census_events", [])
    )
    if primary_positive != 75:
        failures.append(f"Primary positive count: expected 75, found {primary_positive}")
    if census_positive != 99:
        failures.append(f"Census positive count: expected 99, found {census_positive}")

    capsule = root / "reproducibility" / "full-catalog-generalization"
    source_hashes = data.get("source_sha256", {})
    for relative, expected_hash in source_hashes.items():
        source = capsule / relative
        if not source.is_file():
            failures.append(f"Full-catalog website source is missing: {relative}")
        elif sha256_file(source) != expected_hash:
            failures.append(f"Full-catalog website source hash changed: {relative}")
    if len(source_hashes) != 5:
        failures.append(
            f"Full-catalog asset should authenticate 5 sources, found {len(source_hashes)}"
        )
    return failures


def verify_confirmatory_asset(root: Path) -> list[str]:
    path = root / "website" / "assets" / "waveform-comparison-data.js"
    text = path.read_text(encoding="utf-8-sig")
    pattern = re.compile(
        r"^window\.GWTC5_WAVEFORM_DATA = (.*);\n"
        r"window\.GWTC5_EVENT_DATA = (.*);\n$",
        re.DOTALL,
    )
    match = pattern.fullmatch(text)
    if not match:
        return ["Confirmatory website asset has an unexpected assignment format."]
    metrics = json.loads(match.group(1))
    events = json.loads(match.group(2))
    failures: list[str] = []
    if len(metrics) != 90:
        failures.append(f"Confirmatory metrics: expected 90, found {len(metrics)}")
    if len(events) != 18:
        failures.append(f"Confirmatory endpoints: expected 18, found {len(events)}")
    return failures


def main() -> None:
    root = find_repository_root()
    failures: list[str] = []
    for relative in HTML_FILES:
        failures.extend(check_html(root, relative))
    failures.extend(verify_confirmatory_asset(root))
    failures.extend(verify_full_catalog_asset(root))

    index_text = (root / "website" / "index.html").read_text(encoding="utf-8")
    if "https://d3js.org v7.9.0" not in index_text:
        failures.append("Embedded D3 7.9.0 marker is missing from website/index.html.")

    if failures:
        print("Static website verification failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Static website verification: PASS")
    print("HTML pages: 4 | confirmatory rows: 90 | primary: 78 | strict: 93 | census: 104")


if __name__ == "__main__":
    main()
