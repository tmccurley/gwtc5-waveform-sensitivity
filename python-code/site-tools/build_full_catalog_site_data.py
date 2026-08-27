"""Build the browser-ready full-catalog data asset from frozen result tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JavaScript path (default: website/assets/full-catalog-generalization-data.js)",
    )
    return parser.parse_args()


def find_repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "website" / "index.html").is_file():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def keep(row: dict[str, str], fields: tuple[str, ...]) -> dict[str, str]:
    return {field: row[field] for field in fields}


def main() -> None:
    args = parse_args()
    root = find_repository_root()
    capsule = root / "reproducibility" / "full-catalog-generalization"
    generalization = capsule / "results" / "generalization_full"
    analysis = capsule / "results" / "analysis"

    endpoint_path = generalization / "full_catalog_event_endpoints.csv"
    tests_path = generalization / "full_catalog_generalization_tests.csv"
    census_path = analysis / "full_catalog_event_deltas.csv"
    summaries_path = analysis / "full_catalog_population_summaries.csv"
    manifest_path = generalization / "full_catalog_generalization_manifest.json"

    required = (
        endpoint_path,
        tests_path,
        census_path,
        summaries_path,
        manifest_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen site input: " + ", ".join(map(str, missing)))

    endpoint_fields = (
        "event",
        "event_median_NW1",
        "chi_eff_nw1",
        "luminosity_distance_nw1",
        "delta_nw1",
        "population_A_nondevelopment",
        "prior_stage",
        "network_matched_filter_snr",
        "maximum_screening_shift",
    )
    census_fields = (
        "event",
        "prior_role",
        "strict_pair_status",
        "configuration_tag",
        "strict_population_member",
        "chi_eff_normalized_wasserstein_1",
        "luminosity_distance_normalized_wasserstein_1",
        "delta_chi_eff_minus_luminosity_distance",
        "delta_positive",
        "delta_negative",
    )

    strict_events = [keep(row, endpoint_fields) for row in read_csv(endpoint_path)]
    primary_events = [
        row for row in strict_events if row["population_A_nondevelopment"] == "True"
    ]
    census_events = [keep(row, census_fields) for row in read_csv(census_path)]
    tests = read_csv(tests_path)
    summaries = read_csv(summaries_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if len(primary_events) != 78 or len(strict_events) != 93 or len(census_events) != 104:
        raise RuntimeError(
            "Frozen population counts changed: expected 78 primary, 93 strict, and 104 census events."
        )
    if sum(float(row["delta_nw1"]) > 0 for row in primary_events) != 75:
        raise RuntimeError("Primary H3 direction count no longer matches the locked result.")
    if sum(row["delta_positive"] == "True" for row in census_events) != 99:
        raise RuntimeError("All-usable-pairs H3 direction count no longer matches the final census.")

    payload = {
        "analysis_completed_utc": manifest["completed_utc"],
        "primary_events": primary_events,
        "strict_events": strict_events,
        "census_events": census_events,
        "generalization_tests": tests,
        "population_summaries": summaries,
        "source_sha256": {
            path.relative_to(capsule).as_posix(): sha256_file(path) for path in required
        },
    }

    output = args.output or (
        root / "website" / "assets" / "full-catalog-generalization-data.js"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "window.GWTC5_FULL_CATALOG_DATA = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"Created: {output}")
    print("Verified: 78 primary | 93 strict | 104 all usable")


if __name__ == "__main__":
    main()
