"""Build a checksum-verified inventory for the 104-event catalog extension.

This stage is outcome-blind. It uses frozen catalog/design records, exact
filenames, file sizes, and official checksums. It never opens an HDF5 file or
reads posterior metadata/values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_EVENT_COUNT = 104


def parse_args() -> argparse.Namespace:
    extension_root = Path(__file__).resolve().parents[1]
    archive_root = (
        extension_root.parent
        / "gwtc5_h1_rerun"
        / "GWTC5_reproducible_archive_v1_0_0"
    )
    replication_root = (
        extension_root.parent
        / "gwtc5_h1_rerun"
        / "GWTC5_prospective_h3_replication"
    )
    parser = argparse.ArgumentParser(
        description="Inventory and verify all expected GWTC-5 event files."
    )
    parser.add_argument(
        "--ranking",
        type=Path,
        default=archive_root
        / "01_public_source_preparation"
        / "manifests"
        / "catalog_screening_reference"
        / "catalog_xphm_xpnr_event_ranking.csv",
    )
    parser.add_argument(
        "--development-freeze",
        type=Path,
        default=archive_root
        / "data"
        / "processed"
        / "confirmatory_design"
        / "development_set_freeze.json",
    )
    parser.add_argument(
        "--confirmatory-sample",
        type=Path,
        default=archive_root
        / "results"
        / "confirmatory_final_assembly"
        / "confirmatory_final_strict_18_event_sample.csv",
    )
    parser.add_argument(
        "--replication-sample",
        type=Path,
        default=replication_root
        / "results"
        / "replication_final_strict_9_event_sample.csv",
    )
    parser.add_argument(
        "--official-md5",
        type=Path,
        default=extension_root / "design" / "official_zenodo_md5sums.txt",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        default=None,
        help="Directory searched recursively for exact expected filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=extension_root / "results" / "inventory",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Create a quick presence inventory without hashing large files.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def md5_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_official_md5(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        checksum, filename = parts
        if len(checksum) == 32 and filename.endswith(
            "-combined_PEDataRelease.hdf5"
        ):
            output[filename] = checksum.lower()
    return output


def linear_quantile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile of an empty sequence.")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def tertile(value: float, first: float, second: float) -> str:
    if value <= first:
        return "low"
    if value <= second:
        return "medium"
    return "high"


def index_input_files(input_dirs: list[Path]) -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = {}
    for directory in input_dirs:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*-combined_PEDataRelease.hdf5"):
            if path.is_file():
                by_name.setdefault(path.name, []).append(path.resolve())
    return by_name


def event_set_from_csv(path: Path) -> set[str]:
    return {row["event"] for row in read_csv(path)}


def role_for_event(
    event: str,
    pilot_events: set[str],
    development_events: set[str],
    confirmatory_events: set[str],
    replication_events: set[str],
) -> str:
    if event in pilot_events:
        return "development_pilot"
    if event in development_events:
        return "development_diagnostic"
    if event in confirmatory_events:
        return "confirmatory_strict"
    if event in replication_events:
        return "prospective_replication_strict"
    return "previously_unselected"


def main() -> None:
    args = parse_args()
    input_dirs = args.input_dir or [Path.home() / "Downloads"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranking = read_csv(args.ranking)
    if len(ranking) != EXPECTED_EVENT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENT_COUNT} catalog rows; found {len(ranking)}."
        )
    if len({row["event"] for row in ranking}) != EXPECTED_EVENT_COUNT:
        raise RuntimeError("The catalog ranking does not contain 104 unique events.")

    development_record = json.loads(
        args.development_freeze.read_text(encoding="utf-8")
    )
    development_events = set(development_record["development_events"])
    pilot_events = {
        row["event"]
        for row in ranking
        if row["already_downloaded_pilot"].strip().lower() == "true"
    }
    confirmatory_events = event_set_from_csv(args.confirmatory_sample)
    replication_events = event_set_from_csv(args.replication_sample)
    if len(pilot_events) != 5 or len(development_events) != 17:
        raise RuntimeError("Unexpected pilot/development event counts.")
    if len(confirmatory_events) != 18 or len(replication_events) != 9:
        raise RuntimeError("Unexpected confirmatory/replication event counts.")

    official_md5 = parse_official_md5(args.official_md5)
    expected_names = {row["event_filename"] for row in ranking}
    missing_checksums = sorted(expected_names.difference(official_md5))
    if missing_checksums:
        raise RuntimeError(
            "Official checksum manifest lacks expected event files:\n"
            + "\n".join(missing_checksums)
        )

    score_values = [float(row["maximum_screening_shift"]) for row in ranking]
    snr_values = [float(row["network_matched_filter_snr"]) for row in ranking]
    score_edges = (
        linear_quantile(score_values, 1 / 3),
        linear_quantile(score_values, 2 / 3),
    )
    snr_edges = (
        linear_quantile(snr_values, 1 / 3),
        linear_quantile(snr_values, 2 / 3),
    )

    indexed = index_input_files(input_dirs)
    output_rows: list[dict[str, Any]] = []
    for number, source_row in enumerate(
        sorted(ranking, key=lambda row: row["event"]), start=1
    ):
        event = source_row["event"]
        filename = source_row["event_filename"]
        matches = indexed.get(filename, [])
        expected = official_md5[filename]
        actual_path = matches[0] if len(matches) == 1 else None
        size_bytes = actual_path.stat().st_size if actual_path else ""
        actual = ""
        checksum_match: bool | str = ""
        status = "missing"

        if len(matches) > 1:
            status = "ambiguous_duplicate_filename"
        elif actual_path is not None and size_bytes == 0:
            status = "zero_byte_file"
        elif actual_path is not None and args.skip_md5:
            status = "present_not_hashed"
        elif actual_path is not None:
            print(
                f"[{number:03d}/{EXPECTED_EVENT_COUNT}] MD5 {event} "
                f"({size_bytes / (1024**2):.1f} MiB)",
                flush=True,
            )
            actual = md5_file(actual_path)
            checksum_match = actual.lower() == expected.lower()
            status = "verified" if checksum_match else "checksum_mismatch"

        score = float(source_row["maximum_screening_shift"])
        snr = float(source_row["network_matched_filter_snr"])
        output_rows.append(
            {
                "event": event,
                "prior_role": role_for_event(
                    event,
                    pilot_events,
                    development_events,
                    confirmatory_events,
                    replication_events,
                ),
                "expected_filename": filename,
                "screening_rank": source_row["screening_rank"],
                "maximum_screening_shift": source_row[
                    "maximum_screening_shift"
                ],
                "network_matched_filter_snr": source_row[
                    "network_matched_filter_snr"
                ],
                "screening_score_tertile_104": tertile(score, *score_edges),
                "snr_tertile_104": tertile(snr, *snr_edges),
                "match_count": len(matches),
                "actual_path": str(actual_path) if actual_path else "",
                "size_bytes": size_bytes,
                "official_md5": expected,
                "actual_md5": actual,
                "checksum_match": checksum_match,
                "inventory_status": status,
            }
        )

    fields = list(output_rows[0])
    inventory_path = args.output_dir / "full_catalog_input_inventory.csv"
    write_csv(inventory_path, output_rows, fields)

    status_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for row in output_rows:
        status_counts[row["inventory_status"]] = (
            status_counts.get(row["inventory_status"], 0) + 1
        )
        role_counts[row["prior_role"]] = role_counts.get(row["prior_role"], 0) + 1

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_event_count": EXPECTED_EVENT_COUNT,
        "input_directories": [str(path.resolve()) for path in input_dirs],
        "md5_calculated": not args.skip_md5,
        "status_counts": status_counts,
        "role_counts": role_counts,
        "full_catalog_score_tertile_boundaries": list(score_edges),
        "full_catalog_snr_tertile_boundaries": list(snr_edges),
        "complete_and_verified": status_counts == {"verified": EXPECTED_EVENT_COUNT},
        "source_hashes": {
            "ranking_sha256": sha256_file(args.ranking),
            "development_freeze_sha256": sha256_file(args.development_freeze),
            "confirmatory_sample_sha256": sha256_file(args.confirmatory_sample),
            "replication_sample_sha256": sha256_file(args.replication_sample),
            "official_md5_manifest_sha256": sha256_file(args.official_md5),
        },
    }
    (args.output_dir / "full_catalog_inventory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

