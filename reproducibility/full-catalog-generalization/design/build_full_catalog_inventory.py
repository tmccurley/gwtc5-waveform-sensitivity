"""Build the locked GWTC-5 full-catalog event and download inventories.

This preparation script reads only catalog/design records, official Zenodo file
metadata, and local filenames/sizes. It never opens a posterior HDF5 file and
cannot calculate a posterior-disagreement outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


ZENODO_RECORDS = ("20348005", "20348006")
DEVELOPMENT_EVENTS = {
    "GW240612_081540",
    "GW240908_082628",
    "GW241011_233834",
    "GW241230_233618",
    "GW250114_082203",
    "GW240920_124024",
    "GW241127_061008",
    "GW241129_021832",
    "GW240513_183302",
    "GW240615_113620",
    "GW241116_151753",
    "GW240621_195059",
    "GW241225_082815",
    "GW241130_034908",
    "GW240920_073424",
    "GW240919_061559",
    "GW241229_155844",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--confirmatory-lock", type=Path, required=True)
    parser.add_argument("--replication-lock", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--verify-existing-md5",
        action="store_true",
        help="Hash canonical local event files as well as checking their sizes.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_zenodo_file_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for record_id in ZENODO_RECORDS:
        url = f"https://zenodo.org/api/records/{record_id}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "GWTC5-full-catalog-inventory/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            record = json.load(response)
        for item in record["files"]:
            filename = item["key"]
            if not (
                filename.startswith("IGWN-GWTC5p0-29ebe06b7_25-GW")
                and filename.endswith("-combined_PEDataRelease.hdf5")
            ):
                continue
            if filename in index:
                raise RuntimeError(f"Official filename appears twice: {filename}")
            checksum_type, checksum = item["checksum"].split(":", 1)
            if checksum_type.lower() != "md5":
                raise RuntimeError(f"Unexpected checksum type for {filename}")
            index[filename] = {
                "zenodo_record": record_id,
                "official_size_bytes": int(item["size"]),
                "official_md5": checksum.lower(),
                "download_url": (
                    f"https://zenodo.org/records/{record_id}/files/"
                    f"{filename}?download=1"
                ),
                "api_content_url": item["links"]["self"],
            }
    return index


def known_stage_sets(
    confirmatory_lock: dict[str, Any],
    replication_lock: dict[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    confirmatory_final = {row["event"] for row in confirmatory_lock["events"]}
    confirmatory_failures = {
        row["original_selected_event"]
        for row in confirmatory_lock["events"]
        if row["original_selected_event"] != row["event"]
    }
    replication_final = set(replication_lock["ordered_events"])
    replication_failure = {replication_lock["excluded_candidate"]["event"]}
    failures = confirmatory_failures | replication_failure
    return confirmatory_final, replication_final, failures


def main() -> None:
    args = parse_args()
    ranking = read_csv(args.ranking)
    confirmatory_lock = json.loads(args.confirmatory_lock.read_text(encoding="utf-8"))
    replication_lock = json.loads(args.replication_lock.read_text(encoding="utf-8"))
    official = fetch_zenodo_file_index()
    confirmatory_final, replication_final, known_failures = known_stage_sets(
        confirmatory_lock, replication_lock
    )

    if len(ranking) != 104 or len({row["event"] for row in ranking}) != 104:
        raise RuntimeError("The ranking file must contain 104 unique events.")
    if len(DEVELOPMENT_EVENTS) != 17:
        raise RuntimeError("The frozen development set must contain 17 events.")

    inventory: list[dict[str, Any]] = []
    for source in sorted(ranking, key=lambda row: row["event"]):
        event = source["event"]
        filename = source["event_filename"]
        if filename not in official:
            raise FileNotFoundError(f"Official Zenodo index lacks {filename}")
        release = official[filename]
        local_path = args.download_dir / filename
        local_present = local_path.is_file()
        local_size = local_path.stat().st_size if local_present else None
        size_match = (
            local_size == release["official_size_bytes"] if local_present else False
        )
        local_md5 = ""
        md5_match: bool | str = "not_checked"
        if local_present and args.verify_existing_md5:
            local_md5 = md5_file(local_path)
            md5_match = local_md5 == release["official_md5"]

        if event in DEVELOPMENT_EVENTS:
            prior_stage = "development"
        elif event in confirmatory_final:
            prior_stage = "confirmatory_final"
        elif event in replication_final:
            prior_stage = "replication_final"
        elif event in known_failures:
            prior_stage = "known_metadata_failure"
        else:
            prior_stage = "other_nondevelopment"

        inventory.append(
            {
                "event": event,
                "event_filename": filename,
                "population_A_nondevelopment": event not in DEVELOPMENT_EVENTS,
                "population_B_complete_catalog": True,
                "prior_stage": prior_stage,
                "screening_rank": source["screening_rank"],
                "maximum_screening_shift": source["maximum_screening_shift"],
                "network_matched_filter_snr": source[
                    "network_matched_filter_snr"
                ],
                "zenodo_record": release["zenodo_record"],
                "official_size_bytes": release["official_size_bytes"],
                "official_md5": release["official_md5"],
                "download_url": release["download_url"],
                "api_content_url": release["api_content_url"],
                "canonical_local_path": str(local_path.resolve()),
                "canonical_file_present": local_present,
                "local_size_bytes": local_size if local_size is not None else "",
                "size_match": size_match,
                "local_md5": local_md5,
                "md5_match": md5_match,
            }
        )

    fields = list(inventory[0])
    write_csv(args.output_dir / "full_catalog_event_inventory.csv", inventory, fields)

    missing = [
        row
        for row in inventory
        if row["population_A_nondevelopment"]
        and not row["canonical_file_present"]
    ]
    download_fields = [
        "event",
        "event_filename",
        "prior_stage",
        "zenodo_record",
        "official_size_bytes",
        "official_md5",
        "download_url",
        "api_content_url",
        "canonical_local_path",
    ]
    write_csv(
        args.output_dir / "missing_population_A_download_manifest.csv",
        missing,
        download_fields,
    )
    (args.output_dir / "missing_population_A_download_urls.txt").write_text(
        "\n".join(str(row["download_url"]) for row in missing) + "\n",
        encoding="utf-8",
    )

    summary = {
        "official_event_file_count": len(official),
        "catalog_event_count": len(inventory),
        "population_A_count": sum(
            bool(row["population_A_nondevelopment"]) for row in inventory
        ),
        "population_B_count": len(inventory),
        "canonical_files_present_total": sum(
            bool(row["canonical_file_present"]) for row in inventory
        ),
        "population_A_files_present": sum(
            bool(row["population_A_nondevelopment"])
            and bool(row["canonical_file_present"])
            for row in inventory
        ),
        "population_A_files_missing": len(missing),
        "missing_download_bytes": sum(
            int(row["official_size_bytes"]) for row in missing
        ),
        "official_records": list(ZENODO_RECORDS),
        "source_hashes": {
            str(args.ranking.resolve()): sha256_file(args.ranking),
            str(args.confirmatory_lock.resolve()): sha256_file(
                args.confirmatory_lock
            ),
            str(args.replication_lock.resolve()): sha256_file(args.replication_lock),
        },
    }
    (args.output_dir / "full_catalog_inventory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
