"""Seal final metadata eligibility decisions before posterior outcomes open."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOCK_STATUS = "FULL_CATALOG_TIER_DECISIONS_LOCKED_BEFORE_OUTCOMES"
ALLOWED_CLASSIFICATIONS = {
    "Tier A",
    "Tier B descriptive",
    "Tier C descriptive",
    "Excluded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--metadata-summary", type=Path, required=True)
    parser.add_argument("--manual-decisions", type=Path, required=True)
    parser.add_argument("--output-decisions", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    for target in (args.output_decisions, args.output_lock):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite lock output: {target}")

    inventory = read_csv(args.inventory)
    summaries = read_csv(args.metadata_summary)
    manual = read_csv(args.manual_decisions)
    if len(inventory) != 104 or len(summaries) != 104:
        raise RuntimeError("Inventory and metadata summary must have 104 events.")
    inventory_by_event = {row["event"]: row for row in inventory}
    summary_by_event = {row["event"]: row for row in summaries}
    manual_by_event = {row["event"]: row for row in manual}
    if len(manual_by_event) != len(manual):
        raise RuntimeError("Manual decision file contains duplicate events.")

    required_manual = {
        event
        for event, row in summary_by_event.items()
        if row["provisional_status"] == "manual_tier_review_required"
    }
    if set(manual_by_event) != required_manual:
        missing = sorted(required_manual - set(manual_by_event))
        extra = sorted(set(manual_by_event) - required_manual)
        raise RuntimeError(f"Manual decision set mismatch; missing={missing}, extra={extra}")

    decisions: list[dict[str, Any]] = []
    for event in sorted(inventory_by_event):
        summary = summary_by_event[event]
        status = summary["provisional_status"]
        if status == "provisional_tier_A":
            classification = "Tier A"
            include_core = True
            reason = "No review-required recorded-setting difference"
            decision_source = "automatic frozen audit rule"
            outcome_inspected = False
        elif status == "manual_tier_review_required":
            row = manual_by_event[event]
            classification = row["classification"]
            if classification not in ALLOWED_CLASSIFICATIONS:
                raise RuntimeError(f"Invalid classification for {event}")
            include_core = row["include_core"].strip().lower() == "true"
            reason = row["reason"].strip()
            decision_source = "documented metadata-only manual review"
            outcome_inspected = (
                row["posterior_metrics_inspected_for_decision"].strip().lower()
                == "true"
            )
            if not reason:
                raise RuntimeError(f"Manual decision lacks reason for {event}")
            if outcome_inspected:
                raise RuntimeError(f"Outcome inspection recorded for {event}")
            if include_core != (classification == "Tier A"):
                raise RuntimeError(
                    f"include_core must be true exactly for Tier A: {event}"
                )
        else:
            classification = "Excluded"
            include_core = False
            reason = f"Automatic exclusion from metadata audit: {status}"
            decision_source = "automatic frozen audit rule"
            outcome_inspected = False

        decisions.append(
            {
                "event": event,
                "population_A_nondevelopment": inventory_by_event[event][
                    "population_A_nondevelopment"
                ],
                "prior_stage": inventory_by_event[event]["prior_stage"],
                "provisional_status": status,
                "classification": classification,
                "include_core": include_core,
                "reason": reason,
                "decision_source": decision_source,
                "posterior_metrics_inspected_for_decision": outcome_inspected,
            }
        )

    write_csv(args.output_decisions, decisions, list(decisions[0]))
    lock = {
        "status": LOCK_STATUS,
        "lock_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": len(decisions),
        "population_A_event_count": sum(
            row["population_A_nondevelopment"].lower() == "true"
            for row in inventory
        ),
        "tier_A_count": sum(row["classification"] == "Tier A" for row in decisions),
        "population_A_tier_A_count": sum(
            row["classification"] == "Tier A"
            and str(row["population_A_nondevelopment"]).lower() == "true"
            for row in decisions
        ),
        "excluded_or_descriptive_count": sum(
            row["classification"] != "Tier A" for row in decisions
        ),
        "posterior_metrics_inspected_for_decisions": False,
        "source_hashes": {
            "inventory_sha256": sha256_file(args.inventory),
            "metadata_summary_sha256": sha256_file(args.metadata_summary),
            "manual_decisions_sha256": sha256_file(args.manual_decisions),
            "tier_decisions_sha256": sha256_file(args.output_decisions),
            "locking_script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    args.output_lock.parent.mkdir(parents=True, exist_ok=True)
    args.output_lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
