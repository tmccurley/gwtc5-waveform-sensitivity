"""Verify cross-file consistency of the finalized full-catalog outputs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


EXPECTED_EVENTS = 104


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    inventory_dir = root / "results" / "inventory"
    audit_dir = root / "results" / "metadata_audit"
    analysis_dir = root / "results" / "analysis"

    inventory = read_csv(inventory_dir / "full_catalog_input_inventory.csv")
    metadata = read_csv(audit_dir / "full_catalog_metadata_event_summary.csv")
    metrics = read_csv(analysis_dir / "full_catalog_parameter_metrics.csv")
    events = read_csv(analysis_dir / "full_catalog_event_deltas.csv")
    populations = read_csv(
        analysis_dir / "full_catalog_population_summaries.csv"
    )
    consistency = read_csv(
        analysis_dir / "locked_event_consistency_check.csv"
    )
    manifest = json.loads(
        (analysis_dir / "full_catalog_analysis_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(inventory) == EXPECTED_EVENTS
    assert len({row["event"] for row in inventory}) == EXPECTED_EVENTS
    assert all(row["inventory_status"] == "verified" for row in inventory)
    assert len(metadata) == EXPECTED_EVENTS
    assert len({row["event"] for row in metadata}) == EXPECTED_EVENTS
    assert sum(row["strict_pair_status"] == "strict_tier_A_automated" for row in metadata) == 93
    assert sum(row["strict_pair_status"] == "non_tier_A_metadata_difference" for row in metadata) == 11
    assert all(row["all_pair_outcome_usable"] == "True" for row in metadata)
    c01 = [row for row in metadata if row["configuration_tag"] == "C01"]
    assert len(c01) == 1 and c01[0]["event"] == "GW240925_005809"

    assert len(metrics) == EXPECTED_EVENTS * 2
    assert len(events) == EXPECTED_EVENTS
    assert len({row["event"] for row in events}) == EXPECTED_EVENTS
    for row in events:
        chi_eff = float(row["chi_eff_normalized_wasserstein_1"])
        distance = float(row["luminosity_distance_normalized_wasserstein_1"])
        delta = float(row["delta_chi_eff_minus_luminosity_distance"])
        assert all(math.isfinite(value) for value in (chi_eff, distance, delta))
        assert math.isclose(delta, chi_eff - distance, rel_tol=0, abs_tol=1e-15)
        assert (delta > 0) == (row["delta_positive"] == "True")
        assert (delta < 0) == (row["delta_negative"] == "True")

    by_population = {row["population"]: row for row in populations}
    strict_events = [row for row in events if row["strict_population_member"] == "True"]
    assert len(strict_events) == 93
    assert sum(row["delta_positive"] == "True" for row in strict_events) == 89
    assert sum(row["delta_negative"] == "True" for row in strict_events) == 4
    assert int(by_population["strict_matched_settings"]["event_count"]) == 93
    assert int(by_population["strict_matched_settings"]["positive_delta_count"]) == 89
    assert int(by_population["all_outcome_usable_pairs"]["event_count"]) == 104
    assert int(by_population["all_outcome_usable_pairs"]["positive_delta_count"]) == 99

    assert len(consistency) == 27
    assert all(row["delta_sign_preserved"] == "True" for row in consistency)
    assert manifest["status"] == "final_full_catalog"
    assert manifest["checksum_verified_event_count"] == 104
    assert manifest["all_outcome_usable_event_count"] == 104
    assert manifest["strict_matched_settings_event_count"] == 93

    print(
        json.dumps(
            {
                "verification": "PASS",
                "catalog_events": 104,
                "strict_events": 93,
                "strict_positive": 89,
                "all_pairs_positive": 99,
                "locked_signs_preserved": 27,
                "c01_event": "GW240925_005809",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

