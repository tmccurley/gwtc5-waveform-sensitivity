"""Run the locked metadata-only audit for all 104 paired-label events.

The script reuses the exact comparison functions from the sealed replication
audit. It reads posterior dataset schemas and row counts but never indexes a
posterior dataset or calculates a distributional outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py


CORE_PARAMETERS = [
    "chirp_mass",
    "mass_ratio",
    "chi_eff",
    "luminosity_distance",
]
ADDITIONAL_PARAMETERS = ["chirp_mass_source", "chi_p"]
MODELS = ["C00:IMRPhenomXPHM-SpinTaylor", "C00:IMRPhenomXPNR"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-inventory", type=Path, required=True)
    parser.add_argument("--parent-audit-module", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
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


def load_parent_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("sealed_replication_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REQUIRED_PARAMETERS = list(CORE_PARAMETERS)
    if list(module.MODELS) != MODELS:
        raise RuntimeError("Parent audit model labels differ from the locked pair.")
    return module


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        for child in path.iterdir():
            if child.is_dir():
                raise RuntimeError(f"Refusing to remove nested directory: {child}")
            child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def inspect_parameter_schema(event: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as source:
        for model in MODELS:
            posterior = source[model]["posterior_samples"]
            if not isinstance(posterior, h5py.Dataset):
                raise TypeError(f"Posterior is not a dataset: {event} / {model}")
            names = set(posterior.dtype.names or [])
            for parameter in CORE_PARAMETERS + ADDITIONAL_PARAMETERS:
                rows.append(
                    {
                        "event": event,
                        "model": model,
                        "parameter": parameter,
                        "present": parameter in names,
                        "posterior_row_count": int(posterior.shape[0]),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    parent = load_parent_module(args.parent_audit_module)
    inventory = read_csv(args.verified_inventory)
    if len(inventory) != 104 or len({row["event"] for row in inventory}) != 104:
        raise RuntimeError("Verified inventory must contain 104 unique events.")

    integrity_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []

    for row in inventory:
        event = row["event"]
        path = Path(row["canonical_local_path"])
        expected_size = int(row["official_size_bytes"])
        expected_md5 = row["official_md5"].lower()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"Size mismatch for {event}")
        actual_md5 = md5_file(path)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {event}")
        integrity_rows.append(
            {
                "event": event,
                "source_path": str(path),
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "expected_md5": expected_md5,
                "actual_md5": actual_md5,
                "verified": True,
            }
        )

        summary, comparisons = parent.audit_event(event, path)
        summary.update(
            {
                "population_A_nondevelopment": row[
                    "population_A_nondevelopment"
                ],
                "prior_stage": row["prior_stage"],
            }
        )
        summary_rows.append(summary)
        comparison_rows.extend(comparisons)
        schema_rows.extend(inspect_parameter_schema(event, path))
        print(json.dumps({"audited": len(summary_rows), "event": event}), flush=True)

    write_csv(
        args.output_dir / "full_catalog_input_integrity.csv",
        integrity_rows,
        list(integrity_rows[0]),
    )
    write_csv(
        args.output_dir / "full_catalog_metadata_event_summary.csv",
        summary_rows,
        [
            "event",
            "population_A_nondevelopment",
            "prior_stage",
            "source_path",
            "file_opened",
            "xphm_present",
            "xpnr_present",
            "xphm_sample_count",
            "xpnr_sample_count",
            "xphm_parameters_present",
            "xpnr_parameters_present",
            "required_parameters_present",
            "review_required_count",
            "provisional_status",
            "error",
        ],
    )
    write_csv(
        args.output_dir / "full_catalog_pair_metadata_comparison.csv",
        comparison_rows,
        [
            "event",
            "section",
            "setting",
            "comparison_category",
            "xphm_value_json",
            "xpnr_value_json",
        ],
    )
    write_csv(
        args.output_dir / "full_catalog_parameter_schema.csv",
        schema_rows,
        list(schema_rows[0]),
    )
    review_rows = [
        row for row in comparison_rows if row["comparison_category"] == "review_required"
    ]
    write_csv(
        args.output_dir / "full_catalog_manual_tier_review_required.csv",
        review_rows,
        [
            "event",
            "section",
            "setting",
            "comparison_category",
            "xphm_value_json",
            "xpnr_value_json",
        ],
    )

    manifest = {
        "purpose": "locked full-catalog metadata-only Tier A audit",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": len(summary_rows),
        "population_A_count": sum(
            row["population_A_nondevelopment"].lower() == "true"
            for row in inventory
        ),
        "provisional_tier_A_count": sum(
            row["provisional_status"] == "provisional_tier_A"
            for row in summary_rows
        ),
        "manual_review_event_count": len(
            {row["event"] for row in review_rows}
        ),
        "posterior_values_read": False,
        "posterior_outcomes_calculated": False,
        "source_hashes": {
            str(args.verified_inventory.resolve()): sha256_file(
                args.verified_inventory
            ),
            str(args.parent_audit_module.resolve()): sha256_file(
                args.parent_audit_module
            ),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
    }
    result_paths = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    manifest["result_hashes"] = {
        path.name: sha256_file(path) for path in result_paths
    }
    (args.output_dir / "full_catalog_metadata_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
