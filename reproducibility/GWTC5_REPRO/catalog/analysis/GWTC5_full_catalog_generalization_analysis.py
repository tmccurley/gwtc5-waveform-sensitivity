"""Run the post-lock GWTC-5.0 full-catalog H3 generalization analysis.

The script calculates the frozen NW1 definition for chi_eff and luminosity
distance using every finite stored posterior row. It reports a strict
matched-settings population and a broader all-outcome-usable descriptive
population. Incomplete runs are refused unless explicitly requested for
pipeline validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import scipy
from scipy.stats import binomtest, wasserstein_distance, wilcoxon


XPHM_FAMILY = "IMRPhenomXPHM-SpinTaylor"
XPNR_FAMILY = "IMRPhenomXPNR"
PARAMETERS = ("chi_eff", "luminosity_distance")
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_RESAMPLES = 100_000
EXPECTED_CATALOG_EVENTS = 104
WEIGHT_NAMES = {
    "weight",
    "weights",
    "posterior_weight",
    "posterior_weights",
    "log_weight",
    "log_weights",
}
EXPECTED_OUTPUTS = {
    "full_catalog_parameter_metrics.csv",
    "full_catalog_event_deltas.csv",
    "full_catalog_population_summaries.csv",
    "full_catalog_subgroup_summaries.csv",
    "locked_event_consistency_check.csv",
    "full_catalog_analysis_manifest.json",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    archive_root = (
        root.parent / "gwtc5_h1_rerun" / "GWTC5_reproducible_archive_v1_0_0"
    )
    replication_root = (
        root.parent / "gwtc5_h1_rerun" / "GWTC5_prospective_h3_replication"
    )
    parser = argparse.ArgumentParser(
        description="Post-lock full-catalog H3 generalization analysis."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=root / "results" / "inventory" / "full_catalog_input_inventory.csv",
    )
    parser.add_argument(
        "--inventory-summary",
        type=Path,
        default=root / "results" / "inventory" / "full_catalog_inventory_summary.json",
    )
    parser.add_argument(
        "--metadata-summary",
        type=Path,
        default=root
        / "results"
        / "metadata_audit"
        / "full_catalog_metadata_event_summary.csv",
    )
    parser.add_argument(
        "--confirmatory-reference",
        type=Path,
        default=archive_root
        / "reference_results"
        / "confirmatory_analysis"
        / "confirmatory_parameter_metrics.csv",
    )
    parser.add_argument(
        "--replication-reference",
        type=Path,
        default=replication_root
        / "results"
        / "replication_analysis"
        / "replication_parameter_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "analysis",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Permit a clearly provisional run on the currently verified files. "
            "Without this flag, all 104 inputs must be verified."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known outputs from a documented full rerun.",
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


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_output(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in output_dir.iterdir() if path.is_file()]
    unexpected = [path for path in existing if path.name not in EXPECTED_OUTPUTS]
    if unexpected:
        raise FileExistsError(
            "Unexpected file(s) in analysis output directory:\n"
            + "\n".join(str(path) for path in unexpected)
        )
    if existing and not overwrite:
        raise FileExistsError(
            f"Analysis output already exists in {output_dir}; use --overwrite intentionally."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def interval_summary(values: np.ndarray) -> dict[str, float]:
    lower, median, upper = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        "lower_90": float(lower),
        "median": float(median),
        "upper_90": float(upper),
        "width_90": float(upper - lower),
    }


def pair_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x_summary = interval_summary(x)
    y_summary = interval_summary(y)
    average_width = 0.5 * (
        x_summary["width_90"] + y_summary["width_90"]
    )
    raw_w1 = float(wasserstein_distance(x, y))
    nw1 = raw_w1 / average_width if average_width > 0 else math.nan
    return {
        "xphm_lower_90": x_summary["lower_90"],
        "xphm_median": x_summary["median"],
        "xphm_upper_90": x_summary["upper_90"],
        "xphm_width_90": x_summary["width_90"],
        "xpnr_lower_90": y_summary["lower_90"],
        "xpnr_median": y_summary["median"],
        "xpnr_upper_90": y_summary["upper_90"],
        "xpnr_width_90": y_summary["width_90"],
        "average_width_90": average_width,
        "wasserstein_1": raw_w1,
        "normalized_wasserstein_1": nw1,
    }


def load_field(
    source: h5py.File, event: str, model: str, parameter: str
) -> np.ndarray:
    if model not in source or "posterior_samples" not in source[model]:
        raise KeyError(f"Missing posterior for {event} / {model}.")
    posterior = source[model]["posterior_samples"]
    if not isinstance(posterior, h5py.Dataset):
        raise TypeError(f"Posterior is not a dataset for {event} / {model}.")
    names = set(posterior.dtype.names or [])
    found_weights = sorted(names.intersection(WEIGHT_NAMES))
    if found_weights:
        raise RuntimeError(
            f"Explicit posterior weights found for {event} / {model}: {found_weights}"
        )
    if parameter not in names:
        raise KeyError(f"Missing {event} / {model} / {parameter}.")
    values = np.asarray(posterior.fields(parameter)[:], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 4:
        raise RuntimeError(
            f"Too few finite samples for {event} / {model} / {parameter}: "
            f"{values.size}"
        )
    return values


def analyze_events(
    metadata_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameter_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for number, row in enumerate(metadata_rows, start=1):
        event = row["event"]
        print(
            f"[{number:03d}/{len(metadata_rows):03d}] posterior metrics {event}",
            flush=True,
        )
        nw1_by_parameter: dict[str, float] = {}
        xphm_label = row["xphm_label"]
        xpnr_label = row["xpnr_label"]
        with h5py.File(Path(row["source_path"]), "r") as source:
            for parameter in PARAMETERS:
                x = load_field(source, event, xphm_label, parameter)
                y = load_field(source, event, xpnr_label, parameter)
                metrics = pair_metrics(x, y)
                nw1_by_parameter[parameter] = metrics[
                    "normalized_wasserstein_1"
                ]
                parameter_rows.append(
                    {
                        "event": event,
                        "prior_role": row["prior_role"],
                        "strict_pair_status": row["strict_pair_status"],
                        "configuration_tag": row["configuration_tag"],
                        "xphm_label": xphm_label,
                        "xpnr_label": xpnr_label,
                        "parameter": parameter,
                        "xphm_finite_sample_count": int(x.size),
                        "xpnr_finite_sample_count": int(y.size),
                        "sample_use": "all_finite_released_rows",
                        **metrics,
                    }
                )

        delta = (
            nw1_by_parameter["chi_eff"]
            - nw1_by_parameter["luminosity_distance"]
        )
        event_rows.append(
            {
                "event": event,
                "prior_role": row["prior_role"],
                "screening_score_tertile_104": row[
                    "screening_score_tertile_104"
                ],
                "snr_tertile_104": row["snr_tertile_104"],
                "strict_pair_status": row["strict_pair_status"],
                "configuration_tag": row["configuration_tag"],
                "strict_population_member": row["strict_pair_status"]
                == "strict_tier_A_automated",
                "chi_eff_normalized_wasserstein_1": nw1_by_parameter["chi_eff"],
                "luminosity_distance_normalized_wasserstein_1": nw1_by_parameter[
                    "luminosity_distance"
                ],
                "delta_chi_eff_minus_luminosity_distance": delta,
                "delta_positive": delta > 0,
                "delta_zero": delta == 0,
                "delta_negative": delta < 0,
            }
        )
    return parameter_rows, event_rows


def bootstrap_median_interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    medians: list[np.ndarray] = []
    remaining = BOOTSTRAP_RESAMPLES
    chunk_size = 10_000
    while remaining:
        count = min(chunk_size, remaining)
        indices = rng.integers(0, values.size, size=(count, values.size))
        medians.append(np.median(values[indices], axis=1))
        remaining -= count
    combined = np.concatenate(medians)
    low, high = np.quantile(combined, [0.025, 0.975])
    return float(low), float(high)


def population_summary(
    population: str, rows: list[dict[str, Any]], catalog_complete: bool
) -> dict[str, Any]:
    deltas = np.asarray(
        [row["delta_chi_eff_minus_luminosity_distance"] for row in rows],
        dtype=float,
    )
    if not deltas.size:
        raise RuntimeError(f"No events available for population {population}.")
    positives = int(np.sum(deltas > 0))
    zeros = int(np.sum(deltas == 0))
    negatives = int(np.sum(deltas < 0))
    nonzero = positives + negatives
    sign_p = (
        float(binomtest(positives, nonzero, 0.5, alternative="greater").pvalue)
        if nonzero
        else math.nan
    )
    if np.any(deltas != 0):
        signed_rank = wilcoxon(
            deltas,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        )
        wilcoxon_statistic = float(signed_rank.statistic)
        wilcoxon_p = float(signed_rank.pvalue)
    else:
        wilcoxon_statistic = math.nan
        wilcoxon_p = math.nan
    bootstrap_low, bootstrap_high = bootstrap_median_interval(deltas)
    q1, median, q3 = np.quantile(deltas, [0.25, 0.50, 0.75])
    return {
        "population": population,
        "analysis_status": "final_full_catalog" if catalog_complete else "provisional_incomplete_catalog",
        "event_count": int(deltas.size),
        "catalog_fraction": float(deltas.size / EXPECTED_CATALOG_EVENTS),
        "positive_delta_count": positives,
        "zero_delta_count": zeros,
        "negative_delta_count": negatives,
        "positive_fraction_all": float(positives / deltas.size),
        "positive_fraction_nonzero": float(positives / nonzero) if nonzero else math.nan,
        "delta_q1": float(q1),
        "median_delta": float(median),
        "delta_q3": float(q3),
        "bootstrap_median_95_low": bootstrap_low,
        "bootstrap_median_95_high": bootstrap_high,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "one_sided_exact_sign_p": sign_p,
        "wilcoxon_statistic": wilcoxon_statistic,
        "one_sided_wilcoxon_p": wilcoxon_p,
        "inference_label": "post-lock descriptive/generalization",
    }


def subgroup_summaries(
    population: str,
    rows: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    groups = sorted({str(row[field]) for row in rows})
    output: list[dict[str, Any]] = []
    for group in groups:
        selected = [row for row in rows if str(row[field]) == group]
        deltas = np.asarray(
            [row["delta_chi_eff_minus_luminosity_distance"] for row in selected],
            dtype=float,
        )
        output.append(
            {
                "population": population,
                "subgroup_variable": field,
                "subgroup": group,
                "event_count": int(deltas.size),
                "positive_delta_count": int(np.sum(deltas > 0)),
                "zero_delta_count": int(np.sum(deltas == 0)),
                "negative_delta_count": int(np.sum(deltas < 0)),
                "positive_fraction_all": float(np.mean(deltas > 0)),
                "median_delta": float(np.median(deltas)),
                "subgroup_test_performed": False,
            }
        )
    return output


def reference_lookup(paths: Iterable[tuple[str, Path]]) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for source_name, path in paths:
        for row in read_csv(path):
            parameter = row.get("parameter", "")
            if parameter not in PARAMETERS:
                continue
            value = row.get("normalized_wasserstein_1", "")
            if value == "":
                continue
            output[(row["event"], parameter)] = {
                "source": source_name,
                "value": value,
            }
    return output


def consistency_rows(
    parameter_rows: list[dict[str, Any]],
    references: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    current = {
        (row["event"], row["parameter"]): row for row in parameter_rows
    }
    output: list[dict[str, Any]] = []
    events = sorted({event for event, _ in references})
    for event in events:
        if not all((event, parameter) in current for parameter in PARAMETERS):
            continue
        values: dict[str, dict[str, float | str]] = {}
        for parameter in PARAMETERS:
            reference = references[(event, parameter)]
            reference_value = float(reference["value"])
            current_value = float(
                current[(event, parameter)]["normalized_wasserstein_1"]
            )
            values[parameter] = {
                "source": reference["source"],
                "reference": reference_value,
                "current": current_value,
                "difference": current_value - reference_value,
            }
        reference_delta = float(values["chi_eff"]["reference"]) - float(
            values["luminosity_distance"]["reference"]
        )
        current_delta = float(values["chi_eff"]["current"]) - float(
            values["luminosity_distance"]["current"]
        )
        output.append(
            {
                "event": event,
                "locked_source": values["chi_eff"]["source"],
                "locked_chi_eff_nw1": values["chi_eff"]["reference"],
                "full_rows_chi_eff_nw1": values["chi_eff"]["current"],
                "chi_eff_nw1_difference": values["chi_eff"]["difference"],
                "locked_luminosity_distance_nw1": values[
                    "luminosity_distance"
                ]["reference"],
                "full_rows_luminosity_distance_nw1": values[
                    "luminosity_distance"
                ]["current"],
                "luminosity_distance_nw1_difference": values[
                    "luminosity_distance"
                ]["difference"],
                "locked_delta": reference_delta,
                "full_rows_delta": current_delta,
                "delta_sign_preserved": (reference_delta > 0) == (current_delta > 0),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    ensure_output(args.output_dir, args.overwrite)
    inventory_summary = json.loads(
        args.inventory_summary.read_text(encoding="utf-8")
    )
    catalog_complete = bool(inventory_summary.get("complete_and_verified"))
    if not catalog_complete and not args.allow_incomplete:
        raise RuntimeError(
            "The catalog inventory is not complete and verified. Use "
            "--allow-incomplete only for a provisional pipeline-validation run."
        )

    inventory = read_csv(args.inventory)
    verified_events = {
        row["event"]
        for row in inventory
        if row["inventory_status"] == "verified"
    }
    metadata = read_csv(args.metadata_summary)
    metadata_by_event = {row["event"]: row for row in metadata}
    if set(metadata_by_event) != verified_events:
        missing = sorted(verified_events.difference(metadata_by_event))
        extra = sorted(set(metadata_by_event).difference(verified_events))
        raise RuntimeError(
            f"Inventory/metadata event mismatch. Missing={missing}; extra={extra}"
        )

    outcome_rows = sorted(
        [row for row in metadata if bool_text(row["all_pair_outcome_usable"])],
        key=lambda row: row["event"],
    )
    parameter_rows, event_rows = analyze_events(outcome_rows)
    strict_rows = [row for row in event_rows if row["strict_population_member"]]

    population_rows = [
        population_summary(
            "strict_matched_settings", strict_rows, catalog_complete
        ),
        population_summary(
            "all_outcome_usable_pairs", event_rows, catalog_complete
        ),
    ]
    subgroup_rows: list[dict[str, Any]] = []
    for population, rows in (
        ("strict_matched_settings", strict_rows),
        ("all_outcome_usable_pairs", event_rows),
    ):
        for field in (
            "prior_role",
            "screening_score_tertile_104",
            "snr_tertile_104",
        ):
            subgroup_rows.extend(subgroup_summaries(population, rows, field))

    references = reference_lookup(
        (
            ("confirmatory_strict", args.confirmatory_reference),
            ("prospective_replication_strict", args.replication_reference),
        )
    )
    check_rows = consistency_rows(parameter_rows, references)

    write_csv(
        args.output_dir / "full_catalog_parameter_metrics.csv",
        parameter_rows,
        list(parameter_rows[0]),
    )
    write_csv(
        args.output_dir / "full_catalog_event_deltas.csv",
        event_rows,
        list(event_rows[0]),
    )
    write_csv(
        args.output_dir / "full_catalog_population_summaries.csv",
        population_rows,
        list(population_rows[0]),
    )
    write_csv(
        args.output_dir / "full_catalog_subgroup_summaries.csv",
        subgroup_rows,
        list(subgroup_rows[0]),
    )
    check_fields = [
        "event",
        "locked_source",
        "locked_chi_eff_nw1",
        "full_rows_chi_eff_nw1",
        "chi_eff_nw1_difference",
        "locked_luminosity_distance_nw1",
        "full_rows_luminosity_distance_nw1",
        "luminosity_distance_nw1_difference",
        "locked_delta",
        "full_rows_delta",
        "delta_sign_preserved",
    ]
    write_csv(
        args.output_dir / "locked_event_consistency_check.csv",
        check_rows,
        check_fields,
    )

    result_paths = sorted(
        path
        for path in args.output_dir.iterdir()
        if path.is_file() and path.name != "full_catalog_analysis_manifest.json"
    )
    manifest = {
        "analysis": "post-lock GWTC-5 full-catalog H3 generalization",
        "status": (
            "final_full_catalog" if catalog_complete else "provisional_incomplete_catalog"
        ),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_event_count_expected": EXPECTED_CATALOG_EVENTS,
        "checksum_verified_event_count": len(verified_events),
        "all_outcome_usable_event_count": len(event_rows),
        "strict_matched_settings_event_count": len(strict_rows),
        "locked_reference_events_recomputed": len(check_rows),
        "locked_reference_delta_signs_preserved": sum(
            bool(row["delta_sign_preserved"]) for row in check_rows
        ),
        "scope": {
            "model_families": [XPHM_FAMILY, XPNR_FAMILY],
            "configuration_tag_rule": "unique same-prefix event-specific pair (C00 or C01)",
            "parameters": list(PARAMETERS),
            "metric": "NW1 = W1 / mean(XPHM width90, XPNR width90)",
            "posterior_rows": "all finite released rows; no random cap",
            "tests": "post-lock descriptive/generalization only",
        },
        "environment": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
        },
        "input_hashes": {
            str(path.resolve()): sha256_file(path)
            for path in (
                args.inventory,
                args.inventory_summary,
                args.metadata_summary,
                args.confirmatory_reference,
                args.replication_reference,
            )
        },
        "analysis_code_sha256": sha256_file(Path(__file__).resolve()),
        "result_hashes": {
            path.name: sha256_file(path) for path in result_paths
        },
    }
    (args.output_dir / "full_catalog_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "populations": population_rows}, indent=2))


if __name__ == "__main__":
    main()
