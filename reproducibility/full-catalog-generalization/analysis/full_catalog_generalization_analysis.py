"""Run the locked Population A and Population B generalization analyses.

This script refuses to open posterior values until a final metadata decision
table has been sealed. It imports a frozen copy of the original metric/test
implementations so the full-frame stage does not silently redefine them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import h5py
import numpy as np
import scipy
from scipy.stats import binomtest, ks_2samp, spearmanr, wilcoxon


XPHM = "C00:IMRPhenomXPHM-SpinTaylor"
XPNR = "C00:IMRPhenomXPNR"
MODELS = [XPHM, XPNR]
CORE_PARAMETERS = ["chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance"]
SCREENING_PARAMETERS = ["chirp_mass_source", "chi_eff", "luminosity_distance"]
ALL_CORE_PARAMETERS = list(dict.fromkeys(CORE_PARAMETERS + SCREENING_PARAMETERS))
G2_ROBUSTNESS_PARAMETERS = ["chi_eff", "luminosity_distance"]
N_PERMUTATIONS = 100_000
N_BOOTSTRAP = 100_000
N_SPLITS = 100
N_MATCHED_SUBSAMPLES = 100
PERMUTATION_SEED = 20260803
SPLIT_SEED = 20260802
BOOTSTRAP_SEED = 20260825
ALPHA = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--metadata-summary", type=Path, required=True)
    parser.add_argument("--metadata-manifest", type=Path, required=True)
    parser.add_argument("--metric-core-module", type=Path, required=True)
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


def stable_seed(base: int, *parts: str) -> int:
    payload = "\x1f".join([str(base), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_metric_core(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("frozen_metric_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.XPHM != XPHM or module.XPNR != XPNR:
        raise RuntimeError("Metric core uses unexpected model labels.")
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


def load_audited_inputs(
    inventory_path: Path,
    metadata_summary_path: Path,
    metadata_manifest_path: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    inventory = read_csv(inventory_path)
    summaries = read_csv(metadata_summary_path)
    manifest = json.loads(metadata_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("posterior_values_loaded") is not False:
        raise RuntimeError("Metadata audit does not certify outcome-blind execution.")
    if manifest.get("posterior_metrics_calculated") is not False:
        raise RuntimeError("Metadata audit reports posterior metrics.")
    if len(inventory) != 104 or len(summaries) != 104:
        raise RuntimeError("Inventory and metadata summary must each contain 104 events.")
    if {row["event"] for row in inventory} != {row["event"] for row in summaries}:
        raise RuntimeError("Inventory and metadata-summary event sets differ.")
    allowed = {"strict_tier_A_automated", "non_tier_A_metadata_difference"}
    observed = {row["strict_pair_status"] for row in summaries}
    if not observed.issubset(allowed):
        raise RuntimeError(f"Unexpected metadata status values: {sorted(observed - allowed)}")
    return inventory, summaries


def load_samples(
    selected_events: list[str],
    inventory_by_event: dict[str, dict[str, str]],
    model_labels_by_event: dict[str, tuple[str, str]],
) -> tuple[dict[tuple[str, str, str], np.ndarray], list[dict[str, Any]]]:
    samples: dict[tuple[str, str, str], np.ndarray] = {}
    preflight: list[dict[str, Any]] = []
    for number, event in enumerate(selected_events, start=1):
        path = Path(inventory_by_event[event]["actual_path"])
        with h5py.File(path, "r") as source:
            xphm_label, xpnr_label = model_labels_by_event[event]
            for role, model_label in ((XPHM, xphm_label), (XPNR, xpnr_label)):
                posterior = source[model_label]["posterior_samples"]
                names = set(posterior.dtype.names or [])
                weight_names = {
                    "weight",
                    "weights",
                    "posterior_weight",
                    "posterior_weights",
                    "log_weight",
                    "log_weights",
                }
                found_weights = sorted(names.intersection(weight_names))
                if found_weights:
                    raise RuntimeError(
                        f"Explicit weights found for {event}/{model}: {found_weights}"
                    )
                for parameter in ALL_CORE_PARAMETERS:
                    if parameter not in names:
                        raise KeyError(f"Missing {event}/{model}/{parameter}")
                    values = np.asarray(
                        posterior.fields(parameter)[:], dtype=np.float64
                    ).reshape(-1)
                    finite = values[np.isfinite(values)]
                    if finite.size < 4:
                        raise RuntimeError(
                            f"Too few finite values for {event}/{model}/{parameter}"
                        )
                    samples[(event, role, parameter)] = finite
                    preflight.append(
                        {
                            "event": event,
                            "model_role": role,
                            "model_label": model_label,
                            "parameter": parameter,
                            "source_file": path.name,
                            "stored_row_count": int(posterior.shape[0]),
                            "finite_sample_count": int(finite.size),
                            "sampling_method": "all_finite_released_rows",
                            "explicit_weight_field_found": False,
                        }
                    )
        print(json.dumps({"loaded": number, "event": event}), flush=True)
    return samples, preflight


def percentile_width(values: np.ndarray) -> float:
    q05, q95 = np.quantile(values, [0.05, 0.95])
    return float(q95 - q05)


def robustness_metrics(
    parent: ModuleType,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    width_x = percentile_width(x)
    width_y = percentile_width(y)
    w1 = float(scipy.stats.wasserstein_distance(x, y))
    geometric = math.sqrt(width_x * width_y)
    larger = max(width_x, width_y)
    return {
        "nw1_geometric_width": w1 / geometric if geometric > 0 else math.nan,
        "nw1_larger_width": w1 / larger if larger > 0 else math.nan,
        "ks_distance": float(ks_2samp(x, y, method="auto").statistic),
    }


def compute_parameter_metrics(
    parent: ModuleType,
    events: list[str],
    samples: dict[tuple[str, str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        for parameter in ALL_CORE_PARAMETERS:
            x = samples[(event, XPHM, parameter)]
            y = samples[(event, XPNR, parameter)]
            rows.append(
                {
                    "event": event,
                    "parameter": parameter,
                    **parent.pair_metrics(x, y),
                    **robustness_metrics(parent, x, y),
                }
            )
    return rows


def compute_split_references(
    parent: ModuleType,
    events: list[str],
    samples: dict[tuple[str, str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replicate_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for event in events:
        for parameter in G2_ROBUSTNESS_PARAMETERS:
            by_model: dict[str, list[float]] = {}
            for model in MODELS:
                values = samples[(event, model, parameter)]
                half = values.size // 2
                rng = np.random.default_rng(
                    stable_seed(SPLIT_SEED, event, model, parameter, "half_split")
                )
                distances: list[float] = []
                for replicate in range(1, N_SPLITS + 1):
                    order = rng.permutation(values.size)
                    first = values[order[:half]]
                    second = values[order[half : 2 * half]]
                    value = float(parent.split_normalized_w1(first, second))
                    distances.append(value)
                    replicate_rows.append(
                        {
                            "event": event,
                            "model": model,
                            "parameter": parameter,
                            "replicate": replicate,
                            "half_sample_size": half,
                            "split_normalized_wasserstein_1": value,
                        }
                    )
                by_model[model] = distances
            x_p95 = float(np.quantile(by_model[XPHM], 0.95))
            y_p95 = float(np.quantile(by_model[XPNR], 0.95))
            threshold_rows.append(
                {
                    "event": event,
                    "parameter": parameter,
                    "xphm_split_NW1_p95": x_p95,
                    "xpnr_split_NW1_p95": y_p95,
                    "pair_noise_threshold_NW1": max(x_p95, y_p95),
                }
            )
    return replicate_rows, threshold_rows


def matched_size_nw1(
    parent: ModuleType,
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> float:
    size = min(x.size, y.size)
    x_sub = x if x.size == size else x[np.sort(rng.choice(x.size, size, replace=False))]
    y_sub = y if y.size == size else y[np.sort(rng.choice(y.size, size, replace=False))]
    return float(parent.pair_metrics(x_sub, y_sub)["normalized_wasserstein_1"])


def compute_matched_size_sensitivity(
    parent: ModuleType,
    events: list[str],
    samples: dict[tuple[str, str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replicate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for event in events:
        for parameter in G2_ROBUSTNESS_PARAMETERS:
            x = samples[(event, XPHM, parameter)]
            y = samples[(event, XPNR, parameter)]
            rng = np.random.default_rng(
                stable_seed(BOOTSTRAP_SEED, event, parameter, "matched_size")
            )
            values: list[float] = []
            for replicate in range(1, N_MATCHED_SUBSAMPLES + 1):
                value = matched_size_nw1(parent, x, y, rng)
                values.append(value)
                replicate_rows.append(
                    {
                        "event": event,
                        "parameter": parameter,
                        "replicate": replicate,
                        "matched_sample_size": min(x.size, y.size),
                        "matched_size_normalized_wasserstein_1": value,
                    }
                )
            summary_rows.append(
                {
                    "event": event,
                    "parameter": parameter,
                    "matched_sample_size": min(x.size, y.size),
                    "matched_size_NW1_median": float(np.median(values)),
                    "matched_size_NW1_p05": float(np.quantile(values, 0.05)),
                    "matched_size_NW1_p95": float(np.quantile(values, 0.95)),
                }
            )
    return replicate_rows, summary_rows


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = x.size
    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)
        rho = float(spearmanr(x[indices], y[indices]).statistic)
        if math.isfinite(rho):
            values.append(rho)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper), len(values)


def bootstrap_median(
    values: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    medians: list[np.ndarray] = []
    completed = 0
    while completed < N_BOOTSTRAP:
        current = min(5_000, N_BOOTSTRAP - completed)
        indices = rng.integers(0, values.size, size=(current, values.size))
        medians.append(np.median(values[indices], axis=1))
        completed += current
    all_medians = np.concatenate(medians)
    lower, upper = np.quantile(all_medians, [0.025, 0.975])
    return float(lower), float(upper)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [math.nan] * count
    running_min = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = count - reverse_rank + 1
        candidate = min(1.0, p_values[index] * count / rank)
        running_min = min(running_min, candidate)
        adjusted[index] = running_min
    return adjusted


def run_heterogeneity_grid(
    endpoints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate the locked 6-by-3 Population A exploratory grid."""
    predictors = [
        ("log_network_matched_filter_snr", lambda row: math.log(row["network_matched_filter_snr"])),
        ("total_mass_source", lambda row: row["total_mass_source"]),
        ("chirp_mass_source", lambda row: row["chirp_mass_source"]),
        ("absolute_catalog_chi_eff", lambda row: abs(row["catalog_chi_eff"])),
        ("catalog_luminosity_distance", lambda row: row["catalog_luminosity_distance"]),
        ("maximum_screening_shift", lambda row: row["maximum_screening_shift"]),
    ]
    outcomes = ["event_median_NW1", "chi_eff_nw1", "delta_nw1"]
    rows: list[dict[str, Any]] = []
    for predictor_name, extractor in predictors:
        x = np.asarray([extractor(row) for row in endpoints], dtype=float)
        for outcome in outcomes:
            y = np.asarray([row[outcome] for row in endpoints], dtype=float)
            result = spearmanr(x, y)
            rows.append(
                {
                    "population": "A_nondevelopment",
                    "scope": "prespecified_exploratory_heterogeneity",
                    "predictor": predictor_name,
                    "outcome": outcome,
                    "event_count": len(endpoints),
                    "spearman_rho": float(result.statistic),
                    "raw_two_sided_p_value": float(result.pvalue),
                }
            )
    adjusted = benjamini_hochberg(
        [float(row["raw_two_sided_p_value"]) for row in rows]
    )
    for row, value in zip(rows, adjusted, strict=True):
        row["benjamini_hochberg_adjusted_p_value"] = value
        row["reject_fdr_0_05"] = bool(value < ALPHA)
    return rows


def build_event_endpoints(
    metrics: list[dict[str, Any]],
    inventory_by_event: dict[str, dict[str, Any]],
    split_thresholds: list[dict[str, Any]],
    matched_summary: list[dict[str, Any]],
    samples: dict[tuple[str, str, str], np.ndarray],
) -> list[dict[str, Any]]:
    metric_by_key = {(row["event"], row["parameter"]): row for row in metrics}
    threshold_by_key = {
        (row["event"], row["parameter"]): row for row in split_thresholds
    }
    matched_by_key = {
        (row["event"], row["parameter"]): row for row in matched_summary
    }
    events = sorted({row["event"] for row in metrics})
    output: list[dict[str, Any]] = []

    for event in events:
        core_nw1 = [
            float(metric_by_key[(event, parameter)]["normalized_wasserstein_1"])
            for parameter in CORE_PARAMETERS
        ]
        screen_nw1 = [
            float(metric_by_key[(event, parameter)]["normalized_wasserstein_1"])
            for parameter in SCREENING_PARAMETERS
        ]
        row: dict[str, Any] = {
            "event": event,
            "event_median_NW1": float(np.median(core_nw1)),
            "event_max_NW1": float(np.max(core_nw1)),
            "screening_parameter_max_NW1": float(np.max(screen_nw1)),
        }
        for column, name in [
            ("normalized_wasserstein_1", "nw1"),
            ("nw1_geometric_width", "geometric"),
            ("nw1_larger_width", "larger"),
            ("ks_distance", "ks"),
        ]:
            row[f"chi_eff_{name}"] = float(
                metric_by_key[(event, "chi_eff")][column]
            )
            row[f"luminosity_distance_{name}"] = float(
                metric_by_key[(event, "luminosity_distance")][column]
            )
            row[f"delta_{name}"] = (
                row[f"chi_eff_{name}"] - row[f"luminosity_distance_{name}"]
            )

        x = samples[(event, XPHM, "luminosity_distance")]
        y = samples[(event, XPNR, "luminosity_distance")]
        if np.any(x <= 0) or np.any(y <= 0):
            raise RuntimeError(f"Nonpositive luminosity distance for {event}")
        wx = percentile_width(np.log(x))
        wy = percentile_width(np.log(y))
        raw = float(scipy.stats.wasserstein_distance(np.log(x), np.log(y)))
        row["luminosity_distance_log_nw1"] = raw / (0.5 * (wx + wy))
        row["delta_log_distance"] = (
            row["chi_eff_nw1"] - row["luminosity_distance_log_nw1"]
        )

        chi_threshold = float(
            threshold_by_key[(event, "chi_eff")]["pair_noise_threshold_NW1"]
        )
        distance_threshold = float(
            threshold_by_key[(event, "luminosity_distance")][
                "pair_noise_threshold_NW1"
            ]
        )
        row["chi_eff_split_ratio"] = row["chi_eff_nw1"] / chi_threshold
        row["luminosity_distance_split_ratio"] = (
            row["luminosity_distance_nw1"] / distance_threshold
        )
        row["delta_split_ratio"] = (
            row["chi_eff_split_ratio"] - row["luminosity_distance_split_ratio"]
        )
        row["chi_eff_matched_size_nw1"] = float(
            matched_by_key[(event, "chi_eff")]["matched_size_NW1_median"]
        )
        row["luminosity_distance_matched_size_nw1"] = float(
            matched_by_key[(event, "luminosity_distance")][
                "matched_size_NW1_median"
            ]
        )
        row["delta_matched_size"] = (
            row["chi_eff_matched_size_nw1"]
            - row["luminosity_distance_matched_size_nw1"]
        )
        source = inventory_by_event[event]
        row.update(
            {
                "population_A_nondevelopment": bool(
                    source["population_A_nondevelopment"]
                ),
                "prior_stage": source["prior_stage"],
                "network_matched_filter_snr": float(
                    source["network_matched_filter_snr"]
                ),
                "maximum_screening_shift": float(
                    source["maximum_screening_shift"]
                ),
                "total_mass_source": float(source["total_mass_source"]),
                "chirp_mass_source": float(source["chirp_mass_source"]),
                "catalog_chi_eff": float(source["chi_eff"]),
                "catalog_luminosity_distance": float(
                    source["luminosity_distance"]
                ),
            }
        )
        output.append(row)
    return output


def run_population_tests(
    parent: ModuleType,
    population_name: str,
    endpoints: list[dict[str, Any]],
    inferential: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not endpoints:
        raise RuntimeError(f"Population {population_name} is empty.")
    snr = np.asarray(
        [row["network_matched_filter_snr"] for row in endpoints], dtype=float
    )
    median_nw1 = np.asarray(
        [row["event_median_NW1"] for row in endpoints], dtype=float
    )
    score = np.asarray(
        [row["maximum_screening_shift"] for row in endpoints], dtype=float
    )
    screen_max = np.asarray(
        [row["screening_parameter_max_NW1"] for row in endpoints], dtype=float
    )
    chi = np.asarray([row["chi_eff_nw1"] for row in endpoints], dtype=float)
    distance = np.asarray(
        [row["luminosity_distance_nw1"] for row in endpoints], dtype=float
    )
    delta = chi - distance

    permutation_rng = np.random.default_rng(PERMUTATION_SEED)
    g1_rho, g1_p = parent.spearman_permutation_test(
        snr, median_nw1, permutation_rng, n_permutations=N_PERMUTATIONS
    )
    g3_rho, g3_p = parent.spearman_permutation_test(
        score, screen_max, permutation_rng, n_permutations=N_PERMUTATIONS
    )
    g2 = wilcoxon(
        chi,
        distance,
        alternative="greater",
        zero_method="wilcox",
        method="auto",
    )
    positive = int(np.sum(delta > 0))
    zero = int(np.sum(delta == 0))
    negative = int(np.sum(delta < 0))
    sign = binomtest(positive, positive + negative, 0.5, alternative="greater")
    rho_low, rho_high, rho_valid = bootstrap_spearman(
        snr,
        median_nw1,
        stable_seed(BOOTSTRAP_SEED, population_name, "g1_bootstrap"),
    )
    median_low, median_high = bootstrap_median(
        delta,
        stable_seed(BOOTSTRAP_SEED, population_name, "g2_bootstrap"),
    )
    raw = {"G1": float(g1_p), "G2": float(g2.pvalue), "G3": float(g3_p)}
    adjusted = parent.holm_adjust(raw) if inferential else {key: math.nan for key in raw}

    tests = [
        {
            "population": population_name,
            "scope": "locked_generalization" if inferential else "descriptive_sensitivity",
            "endpoint": "G1",
            "test": "one-sided Spearman permutation",
            "event_count": len(endpoints),
            "effect": g1_rho,
            "raw_p_value": g1_p,
            "holm_adjusted_p_value": adjusted["G1"],
            "reject_holm_0_05": bool(inferential and adjusted["G1"] < ALPHA),
            "effect_interval_low": rho_low,
            "effect_interval_high": rho_high,
            "interval_method": f"event bootstrap; {rho_valid} finite of {N_BOOTSTRAP}",
        },
        {
            "population": population_name,
            "scope": "locked_generalization" if inferential else "descriptive_sensitivity",
            "endpoint": "G2",
            "test": "one-sided Wilcoxon signed-rank",
            "event_count": len(endpoints),
            "effect": float(np.median(delta)),
            "effect_iqr_low": float(np.quantile(delta, 0.25)),
            "effect_iqr_high": float(np.quantile(delta, 0.75)),
            "test_statistic": float(g2.statistic),
            "raw_p_value": float(g2.pvalue),
            "holm_adjusted_p_value": adjusted["G2"],
            "reject_holm_0_05": bool(inferential and adjusted["G2"] < ALPHA),
            "effect_interval_low": median_low,
            "effect_interval_high": median_high,
            "interval_method": f"event bootstrap median; {N_BOOTSTRAP} resamples",
            "positive_count": positive,
            "zero_count": zero,
            "negative_count": negative,
            "positive_fraction_nonzero": positive / (positive + negative),
            "sign_test_p_value": float(sign.pvalue),
        },
        {
            "population": population_name,
            "scope": "locked_generalization" if inferential else "descriptive_sensitivity",
            "endpoint": "G3",
            "test": "one-sided Spearman permutation",
            "event_count": len(endpoints),
            "effect": g3_rho,
            "raw_p_value": g3_p,
            "holm_adjusted_p_value": adjusted["G3"],
            "reject_holm_0_05": bool(inferential and adjusted["G3"] < ALPHA),
        },
    ]

    robustness: list[dict[str, Any]] = []
    for name in [
        "delta_nw1",
        "delta_geometric",
        "delta_larger",
        "delta_log_distance",
        "delta_ks",
        "delta_split_ratio",
        "delta_matched_size",
    ]:
        values = np.asarray([row[name] for row in endpoints], dtype=float)
        robustness.append(
            {
                "population": population_name,
                "representation": name,
                "event_count": values.size,
                "positive_count": int(np.sum(values > 0)),
                "zero_count": int(np.sum(values == 0)),
                "negative_count": int(np.sum(values < 0)),
                "median_difference": float(np.median(values)),
                "minimum_difference": float(np.min(values)),
                "maximum_difference": float(np.max(values)),
            }
        )
    return tests, robustness


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    parent = load_metric_core(args.metric_core_module)
    inventory_rows, metadata_rows = load_audited_inputs(
        args.inventory, args.metadata_summary, args.metadata_manifest
    )
    ranking_rows = read_csv(args.ranking)
    ranking_by_event = {row["event"]: row for row in ranking_rows}
    if len(ranking_rows) != 104 or len(ranking_by_event) != 104:
        raise RuntimeError("Ranking input must contain 104 unique events.")
    catalog_fields = [
        "total_mass_source",
        "chirp_mass_source",
        "chi_eff",
        "luminosity_distance",
    ]
    inventory_by_event: dict[str, dict[str, Any]] = {}
    for raw_row in inventory_rows:
        event = raw_row["event"]
        if event not in ranking_by_event:
            raise RuntimeError(f"Inventory event is absent from ranking: {event}")
        row: dict[str, Any] = dict(raw_row)
        row["population_A_nondevelopment"] = not row["prior_role"].startswith(
            "development_"
        )
        row["prior_stage"] = row["prior_role"]
        for column in ["network_matched_filter_snr", "maximum_screening_shift"]:
            row[column] = float(row[column])
        for column in catalog_fields:
            row[column] = float(ranking_by_event[event][column])
        inventory_by_event[event] = row
    metadata_by_event = {row["event"]: row for row in metadata_rows}
    tier_a = sorted(
        row["event"]
        for row in metadata_rows
        if row["strict_pair_status"] == "strict_tier_A_automated"
        and row["all_pair_outcome_usable"].lower() == "true"
    )
    if not tier_a:
        raise RuntimeError("No Tier A events are sealed for analysis.")
    model_labels_by_event = {
        event: (
            metadata_by_event[event]["xphm_label"],
            metadata_by_event[event]["xpnr_label"],
        )
        for event in tier_a
    }

    samples, preflight_rows = load_samples(
        tier_a, inventory_by_event, model_labels_by_event
    )
    parameter_rows = compute_parameter_metrics(parent, tier_a, samples)
    split_rows, threshold_rows = compute_split_references(parent, tier_a, samples)
    matched_rows, matched_summary_rows = compute_matched_size_sensitivity(
        parent, tier_a, samples
    )
    endpoints = build_event_endpoints(
        parameter_rows,
        inventory_by_event,
        threshold_rows,
        matched_summary_rows,
        samples,
    )

    population_a = [row for row in endpoints if row["population_A_nondevelopment"]]
    population_b = list(endpoints)
    tests_a, robustness_a = run_population_tests(
        parent, "A_nondevelopment", population_a, inferential=True
    )
    tests_b, robustness_b = run_population_tests(
        parent, "B_complete_catalog", population_b, inferential=False
    )
    heterogeneity_rows = run_heterogeneity_grid(population_a)

    write_csv(args.output_dir / "full_catalog_preflight_samples.csv", preflight_rows, list(preflight_rows[0]))
    write_csv(args.output_dir / "full_catalog_parameter_metrics.csv", parameter_rows, list(parameter_rows[0]))
    write_csv(args.output_dir / "full_catalog_split_replicates.csv", split_rows, list(split_rows[0]))
    write_csv(args.output_dir / "full_catalog_split_references.csv", threshold_rows, list(threshold_rows[0]))
    write_csv(args.output_dir / "full_catalog_matched_size_replicates.csv", matched_rows, list(matched_rows[0]))
    write_csv(args.output_dir / "full_catalog_matched_size_summary.csv", matched_summary_rows, list(matched_summary_rows[0]))
    write_csv(
        args.output_dir / "full_catalog_event_endpoints.csv",
        endpoints,
        list(endpoints[0]),
    )
    test_rows = tests_a + tests_b
    write_csv(args.output_dir / "full_catalog_generalization_tests.csv", test_rows, sorted({key for row in test_rows for key in row}))
    robust_rows = robustness_a + robustness_b
    write_csv(args.output_dir / "full_catalog_h3_robustness_summary.csv", robust_rows, list(robust_rows[0]))
    write_csv(
        args.output_dir / "full_catalog_heterogeneity_grid.csv",
        heterogeneity_rows,
        list(heterogeneity_rows[0]),
    )

    manifest = {
        "analysis": "locked GWTC-5 full-catalog generalization",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "population_A_tier_A_count": len(population_a),
        "population_B_tier_A_count": len(population_b),
        "constants": {
            "n_permutations": N_PERMUTATIONS,
            "n_bootstrap": N_BOOTSTRAP,
            "n_splits": N_SPLITS,
            "n_matched_subsamples": N_MATCHED_SUBSAMPLES,
            "permutation_seed": PERMUTATION_SEED,
            "split_seed": SPLIT_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "alpha": ALPHA,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
        },
        "source_hashes": {
            str(args.inventory.resolve()): sha256_file(args.inventory),
            str(args.ranking.resolve()): sha256_file(args.ranking),
            str(args.metadata_summary.resolve()): sha256_file(args.metadata_summary),
            str(args.metadata_manifest.resolve()): sha256_file(args.metadata_manifest),
            str(args.metric_core_module.resolve()): sha256_file(args.metric_core_module),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
    }
    result_paths = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    manifest["result_hashes"] = {path.name: sha256_file(path) for path in result_paths}
    (args.output_dir / "full_catalog_generalization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tests": tests_a, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
