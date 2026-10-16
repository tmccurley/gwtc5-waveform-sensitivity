"""Dependency-light copy of the frozen parent metric and test definitions.

The executable formulas, constants, and directional permutation logic are
copied from `GWTC5_confirmatory_analysis.py`. This module omits plotting and
table-management dependencies so the full-catalog run can use the already
verified NumPy, SciPy, and h5py environment.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata, wasserstein_distance


XPHM = "C00:IMRPhenomXPHM-SpinTaylor"
XPNR = "C00:IMRPhenomXPNR"
N_BINS_JSD = 60


def interval_summary(values: np.ndarray) -> dict[str, float]:
    lower, median, upper = np.quantile(values, [0.05, 0.50, 0.95])
    return {
        "lower_90": float(lower),
        "median": float(median),
        "upper_90": float(upper),
        "width_90": float(upper - lower),
        "variance": float(np.var(values, ddof=1)),
    }


def js_divergence_bits(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = N_BINS_JSD,
) -> float:
    pooled_min = float(min(np.min(x), np.min(y)))
    pooled_max = float(max(np.max(x), np.max(y)))
    if pooled_min == pooled_max:
        return 0.0

    edges = np.linspace(pooled_min, pooled_max, bins + 1)
    px = np.histogram(x, bins=edges)[0].astype(np.float64)
    py = np.histogram(y, bins=edges)[0].astype(np.float64)
    px /= px.sum()
    py /= py.sum()
    midpoint = 0.5 * (px + py)

    def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

    return 0.5 * kl_bits(px, midpoint) + 0.5 * kl_bits(py, midpoint)


def pair_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | bool]:
    sx = interval_summary(x)
    sy = interval_summary(y)

    average_width = 0.5 * (sx["width_90"] + sy["width_90"])
    raw_w1 = float(wasserstein_distance(x, y))
    normalized_w1 = raw_w1 / average_width if average_width > 0 else math.nan

    median_denominator = math.sqrt(sx["variance"] + sy["variance"])
    standardized_median = (
        abs(sx["median"] - sy["median"]) / median_denominator
        if median_denominator > 0
        else math.nan
    )

    intersection = max(
        0.0,
        min(sx["upper_90"], sy["upper_90"])
        - max(sx["lower_90"], sy["lower_90"]),
    )
    union = max(sx["upper_90"], sy["upper_90"]) - min(
        sx["lower_90"], sy["lower_90"]
    )
    overlap = intersection / union if union > 0 else 1.0

    log_width_ratio = (
        math.log(sx["width_90"] / sy["width_90"])
        if sx["width_90"] > 0 and sy["width_90"] > 0
        else math.nan
    )

    return {
        "xphm_sample_count": int(x.size),
        "xpnr_sample_count": int(y.size),
        "xphm_median": sx["median"],
        "xpnr_median": sy["median"],
        "xphm_lower_90": sx["lower_90"],
        "xphm_upper_90": sx["upper_90"],
        "xpnr_lower_90": sy["lower_90"],
        "xpnr_upper_90": sy["upper_90"],
        "xphm_width_90": sx["width_90"],
        "xpnr_width_90": sy["width_90"],
        "average_width_90": average_width,
        "wasserstein_1": raw_w1,
        "normalized_wasserstein_1": normalized_w1,
        "js_divergence_bits": js_divergence_bits(x, y),
        "standardized_median_displacement": standardized_median,
        "interval_overlap_90": overlap,
        "log_width_ratio": log_width_ratio,
    }


def equal_size_w1(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size:
        raise ValueError("equal_size_w1 requires equal sample sizes")
    return float(np.mean(np.abs(np.sort(x) - np.sort(y))))


def split_normalized_w1(first: np.ndarray, second: np.ndarray) -> float:
    s1 = interval_summary(first)
    s2 = interval_summary(second)
    average_width = 0.5 * (s1["width_90"] + s2["width_90"])
    if average_width <= 0:
        return math.nan
    return equal_size_w1(first, second) / average_width


def spearman_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int = 100_000,
    alternative: str = "greater",
    chunk_size: int = 5_000,
) -> tuple[float, float]:
    if x.size != y.size:
        raise ValueError("x and y must have equal length")
    if alternative != "greater":
        raise NotImplementedError("Only the frozen greater alternative is used")

    xr = rankdata(x).astype(np.float64)
    yr = rankdata(y).astype(np.float64)
    xc = xr - xr.mean()
    yc = yr - yr.mean()
    denominator = math.sqrt(float(np.dot(xc, xc) * np.dot(yc, yc)))
    if denominator == 0:
        raise RuntimeError("Spearman statistic is undefined for constant input")
    observed = float(np.dot(xc, yc) / denominator)

    exceedances = 0
    completed = 0
    n = x.size
    while completed < n_permutations:
        current = min(chunk_size, n_permutations - completed)
        order = np.argsort(rng.random((current, n)), axis=1)
        permuted_y = yc[order]
        statistics = permuted_y @ xc / denominator
        exceedances += int(np.count_nonzero(statistics >= observed))
        completed += current

    p_value = (exceedances + 1) / (n_permutations + 1)
    return observed, float(p_value)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for index, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (m - index) * p_value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted
