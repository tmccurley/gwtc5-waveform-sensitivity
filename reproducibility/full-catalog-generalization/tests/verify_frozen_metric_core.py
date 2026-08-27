"""Verify the frozen metric copy against the archived confirmatory outputs.

This regression check recalculates every parent metric for the 18 previously
analyzed confirmatory events from the official event files. It does not inspect
any outcome from the newly acquired portion of the full-catalog frame.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path
from types import ModuleType

import h5py
import numpy as np


METRIC_FIELDS = [
    "xphm_sample_count",
    "xpnr_sample_count",
    "xphm_median",
    "xpnr_median",
    "xphm_lower_90",
    "xphm_upper_90",
    "xpnr_lower_90",
    "xpnr_upper_90",
    "xphm_width_90",
    "xpnr_width_90",
    "average_width_90",
    "wasserstein_1",
    "normalized_wasserstein_1",
    "js_divergence_bits",
    "standardized_median_displacement",
    "interval_overlap_90",
    "log_width_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-core-module", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("frozen_metric_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_same(name: str, actual: float, expected: float) -> None:
    if math.isnan(actual) and math.isnan(expected):
        return
    if not math.isclose(actual, expected, rel_tol=2e-13, abs_tol=2e-13):
        raise AssertionError(f"{name}: actual={actual!r}, expected={expected!r}")


def main() -> None:
    args = parse_args()
    core = load_module(args.metric_core_module)
    inventory = {row["event"]: row for row in read_csv(args.inventory)}
    reference_rows = read_csv(args.reference_metrics)
    events = sorted({row["event"] for row in reference_rows})
    if len(events) != 18 or len(reference_rows) != 90:
        raise RuntimeError("Expected 18 events and 90 parameter rows in reference.")

    checked = 0
    for reference in reference_rows:
        event = reference["event"]
        parameter = reference["parameter"]
        source_path = inventory[event].get("canonical_local_path") or inventory[
            event
        ].get("actual_path")
        if not source_path:
            raise KeyError(f"Inventory lacks a local path for {event}")
        path = Path(source_path)
        with h5py.File(path, "r") as source:
            x = np.asarray(
                source[core.XPHM]["posterior_samples"].fields(parameter)[:],
                dtype=np.float64,
            ).reshape(-1)
            y = np.asarray(
                source[core.XPNR]["posterior_samples"].fields(parameter)[:],
                dtype=np.float64,
            ).reshape(-1)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        actual = core.pair_metrics(x, y)
        for field in METRIC_FIELDS:
            expected = float(reference[field])
            assert_same(f"{event}/{parameter}/{field}", float(actual[field]), expected)
        checked += 1

    print(
        f"PASS: {checked} event-parameter rows and "
        f"{checked * len(METRIC_FIELDS)} archived values match."
    )


if __name__ == "__main__":
    main()
