"""Fast smoke tests for the dependency-light generalization pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    analysis = load(
        "full_catalog_generalization_analysis",
        ROOT / "analysis" / "full_catalog_generalization_analysis.py",
    )
    core = load("frozen_metric_core", ROOT / "analysis" / "frozen_metric_core.py")

    assert analysis.benjamini_hochberg([0.01, 0.04, 0.03]) == [0.03, 0.04, 0.04]
    analysis.N_PERMUTATIONS = 199
    analysis.N_BOOTSTRAP = 199
    endpoints = []
    for index in range(12):
        value = float(index + 1)
        endpoints.append(
            {
                "network_matched_filter_snr": 8.0 + value,
                "event_median_NW1": 0.01 + value / 1_000,
                "maximum_screening_shift": 0.02 + value / 2_000,
                "screening_parameter_max_NW1": 0.03 + value / 1_500,
                "chi_eff_nw1": 0.08 + value / 1_000,
                "luminosity_distance_nw1": 0.02 + value / 2_000,
                "delta_nw1": 0.06 + value / 2_000,
                "delta_geometric": 0.05 + value / 2_000,
                "delta_larger": 0.04 + value / 2_000,
                "delta_log_distance": 0.03 + value / 2_000,
                "delta_ks": 0.02 + value / 2_000,
                "delta_split_ratio": 0.5 + value / 100,
                "delta_matched_size": 0.06 + value / 2_000,
                "total_mass_source": 20.0 + value,
                "chirp_mass_source": 10.0 + value / 2,
                "catalog_chi_eff": -0.3 + value / 20,
                "catalog_luminosity_distance": 500.0 + 10 * value,
            }
        )

    tests, robustness = analysis.run_population_tests(
        core, "synthetic", endpoints, inferential=True
    )
    grid = analysis.run_heterogeneity_grid(endpoints)
    assert len(tests) == 3
    assert len(robustness) == 7
    assert len(grid) == 18
    assert all(0.0 <= float(row["holm_adjusted_p_value"]) <= 1.0 for row in tests)
    assert all(
        0.0 <= float(row["benjamini_hochberg_adjusted_p_value"]) <= 1.0
        for row in grid
    )
    print("PASS: dependency-light pipeline smoke tests")


if __name__ == "__main__":
    main()
