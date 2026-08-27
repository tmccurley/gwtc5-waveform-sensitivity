"""Render synthetic previews to catch clipping and layout failures."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_figures():
    path = ROOT / "figures" / "make_full_catalog_figures.py"
    spec = importlib.util.spec_from_file_location("full_catalog_figures", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    figures = load_figures()
    endpoints = []
    mechanisms = []
    for index in range(36):
        stage = (
            "confirmatory_final"
            if index < 8
            else "replication_final"
            if index < 14
            else "full_frame"
        )
        distance = 0.012 + (index % 11) * 0.004
        chi = 0.026 + (index % 13) * 0.006
        endpoints.append(
            {
                "population_A_nondevelopment": "True",
                "prior_stage": stage,
                "network_matched_filter_snr": str(8 + index * 0.35),
                "event_median_NW1": str(0.018 + index * 0.0014),
                "luminosity_distance_nw1": str(distance),
                "chi_eff_nw1": str(chi),
                "delta_nw1": str(chi - distance),
                "maximum_screening_shift": str(0.01 + index * 0.003),
                "screening_parameter_max_NW1": str(0.015 + index * 0.0025),
            }
        )
        mechanisms.append(
            {
                "population_A_nondevelopment": "True",
                "prior_stage": stage,
                "chi_eff_nw1": str(chi),
                "joint_q_chi_eff_sliced_w1": str(0.01 + chi * 0.75),
                "chi_p_available": "True",
                "chi_p_nw1": str(chi * 0.38),
                "chi_eff_share_of_l1_shift": str(0.35 + (index % 12) * 0.045),
                "absolute_shift_pc1_alignment": str(0.2 + (index % 10) * 0.075),
            }
        )
    tests = [
        {
            "population": "A_nondevelopment",
            "endpoint": endpoint,
            "effect": effect,
            "holm_adjusted_p_value": p_value,
            "positive_count": "34",
            "event_count": "36",
        }
        for endpoint, effect, p_value in [
            ("G1", "0.42", "0.012"),
            ("G2", "0.051", "1.2e-8"),
            ("G3", "0.71", "0.0002"),
        ]
    ]
    summary = {
        "populations": {
            "A_nondevelopment": {
                "rho_joint_vs_chi_eff_nw1": 0.91,
                "joint_eligible_count": 36,
                "median_chi_eff_share_of_l1_shift": 0.72,
                "median_absolute_shift_pc1_alignment": 0.61,
                "chi_eff_vs_chi_p": {
                    "nw1": {
                        "positive_count": 35,
                        "eligible_count": 36,
                        "median_difference": 0.043,
                    }
                },
            }
        }
    }
    output = ROOT / "tests" / "rendered_previews"
    figures.make_generalization_figure(
        endpoints, tests, output / "synthetic_generalization.png"
    )
    figures.make_mechanism_figure(
        mechanisms, summary, output / "synthetic_mechanism.png"
    )
    print(f"PASS: rendered previews in {output}")


if __name__ == "__main__":
    main()
