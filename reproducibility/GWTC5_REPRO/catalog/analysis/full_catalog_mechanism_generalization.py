"""Run the locked exploratory mechanism analyses for the full catalog frame.

The script opens posterior values only after the metadata decisions have been
sealed. Joint analyses use every released row finite in both mass ratio and
effective spin; one-dimensional analyses use every finite row for that field.
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
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance


XPHM = "C00:IMRPhenomXPHM-SpinTaylor"
XPNR = "C00:IMRPhenomXPNR"
MODELS = (XPHM, XPNR)
DIRECTION_COUNT = 360


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
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


def width_90(values: np.ndarray) -> float:
    low, high = np.quantile(values, [0.05, 0.95])
    return float(high - low)


def alternate_metrics(
    core: ModuleType,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    primary = core.pair_metrics(x, y)
    width_x = width_90(x)
    width_y = width_90(y)
    raw_w1 = float(primary["wasserstein_1"])
    geometric = math.sqrt(width_x * width_y)
    larger = max(width_x, width_y)
    return {
        "nw1": float(primary["normalized_wasserstein_1"]),
        "nw1_geometric_width": raw_w1 / geometric if geometric > 0 else math.nan,
        "nw1_larger_width": raw_w1 / larger if larger > 0 else math.nan,
        "standardized_median_displacement": float(
            primary["standardized_median_displacement"]
        ),
        "ks_distance": float(ks_2samp(x, y, method="auto").statistic),
    }


def sliced_wasserstein(
    x_q: np.ndarray,
    x_chi: np.ndarray,
    y_q: np.ndarray,
    y_chi: np.ndarray,
    q_scale: float,
    chi_scale: float,
) -> tuple[float, float, float]:
    if q_scale <= 0 or chi_scale <= 0:
        return math.nan, math.nan, math.nan
    pooled_q_center = float(np.median(np.concatenate((x_q, y_q))))
    pooled_chi_center = float(np.median(np.concatenate((x_chi, y_chi))))
    x = np.column_stack(
        ((x_q - pooled_q_center) / q_scale, (x_chi - pooled_chi_center) / chi_scale)
    )
    y = np.column_stack(
        ((y_q - pooled_q_center) / q_scale, (y_chi - pooled_chi_center) / chi_scale)
    )
    angles = np.arange(DIRECTION_COUNT, dtype=np.float64) * math.pi / DIRECTION_COUNT
    distances = np.empty(DIRECTION_COUNT, dtype=np.float64)
    for start in range(0, DIRECTION_COUNT, 24):
        stop = min(start + 24, DIRECTION_COUNT)
        directions = np.column_stack(
            (np.cos(angles[start:stop]), np.sin(angles[start:stop]))
        )
        x_projections = x @ directions.T
        y_projections = y @ directions.T
        for offset in range(stop - start):
            distances[start + offset] = wasserstein_distance(
                x_projections[:, offset], y_projections[:, offset]
            )
    return (
        float(np.mean(distances)),
        float(np.min(distances)),
        float(np.max(distances)),
    )


def principal_alignment(
    x_q: np.ndarray,
    x_chi: np.ndarray,
    y_q: np.ndarray,
    y_chi: np.ndarray,
    q_scale: float,
    chi_scale: float,
    shift: np.ndarray,
) -> tuple[float, float, float]:
    x = np.column_stack(
        (
            (x_q - np.median(x_q)) / q_scale,
            (x_chi - np.median(x_chi)) / chi_scale,
        )
    )
    y = np.column_stack(
        (
            (y_q - np.median(y_q)) / q_scale,
            (y_chi - np.median(y_chi)) / chi_scale,
        )
    )
    covariance = 0.5 * (np.cov(x, rowvar=False) + np.cov(y, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    shift_norm = float(np.linalg.norm(shift))
    alignment = (
        abs(float(np.dot(shift / shift_norm, axis)))
        if shift_norm > 0
        else math.nan
    )
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    if angle < 0:
        angle += 180.0
    fraction = float(np.max(eigenvalues) / np.sum(eigenvalues))
    return alignment, angle, fraction


def read_model_arrays(
    posterior: h5py.Dataset,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    int,
]:
    names = set(posterior.dtype.names or ())
    weights = names.intersection(
        {
            "weight",
            "weights",
            "posterior_weight",
            "posterior_weights",
            "log_weight",
            "log_weights",
        }
    )
    if weights:
        raise RuntimeError(f"Explicit posterior weight fields found: {sorted(weights)}")
    required = {"mass_ratio", "chi_eff"}
    if not required.issubset(names):
        raise KeyError(f"Missing required joint fields: {sorted(required - names)}")
    q_raw = np.asarray(posterior.fields("mass_ratio")[:], dtype=np.float64).reshape(-1)
    chi_raw = np.asarray(posterior.fields("chi_eff")[:], dtype=np.float64).reshape(-1)
    q_finite = q_raw[np.isfinite(q_raw)]
    chi_finite = chi_raw[np.isfinite(chi_raw)]
    joint_mask = np.isfinite(q_raw) & np.isfinite(chi_raw)
    q_joint = q_raw[joint_mask]
    chi_joint = chi_raw[joint_mask]
    chi_p: np.ndarray | None = None
    if "chi_p" in names:
        raw = np.asarray(posterior.fields("chi_p")[:], dtype=np.float64).reshape(-1)
        chi_p = raw[np.isfinite(raw)]
        if chi_p.size < 4:
            chi_p = None
    if q_joint.size < 4:
        raise RuntimeError("Too few jointly finite mass-ratio/effective-spin rows.")
    return q_finite, chi_finite, q_joint, chi_joint, chi_p, int(posterior.shape[0])


def analyze_events(
    core: ModuleType,
    events: list[str],
    inventory_by_event: dict[str, dict[str, str]],
    model_labels_by_event: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        source_row = inventory_by_event[event]
        path = Path(source_row["actual_path"])
        arrays: dict[
            str,
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray | None,
                int,
            ],
        ] = {}
        with h5py.File(path, "r") as source:
            xphm_label, xpnr_label = model_labels_by_event[event]
            arrays[XPHM] = read_model_arrays(
                source[xphm_label]["posterior_samples"]
            )
            arrays[XPNR] = read_model_arrays(
                source[xpnr_label]["posterior_samples"]
            )

        x_q, x_chi, x_q_joint, x_chi_joint, x_chip, x_stored = arrays[XPHM]
        y_q, y_chi, y_q_joint, y_chi_joint, y_chip, y_stored = arrays[XPNR]
        q_metrics = alternate_metrics(core, x_q, y_q)
        chi_metrics = alternate_metrics(core, x_chi, y_chi)
        q_scale = 0.5 * (width_90(x_q_joint) + width_90(y_q_joint))
        chi_scale = 0.5 * (width_90(x_chi_joint) + width_90(y_chi_joint))
        sliced_mean, sliced_min, sliced_max = sliced_wasserstein(
            x_q_joint,
            x_chi_joint,
            y_q_joint,
            y_chi_joint,
            q_scale,
            chi_scale,
        )
        delta_q = (
            float(np.median(y_q_joint)) - float(np.median(x_q_joint))
        ) / q_scale
        delta_chi = (
            float(np.median(y_chi_joint)) - float(np.median(x_chi_joint))
        ) / chi_scale
        shift = np.asarray([delta_q, delta_chi], dtype=np.float64)
        shift_l1 = abs(delta_q) + abs(delta_chi)
        alignment, axis_angle, axis_fraction = principal_alignment(
            x_q_joint,
            x_chi_joint,
            y_q_joint,
            y_chi_joint,
            q_scale,
            chi_scale,
            shift,
        )
        row: dict[str, Any] = {
            "event": event,
            "prior_stage": source_row["prior_stage"],
            "population_A_nondevelopment": str(
                source_row["population_A_nondevelopment"]
            ).lower()
            == "true",
            "source_file": path.name,
            "xphm_label": xphm_label,
            "xpnr_label": xpnr_label,
            "xphm_stored_rows": x_stored,
            "xpnr_stored_rows": y_stored,
            "xphm_joint_finite_rows": int(x_q_joint.size),
            "xpnr_joint_finite_rows": int(y_q_joint.size),
            "mass_ratio_nw1": q_metrics["nw1"],
            "chi_eff_nw1": chi_metrics["nw1"],
            "joint_q_chi_eff_sliced_w1": sliced_mean,
            "joint_projection_min_w1": sliced_min,
            "joint_projection_max_w1": sliced_max,
            "standardized_median_shift_q": delta_q,
            "standardized_median_shift_chi_eff": delta_chi,
            "standardized_median_shift_norm": float(np.linalg.norm(shift)),
            "chi_eff_share_of_l1_shift": (
                abs(delta_chi) / shift_l1 if shift_l1 > 0 else math.nan
            ),
            "xphm_spearman_q_chi_eff": float(
                spearmanr(x_q_joint, x_chi_joint).statistic
            ),
            "xpnr_spearman_q_chi_eff": float(
                spearmanr(y_q_joint, y_chi_joint).statistic
            ),
            "absolute_shift_pc1_alignment": alignment,
            "pc1_angle_degrees": axis_angle,
            "pc1_variance_fraction": axis_fraction,
            "chi_p_available": x_chip is not None and y_chip is not None,
        }
        for metric_name, value in chi_metrics.items():
            row[f"chi_eff_{metric_name}"] = value
        if x_chip is not None and y_chip is not None:
            chip_metrics = alternate_metrics(core, x_chip, y_chip)
            row["xphm_chi_p_finite_rows"] = int(x_chip.size)
            row["xpnr_chi_p_finite_rows"] = int(y_chip.size)
            for metric_name, value in chip_metrics.items():
                row[f"chi_p_{metric_name}"] = value
                row[f"chi_eff_minus_chi_p_{metric_name}"] = (
                    chi_metrics[metric_name] - value
                )
        else:
            missing_rows.append(
                {
                    "event": event,
                    "parameter": "chi_p",
                    "xphm_available": x_chip is not None,
                    "xpnr_available": y_chip is not None,
                    "action": "retained_for_joint_core; omitted_from_chi_p_comparison",
                }
            )
        rows.append(row)
        print(json.dumps({"analyzed": index, "event": event}), flush=True)
    return rows, missing_rows


def finite_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = [float(row[key]) for row in rows if key in row and row[key] not in (None, "")]
    return np.asarray([value for value in values if math.isfinite(value)], dtype=float)


def summarize_population(
    name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    joint = finite_array(rows, "joint_q_chi_eff_sliced_w1")
    chi = finite_array(rows, "chi_eff_nw1")
    if joint.size != chi.size:
        raise RuntimeError("Joint and chi_eff result counts differ.")
    result: dict[str, Any] = {
        "population": name,
        "event_count": len(rows),
        "joint_eligible_count": int(joint.size),
        "joint_sliced_w1_median": float(np.median(joint)),
        "joint_sliced_w1_minimum": float(np.min(joint)),
        "joint_sliced_w1_maximum": float(np.max(joint)),
        "rho_joint_vs_chi_eff_nw1": float(spearmanr(joint, chi).statistic),
        "median_chi_eff_share_of_l1_shift": float(
            np.nanmedian(finite_array(rows, "chi_eff_share_of_l1_shift"))
        ),
        "median_absolute_shift_pc1_alignment": float(
            np.nanmedian(finite_array(rows, "absolute_shift_pc1_alignment"))
        ),
        "median_pc1_variance_fraction": float(
            np.nanmedian(finite_array(rows, "pc1_variance_fraction"))
        ),
    }
    chi_p_rows = [row for row in rows if bool(row.get("chi_p_available"))]
    result["chi_p_eligible_count"] = len(chi_p_rows)
    result["chi_eff_vs_chi_p"] = {}
    for metric_name in [
        "nw1",
        "nw1_geometric_width",
        "nw1_larger_width",
        "standardized_median_displacement",
        "ks_distance",
    ]:
        differences = finite_array(
            chi_p_rows, f"chi_eff_minus_chi_p_{metric_name}"
        )
        result["chi_eff_vs_chi_p"][metric_name] = {
            "eligible_count": int(differences.size),
            "positive_count": int(np.sum(differences > 0)),
            "zero_count": int(np.sum(differences == 0)),
            "negative_count": int(np.sum(differences < 0)),
            "median_difference": float(np.median(differences)),
            "minimum_difference": float(np.min(differences)),
            "maximum_difference": float(np.max(differences)),
        }
    return result


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir, args.overwrite)
    core = load_metric_core(args.metric_core_module)
    inventory_rows, metadata_rows = load_audited_inputs(
        args.inventory, args.metadata_summary, args.metadata_manifest
    )
    inventory_by_event: dict[str, dict[str, str]] = {}
    for raw_row in inventory_rows:
        row = dict(raw_row)
        row["population_A_nondevelopment"] = not row["prior_role"].startswith(
            "development_"
        )
        row["prior_stage"] = row["prior_role"]
        inventory_by_event[row["event"]] = row
    metadata_by_event = {row["event"]: row for row in metadata_rows}
    events = sorted(
        row["event"]
        for row in metadata_rows
        if row["strict_pair_status"] == "strict_tier_A_automated"
        and row["all_pair_outcome_usable"].lower() == "true"
    )
    if not events:
        raise RuntimeError("No Tier A events are sealed for analysis.")
    model_labels_by_event = {
        event: (
            metadata_by_event[event]["xphm_label"],
            metadata_by_event[event]["xpnr_label"],
        )
        for event in events
    }

    event_rows, missing_rows = analyze_events(
        core, events, inventory_by_event, model_labels_by_event
    )
    population_a = [row for row in event_rows if row["population_A_nondevelopment"]]
    summary = {
        "analysis": "locked exploratory full-catalog mechanism generalization",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "direction_count": DIRECTION_COUNT,
        "populations": {
            "A_nondevelopment": summarize_population(
                "A_nondevelopment", population_a
            ),
            "B_complete_catalog": summarize_population(
                "B_complete_catalog", event_rows
            ),
        },
        "missing_chi_p": missing_rows,
    }

    event_path = args.output_dir / "full_catalog_mechanism_event_metrics.csv"
    missing_path = args.output_dir / "full_catalog_mechanism_missing_parameters.csv"
    summary_path = args.output_dir / "full_catalog_mechanism_summary.json"
    fields = sorted({key for row in event_rows for key in row})
    write_csv(event_path, event_rows, fields)
    write_csv(
        missing_path,
        missing_rows,
        ["event", "parameter", "xphm_available", "xpnr_available", "action"],
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version,
        },
        "constants": {"direction_count": DIRECTION_COUNT},
        "source_hashes": {
            str(args.inventory.resolve()): sha256_file(args.inventory),
            str(args.metadata_summary.resolve()): sha256_file(args.metadata_summary),
            str(args.metadata_manifest.resolve()): sha256_file(args.metadata_manifest),
            str(args.metric_core_module.resolve()): sha256_file(
                args.metric_core_module
            ),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
        "result_hashes": {
            event_path.name: sha256_file(event_path),
            missing_path.name: sha256_file(missing_path),
            summary_path.name: sha256_file(summary_path),
        },
    }
    (args.output_dir / "full_catalog_mechanism_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
