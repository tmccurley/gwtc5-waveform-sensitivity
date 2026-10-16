"""Create publication-ready PNG figures from sealed full-catalog result tables.

This presentation-only program reads completed CSV/JSON outputs; it does not
open posterior files or recalculate scientific results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#FFFFFF"
TEXT = "#17202A"
MUTED = "#5D6D7E"
GRID = "#DCE3E8"
BLUE = "#2878B5"
ORANGE = "#E07A1F"
GREEN = "#2A9D6F"
RED = "#C44536"
PURPLE = "#7B5AA6"
LIGHT_LINE = "#AEB8C2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-endpoints", type=Path, required=True)
    parser.add_argument("--generalization-tests", type=Path, required=True)
    parser.add_argument("--mechanism-events", type=Path, required=True)
    parser.add_argument("--mechanism-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    if not path.is_file():
        raise FileNotFoundError(f"Required font not found: {path}")
    return ImageFont.truetype(str(path), size=size)


def pretty_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    if magnitude >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def padded_limits(values: list[float], include_zero: bool = False) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    if span == 0:
        span = max(abs(low), 1.0)
    return low - 0.06 * span, high + 0.06 * span


def rotated_label(
    image: Image.Image,
    center: tuple[int, int],
    text_value: str,
    text_font: ImageFont.FreeTypeFont,
) -> None:
    box = text_font.getbbox(text_value)
    width = box[2] - box[0] + 20
    height = box[3] - box[1] + 20
    layer = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((width / 2, height / 2), text_value, font=text_font, fill=TEXT, anchor="mm")
    layer = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(layer, (center[0] - layer.width // 2, center[1] - layer.height // 2))


def draw_axes(
    image: Image.Image,
    panel: tuple[int, int, int, int],
    x_values: list[float],
    y_values: list[float],
    x_label: str,
    y_label: str,
    panel_letter: str,
    title: str,
    include_x_zero: bool = False,
    include_y_zero: bool = False,
    draw_x_tick_labels: bool = True,
) -> tuple[Callable[[float], float], Callable[[float], float], tuple[int, int, int, int]]:
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = panel
    plot = (left + 145, top + 105, right - 45, bottom - 125)
    p_left, p_top, p_right, p_bottom = plot
    x_low, x_high = padded_limits(x_values, include_x_zero)
    y_low, y_high = padded_limits(y_values, include_y_zero)

    sx = lambda value: p_left + (value - x_low) * (p_right - p_left) / (x_high - x_low)
    sy = lambda value: p_bottom - (value - y_low) * (p_bottom - p_top) / (y_high - y_low)
    label_font = font(34)
    tick_font = font(27)
    for index in range(6):
        fraction = index / 5
        x_value = x_low + fraction * (x_high - x_low)
        y_value = y_low + fraction * (y_high - y_low)
        x_pixel = sx(x_value)
        y_pixel = sy(y_value)
        draw.line((x_pixel, p_top, x_pixel, p_bottom), fill=GRID, width=2)
        draw.line((p_left, y_pixel, p_right, y_pixel), fill=GRID, width=2)
        if draw_x_tick_labels:
            draw.text((x_pixel, p_bottom + 18), pretty_number(x_value), font=tick_font, fill=MUTED, anchor="ma")
        draw.text((p_left - 18, y_pixel), pretty_number(y_value), font=tick_font, fill=MUTED, anchor="rm")
    draw.line((p_left, p_bottom, p_right, p_bottom), fill=TEXT, width=4)
    draw.line((p_left, p_top, p_left, p_bottom), fill=TEXT, width=4)
    draw.text(((p_left + p_right) / 2, bottom - 48), x_label, font=label_font, fill=TEXT, anchor="mm")
    rotated_label(image, (left + 45, (p_top + p_bottom) // 2), y_label, label_font)
    draw.text((left + 18, top + 20), panel_letter, font=font(43, True), fill=TEXT, anchor="la")
    draw.text((left + 80, top + 27), title, font=font(34, True), fill=TEXT, anchor="la")
    return sx, sy, plot


def stage_color(stage: str) -> str:
    if stage in {"confirmatory_final", "confirmatory_strict"}:
        return ORANGE
    if stage in {"replication_final", "prospective_replication_strict"}:
        return GREEN
    return BLUE


def annotation_box(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    lines: list[str],
) -> None:
    left, top, _, _ = plot
    text_value = "\n".join(lines)
    text_font = font(26)
    box = draw.multiline_textbbox((left + 20, top + 18), text_value, font=text_font, spacing=7)
    draw.rounded_rectangle((box[0] - 12, box[1] - 9, box[2] + 12, box[3] + 9), radius=10, fill="#FFFFFFE8", outline=LIGHT_LINE, width=2)
    draw.multiline_text((left + 20, top + 18), text_value, font=text_font, fill=TEXT, spacing=7)


def test_by_endpoint(rows: list[dict[str, str]], endpoint: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["population"] == "A_nondevelopment" and row["endpoint"] == endpoint
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Population A test row for {endpoint}")
    return matches[0]


def make_generalization_figure(
    endpoints: list[dict[str, str]],
    tests: list[dict[str, str]],
    output: Path,
) -> None:
    rows = [row for row in endpoints if row["population_A_nondevelopment"].lower() == "true"]
    image = Image.new("RGBA", (2500, 1850), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((1250, 48), "Full non-development catalog generalization", font=font(54, True), fill=TEXT, anchor="ma")
    draw.text((1250, 113), "Every point is one metadata-eligible GWTC-5 event", font=font(30), fill=MUTED, anchor="ma")
    panels = [
        (40, 170, 1240, 980),
        (1260, 170, 2460, 980),
        (40, 1000, 1240, 1810),
        (1260, 1000, 2460, 1810),
    ]

    snr = [float(row["network_matched_filter_snr"]) for row in rows]
    median_nw1 = [float(row["event_median_NW1"]) for row in rows]
    sx, sy, plot = draw_axes(
        image, panels[0], snr, median_nw1,
        "Catalog network matched-filter SNR", "Median NW1 across four parameters", "A", "SNR association"
    )
    for row, x, y in zip(rows, snr, median_nw1, strict=True):
        color = stage_color(row["prior_stage"])
        draw.ellipse((sx(x) - 10, sy(y) - 10, sx(x) + 10, sy(y) + 10), fill=color, outline=BACKGROUND, width=2)
    g1 = test_by_endpoint(tests, "G1")
    annotation_box(draw, plot, [f"Spearman rho = {float(g1['effect']):.3f}", f"Holm-adjusted p = {float(g1['holm_adjusted_p_value']):.3g}"])

    distance = [float(row["luminosity_distance_nw1"]) for row in rows]
    chi = [float(row["chi_eff_nw1"]) for row in rows]
    shared = distance + chi
    sx, sy, plot = draw_axes(
        image, panels[1], shared, shared,
        "Luminosity-distance NW1", "Effective-spin NW1", "B", "Parameter ordering"
    )
    low, high = min(shared), max(shared)
    draw.line((sx(low), sy(low), sx(high), sy(high)), fill=LIGHT_LINE, width=4)
    for row, x, y in zip(rows, distance, chi, strict=True):
        color = stage_color(row["prior_stage"])
        draw.ellipse((sx(x) - 10, sy(y) - 10, sx(x) + 10, sy(y) + 10), fill=color, outline=BACKGROUND, width=2)
    g2 = test_by_endpoint(tests, "G2")
    annotation_box(draw, plot, [f"chi_eff > distance: {g2['positive_count']}/{g2['event_count']}", f"Median delta = {float(g2['effect']):.3f}", f"Holm-adjusted p = {float(g2['holm_adjusted_p_value']):.3g}"])

    screening = [float(row["maximum_screening_shift"]) for row in rows]
    observed = [float(row["screening_parameter_max_NW1"]) for row in rows]
    sx, sy, plot = draw_axes(
        image, panels[2], screening, observed,
        "Catalog screening score", "Maximum complete-posterior NW1", "C", "Screening validation"
    )
    for row, x, y in zip(rows, screening, observed, strict=True):
        color = stage_color(row["prior_stage"])
        draw.ellipse((sx(x) - 10, sy(y) - 10, sx(x) + 10, sy(y) + 10), fill=color, outline=BACKGROUND, width=2)
    g3 = test_by_endpoint(tests, "G3")
    annotation_box(draw, plot, [f"Spearman rho = {float(g3['effect']):.3f}", f"Holm-adjusted p = {float(g3['holm_adjusted_p_value']):.3g}"])

    ordered = sorted(
        ((float(row["delta_nw1"]), row) for row in rows), key=lambda item: item[0]
    )
    ranks = list(range(1, len(ordered) + 1))
    deltas = [item[0] for item in ordered]
    sx, sy, plot = draw_axes(
        image, panels[3], [float(value) for value in ranks], deltas,
        "Events ranked by paired difference", "chi_eff NW1 minus distance NW1", "D", "Event-level exceptions", include_y_zero=True,
        draw_x_tick_labels=False,
    )
    p_left, _, p_right, p_bottom = plot
    draw.text((sx(1), p_bottom + 18), "1", font=font(27), fill=MUTED, anchor="ma")
    draw.text((sx(len(ranks)), p_bottom + 18), str(len(ranks)), font=font(27), fill=MUTED, anchor="ma")
    draw.line((p_left, sy(0.0), p_right, sy(0.0)), fill=TEXT, width=3)
    for rank, (delta, row) in zip(ranks, ordered, strict=True):
        color = GREEN if delta > 0 else RED if delta < 0 else MUTED
        draw.line((sx(rank), sy(0.0), sx(rank), sy(delta)), fill=color, width=4)
        draw.ellipse((sx(rank) - 7, sy(delta) - 7, sx(rank) + 7, sy(delta) + 7), fill=color)
    annotation_box(draw, plot, [f"Minimum delta = {min(deltas):.3f}", f"Maximum delta = {max(deltas):.3f}"])

    legend_y = 154
    legend_items = [(BLUE, "new full-frame event"), (ORANGE, "original confirmatory event"), (GREEN, "replication event")]
    x_cursor = 570
    for color, label in legend_items:
        draw.ellipse((x_cursor, legend_y - 9, x_cursor + 18, legend_y + 9), fill=color)
        draw.text((x_cursor + 28, legend_y), label, font=font(25), fill=MUTED, anchor="lm")
        x_cursor += 390 if color == BLUE else 430

    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, dpi=(300, 300), optimize=True)


def make_mechanism_figure(
    event_rows: list[dict[str, str]],
    summary: dict[str, Any],
    output: Path,
) -> None:
    rows = [row for row in event_rows if row["population_A_nondevelopment"].lower() == "true"]
    image = Image.new("RGBA", (2700, 1080), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((1350, 42), "Exploratory mechanism generalization", font=font(52, True), fill=TEXT, anchor="ma")
    draw.text((1350, 105), "Joint movement and effective-spin versus precession disagreement", font=font(29), fill=MUTED, anchor="ma")
    panels = [(30, 150, 890, 1040), (920, 150, 1780, 1040), (1810, 150, 2670, 1040)]

    chi = [float(row["chi_eff_nw1"]) for row in rows]
    joint = [float(row["joint_q_chi_eff_sliced_w1"]) for row in rows]
    sx, sy, plot = draw_axes(image, panels[0], chi, joint, "Effective-spin NW1", "Joint (q, chi_eff) sliced W1", "A", "Marginal and joint movement")
    for row, x, y in zip(rows, chi, joint, strict=True):
        color = stage_color(row["prior_stage"])
        draw.ellipse((sx(x) - 9, sy(y) - 9, sx(x) + 9, sy(y) + 9), fill=color, outline=BACKGROUND, width=2)
    population = summary["populations"]["A_nondevelopment"]
    annotation_box(draw, plot, [f"Spearman rho = {float(population['rho_joint_vs_chi_eff_nw1']):.3f}", f"n = {population['joint_eligible_count']}"])

    chip_rows = [row for row in rows if row["chi_p_available"].lower() == "true"]
    chip = [float(row["chi_p_nw1"]) for row in chip_rows]
    chie = [float(row["chi_eff_nw1"]) for row in chip_rows]
    sx, sy, plot = draw_axes(
        image,
        panels[1],
        [-0.15, 1.15],
        chip + chie,
        "Parameter",
        "NW1",
        "B",
        "Effective versus precession spin",
        draw_x_tick_labels=False,
    )
    for left_value, right_value in zip(chip, chie, strict=True):
        color = GREEN if right_value > left_value else RED
        draw.line((sx(0.0), sy(left_value), sx(1.0), sy(right_value)), fill="#C8D0D7", width=2)
        draw.ellipse((sx(0.0) - 7, sy(left_value) - 7, sx(0.0) + 7, sy(left_value) + 7), fill=PURPLE)
        draw.ellipse((sx(1.0) - 7, sy(right_value) - 7, sx(1.0) + 7, sy(right_value) + 7), fill=color)
    p_left, _, p_right, p_bottom = plot
    draw.text((sx(0.0), p_bottom + 24), "chi_p", font=font(29), fill=TEXT, anchor="ma")
    draw.text((sx(1.0), p_bottom + 24), "chi_eff", font=font(29), fill=TEXT, anchor="ma")
    chip_summary = population["chi_eff_vs_chi_p"]["nw1"]
    annotation_box(draw, plot, [f"chi_eff > chi_p: {chip_summary['positive_count']}/{chip_summary['eligible_count']}", f"Median difference = {float(chip_summary['median_difference']):.3f}"])

    shares = [float(row["chi_eff_share_of_l1_shift"]) for row in rows]
    alignments = [float(row["absolute_shift_pc1_alignment"]) for row in rows]
    sx, sy, plot = draw_axes(image, panels[2], shares, alignments, "chi_eff share of median-shift magnitude", "Absolute shift–PC1 alignment", "C", "Geometry of the joint shift")
    for row, x, y in zip(rows, shares, alignments, strict=True):
        color = stage_color(row["prior_stage"])
        draw.ellipse((sx(x) - 9, sy(y) - 9, sx(x) + 9, sy(y) + 9), fill=color, outline=BACKGROUND, width=2)
    annotation_box(draw, plot, [f"Median chi_eff share = {float(population['median_chi_eff_share_of_l1_shift']):.3f}", f"Median alignment = {float(population['median_absolute_shift_pc1_alignment']):.3f}"])

    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, dpi=(300, 300), optimize=True)


def main() -> None:
    args = parse_args()
    endpoints = read_csv(args.event_endpoints)
    tests = read_csv(args.generalization_tests)
    mechanism_rows = read_csv(args.mechanism_events)
    mechanism_summary = json.loads(args.mechanism_summary.read_text(encoding="utf-8"))
    make_generalization_figure(
        endpoints, tests, args.output_dir / "fig_full_catalog_generalization.png"
    )
    make_mechanism_figure(
        mechanism_rows,
        mechanism_summary,
        args.output_dir / "fig_full_catalog_mechanism.png",
    )
    print(f"Created two 300-DPI figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
