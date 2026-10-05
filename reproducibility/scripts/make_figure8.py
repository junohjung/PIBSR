#!/usr/bin/env python3
"""Recreate Figure 8 and its cavity discretization metrics from fixed data."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
FIELD_SPECS = (("u", r"$u$"), ("v", r"$v$"), ("p", r"$p$"))
ROW_LABELS = ("Higher-resolution\nreference", "Coarse\nobservations", "Absolute\ndifference")
ERROR_POWER_GAMMA = 0.55

@dataclass(frozen=True)
class Frame:
    step: int
    time: float
    coarse_index: int
    fine_path: Path
    matched_path: Path
    coarse_path: Path
    fine_table: np.ndarray
    matched_table: np.ndarray
    coarse_table: np.ndarray


@dataclass(frozen=True)
class PreparedFrame:
    source: Frame
    fine_x: np.ndarray
    fine_y: np.ndarray
    matched_x: np.ndarray
    matched_y: np.ndarray
    fine_fields: dict[str, np.ndarray]
    matched_reference_fields: dict[str, np.ndarray]
    coarse_fields: dict[str, np.ndarray]
    absolute_difference: dict[str, np.ndarray]
    pressure_alignment: dict[str, float]


def load_table(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required complete-run frame is absent: {path}")
    values = np.loadtxt(path, dtype=np.float64)
    if values.shape != shape:
        raise RuntimeError(f"Frame schema drift for {path}: observed={values.shape}, expected={shape}")
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Frame contains nonfinite values: {path}")
    return values


def table_to_fields(table: np.ndarray, count: int, has_time: bool) -> dict[str, np.ndarray]:
    start = 3 if has_time else 2
    return {
        "u": table[:, start].reshape(count, count),
        "v": table[:, start + 1].reshape(count, count),
        "p": table[:, start + 2].reshape(count, count),
    }


def pressure_tolerance(values: np.ndarray) -> float:
    return 256.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(values))))


def prepare_frames(frames: list[Frame]) -> list[PreparedFrame]:
    fine_nodes = np.linspace(0.0, 1.0, 201, dtype=np.float64)
    matched_nodes = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    prepared: list[PreparedFrame] = []
    for frame in frames:
        fine_fields = table_to_fields(frame.fine_table, 201, has_time=True)
        matched_reference = table_to_fields(frame.matched_table, 21, has_time=True)
        coarse = table_to_fields(frame.coarse_table, 21, has_time=False)

        reference_mean_before = float(np.mean(matched_reference["p"]))
        coarse_mean_before = float(np.mean(coarse["p"]))
        matched_reference = {name: values.copy() for name, values in matched_reference.items()}
        coarse = {name: values.copy() for name, values in coarse.items()}
        matched_reference["p"] -= reference_mean_before
        coarse["p"] -= coarse_mean_before
        reference_mean_after = float(np.mean(matched_reference["p"]))
        coarse_mean_after = float(np.mean(coarse["p"]))
        if abs(reference_mean_after) > pressure_tolerance(matched_reference["p"]):
            raise RuntimeError(f"Reference pressure gauge alignment failed at t={frame.time:g}")
        if abs(coarse_mean_after) > pressure_tolerance(coarse["p"]):
            raise RuntimeError(f"Coarse pressure gauge alignment failed at t={frame.time:g}")

        difference = {
            name: np.abs(matched_reference[name] - coarse[name]) for name, _ in FIELD_SPECS
        }
        prepared.append(
            PreparedFrame(
                source=frame,
                fine_x=fine_nodes,
                fine_y=fine_nodes,
                matched_x=matched_nodes,
                matched_y=matched_nodes,
                fine_fields=fine_fields,
                matched_reference_fields=matched_reference,
                coarse_fields=coarse,
                absolute_difference=difference,
                pressure_alignment={
                    "reference_matched_mean_before": reference_mean_before,
                    "coarse_mean_before": coarse_mean_before,
                    "reference_matched_mean_after": reference_mean_after,
                    "coarse_mean_after": coarse_mean_after,
                },
            )
        )
    return prepared


def compute_metrics(frames: list[PreparedFrame]) -> dict[str, Any]:
    x, y = np.meshgrid(frames[0].matched_x, frames[0].matched_y, indexing="xy")
    interior = (x > 0.0) & (x < 1.0) & (y > 0.0) & (y < 1.0)
    rows: list[dict[str, Any]] = []
    for frame in frames:
        for field_name, _ in FIELD_SPECS:
            reference = frame.matched_reference_fields[field_name]
            coarse = frame.coarse_fields[field_name]
            error = coarse - reference
            reference_l2 = float(np.linalg.norm(reference.ravel()))
            error_l2 = float(np.linalg.norm(error.ravel()))
            relative_l2 = error_l2 / reference_l2 if reference_l2 > 0.0 else None
            rows.append(
                {
                    "time": frame.source.time,
                    "high_step": frame.source.step,
                    "coarse_index": frame.source.coarse_index,
                    "field": field_name,
                    "rms": float(np.sqrt(np.mean(error**2))),
                    "interior_rms": float(np.sqrt(np.mean(error[interior] ** 2))),
                    "relative_l2": relative_l2,
                    "linf": float(np.max(np.abs(error))),
                    "reference_l2": reference_l2,
                    "error_l2": error_l2,
                    "sample_count": int(error.size),
                    "interior_sample_count": int(np.count_nonzero(interior)),
                    "pressure_gauge_aligned_on_matched_grid": field_name == "p",
                }
            )
    return {
        "schema": "pimfa.ramped_cavity.figure14.metrics.v1",
        "definition": {
            "grid": "common endpoint-inclusive 21 x 21 Cartesian reporting grid",
            "difference": "coarse observation minus higher-resolution reference",
            "rms": "sqrt(mean(error^2)) over all 441 matched nodes",
            "interior_rms": "sqrt(mean(error^2)) over the 19 x 19 strict-interior nodes",
            "relative_l2": "||error||_2 / ||higher-resolution reference||_2 on 441 nodes",
            "linf": "max(abs(error)) on 441 nodes",
            "pressure": (
                "reference and coarse pressure independently shifted to zero arithmetic mean "
                "on the common 21 x 21 grid before all pressure metrics"
            ),
        },
        "rows": rows,
    }


def nondegenerate_limits(minimum: float, maximum: float) -> tuple[float, float]:
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise RuntimeError("Plot scale contains a nonfinite limit")
    if minimum < maximum:
        return minimum, maximum
    padding = max(1.0e-15, abs(minimum) * 1.0e-12)
    return minimum - padding, maximum + padding


def plot_limits(frames: list[PreparedFrame]) -> dict[str, dict[str, float]]:
    limits: dict[str, dict[str, float]] = {}
    for field_name, _ in FIELD_SPECS:
        field_min = min(
            min(float(np.min(frame.fine_fields[field_name])), float(np.min(frame.coarse_fields[field_name])))
            for frame in frames
        )
        field_max = max(
            max(float(np.max(frame.fine_fields[field_name])), float(np.max(frame.coarse_fields[field_name])))
            for frame in frames
        )
        field_min, field_max = nondegenerate_limits(field_min, field_max)
        difference_max = max(
            float(np.max(frame.absolute_difference[field_name])) for frame in frames
        )
        if not np.isfinite(difference_max) or difference_max < 0.0:
            raise RuntimeError("Difference scale is invalid")
        display_difference_max = difference_max if difference_max > 0.0 else 1.0e-15
        limits[field_name] = {
            "field_min": field_min,
            "field_max": field_max,
            "absolute_difference_max": difference_max,
            "absolute_difference_display_max": display_difference_max,
        }
    return limits


def vorticity(x: np.ndarray, y: np.ndarray, fields: dict[str, np.ndarray]) -> np.ndarray:
    return np.gradient(fields["v"], x, axis=1, edge_order=2) - np.gradient(
        fields["u"], y, axis=0, edge_order=2
    )


def add_vorticity_contours(
    axis: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    fields: dict[str, np.ndarray],
    time_value: float,
) -> None:
    # The prescribed ramp makes t=0.01 nearly quiescent; omitting contours in
    # only that predeclared first column avoids visually magnifying roundoff.
    if time_value <= 0.01:
        return
    omega = vorticity(x, y, fields)
    finite = omega[np.isfinite(omega)]
    if finite.size < 2 or np.ptp(finite) == 0.0:
        return
    levels = np.unique(np.quantile(finite, np.linspace(0.06, 0.94, 13)))
    if levels.size >= 2:
        axis.contour(
            x,
            y,
            omega,
            levels=levels,
            colors="black",
            linewidths=0.16,
            alpha=0.50,
        )


def draw_panel(
    axis: mpl.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
) -> mpl.image.AxesImage:
    image = axis.imshow(
        values,
        origin="lower",
        extent=(float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
        interpolation="bilinear",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return image


def make_pdf(frames: list[PreparedFrame], limits: dict[str, dict[str, float]], output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.linewidth": 0.35,
        }
    )
    figure = plt.figure(figsize=(8.0, 12.38))
    grid = figure.add_gridspec(
        11,
        6,
        width_ratios=(1, 1, 1, 1, 1, 0.035),
        height_ratios=(1, 1, 1, 0.11, 1, 1, 1, 0.11, 1, 1, 1),
        left=0.145,
        right=0.925,
        bottom=0.038,
        top=0.972,
        wspace=0.038,
        hspace=0.038,
    )
    image_cmap = mpl.colormaps["turbo"]
    error_cmap = mpl.colormaps["magma"]
    block_starts = (0, 4, 8)

    for block, (field_name, _) in enumerate(FIELD_SPECS):
        start = block_starts[block]
        field_limits = limits[field_name]
        value_norm = Normalize(
            vmin=field_limits["field_min"], vmax=field_limits["field_max"], clip=False
        )
        error_norm = PowerNorm(
            gamma=ERROR_POWER_GAMMA,
            vmin=0.0,
            vmax=field_limits["absolute_difference_display_max"],
            clip=False,
        )
        row_images: list[mpl.image.AxesImage] = []
        for row_offset in range(3):
            last_image: mpl.image.AxesImage | None = None
            for column_index, frame in enumerate(frames):
                axis = figure.add_subplot(grid[start + row_offset, column_index])
                if row_offset == 0:
                    shown = frame.fine_fields[field_name]
                    last_image = draw_panel(
                        axis, frame.fine_x, frame.fine_y, shown, image_cmap, value_norm
                    )
                    if field_name in ("u", "v"):
                        add_vorticity_contours(
                            axis,
                            frame.fine_x,
                            frame.fine_y,
                            frame.fine_fields,
                            frame.source.time,
                        )
                elif row_offset == 1:
                    shown = frame.coarse_fields[field_name]
                    last_image = draw_panel(
                        axis, frame.matched_x, frame.matched_y, shown, image_cmap, value_norm
                    )
                    if field_name in ("u", "v"):
                        add_vorticity_contours(
                            axis,
                            frame.matched_x,
                            frame.matched_y,
                            frame.coarse_fields,
                            frame.source.time,
                        )
                else:
                    shown = frame.absolute_difference[field_name]
                    last_image = draw_panel(
                        axis, frame.matched_x, frame.matched_y, shown, error_cmap, error_norm
                    )

                if block == 0 and row_offset == 0:
                    axis.set_title(rf"$t={frame.source.time:g}$", fontsize=10, pad=5.0)
                if column_index == 0:
                    axis.set_ylabel(ROW_LABELS[row_offset], rotation=90, labelpad=37, fontsize=8.2)
                    axis.text(
                        -0.15,
                        0.5,
                        r"$y$",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                        fontsize=9,
                    )
                if block == 2 and row_offset == 2:
                    axis.set_xlabel(r"$x$", labelpad=2.0, fontsize=9)
            if last_image is None:
                raise AssertionError("No image was drawn for a Figure 14 row")
            row_images.append(last_image)

        field_color_axis = figure.add_subplot(grid[start : start + 2, 5])
        position = field_color_axis.get_position()
        field_color_axis.set_position(
            [position.x0, position.y0 + 0.004, position.width, position.height - 0.004]
        )
        field_colorbar = figure.colorbar(row_images[0], cax=field_color_axis)
        field_colorbar.set_ticks((field_limits["field_min"], field_limits["field_max"]))
        field_colorbar.ax.tick_params(labelsize=7, width=0.35, length=2, pad=1)
        field_colorbar.outline.set_linewidth(0.35)

        error_color_axis = figure.add_subplot(grid[start + 2, 5])
        position = error_color_axis.get_position()
        error_color_axis.set_position(
            [position.x0, position.y0, position.width, position.height - 0.004]
        )
        error_colorbar = figure.colorbar(row_images[2], cax=error_color_axis)
        error_ticks = (0.0, field_limits["absolute_difference_display_max"])
        error_colorbar.set_ticks(error_ticks)
        error_colorbar.ax.tick_params(labelsize=7, width=0.35, length=2, pad=1)
        error_colorbar.outline.set_linewidth(0.35)

        figure.text(
            0.972,
            (0.82, 0.505, 0.19)[block],
            rf"$\mathbf{{{field_name}}}$",
            ha="left",
            va="center",
            fontsize=11,
        )

    figure.text(
        0.145,
        0.018,
        r"$x\rightarrow\qquad y\uparrow$",
        ha="left",
        va="center",
        fontsize=7.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime(2026, 8, 6, tzinfo=dt.timezone.utc)
    figure.savefig(
        output,
        format="pdf",
        dpi=300,
        metadata={
            "Title": "Smooth-ramped cavity higher-resolution reference and coarse observations",
            "Author": "PI-MFA authors",
            "Subject": (
                "Higher-resolution reference, coarse observations, and matched-grid absolute "
                "differences at five predeclared times"
            ),
            "Creator": SCRIPT_PATH.name,
            "CreationDate": timestamp,
            "ModDate": timestamp,
        },
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/figures")
    args = parser.parse_args()
    data = ROOT / "figure_data/cavity_comparison"
    entries = json.loads((data / "manifest.json").read_text())
    for entry in entries:
        if hashlib.sha256((data / entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Input checksum mismatch: " + entry["path"])
    frames = []
    for time in (0.01, 2.5, 5., 7.5, 10.):
        paths = {e["role"]: data / e["path"] for e in entries if e["time"] == time}
        fine, matched, coarse = [paths[k] for k in ("fine_reference", "matched_reference", "coarse_observation")]
        frames.append(Frame(round(time * 1000), time, round(time * 100), fine, matched, coarse,
                            load_table(fine, (40401, 6)), load_table(matched, (441, 6)),
                            load_table(coarse, (441, 5))))
    prepared = prepare_frames(frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_pdf(prepared, plot_limits(prepared), args.output_dir / "Figure8.pdf")
    (args.output_dir / "Figure8_metrics.json").write_text(json.dumps(compute_metrics(prepared), indent=2) + "\n")
    print(args.output_dir.resolve())

if __name__ == "__main__":
    main()
