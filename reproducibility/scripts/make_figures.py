#!/usr/bin/env python3
"""Create the supported figures from the packaged numerical records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPRO_ROOT / "work" / "figures"

FIGURE_GROUPS = {
    "one_d": (
        "one_d_observations",
        "one_d_weight_selection",
        "one_d_balance_histories",
    ),
    "burgers": (
        "burgers_discrepancy",
        "burgers_weight_selection",
        "burgers_balance_histories",
    ),
    "cavity": (
        "cavity_weight_selection",
        "cavity_balance_histories",
    ),
}
FIGURE_NAMES = tuple(name for group in FIGURE_GROUPS.values() for name in group)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid Boolean value {value!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(plt, figure, output_dir: Path, stem: str, title: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    metadata = {
        "Title": title,
        "Author": "PI-BSR manuscript authors",
        "Creator": Path(__file__).name,
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(pdf, metadata=metadata, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(
        png, dpi=300, metadata={"Software": Path(__file__).name},
        bbox_inches="tight", pad_inches=0.03,
    )
    plt.close(figure)
    return [pdf, png]


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.titlesize": 8.7,
            "axes.labelsize": 8.4,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    return plt


def initial_condition(np, x):
    return 0.45 * (1.0 + np.tanh((0.65 - x) / 0.03)) * 4.0 * x * (1.0 - x)


def analytical_one_d(np, x, times):
    modes = np.arange(1, 901, dtype=np.float64)
    quadrature_x = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    alpha = 10.0
    integrand = (
        np.exp(-alpha * quadrature_x)[:, None]
        * initial_condition(np, quadrature_x)[:, None]
        * np.sin(np.pi * np.outer(quadrature_x, modes))
    )
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    coefficients = 2.0 * integrate(integrand, quadrature_x, axis=0)
    decay = 0.05 * (np.pi * modes) ** 2 + 5.0
    temporal = np.exp(-np.outer(times, decay)) * coefficients[None, :]
    spatial = np.exp(alpha * x)[:, None] * np.sin(np.pi * np.outer(x, modes))
    return temporal @ spatial.T


def one_d_observations(plt, np, output_dir: Path) -> list[Path]:
    data_dir = REPRO_ROOT / "data" / "one_d_convection_diffusion"
    frames = []
    x = None
    for index in range(400):
        values = np.loadtxt(data_dir / f"rho_T3_{index:04d}.txt")
        if values.shape != (60, 2):
            raise ValueError(f"unexpected 1-D frame shape at index {index}: {values.shape}")
        if x is None:
            x = values[:, 0]
        elif not np.array_equal(x, values[:, 0]):
            raise ValueError(f"1-D spatial grid changed at frame {index}")
        frames.append(values[:, 1])
    assert x is not None
    numerical = np.asarray(frames)
    times = np.linspace(0.0, 0.4, 400)
    analytical = analytical_one_d(np, x, times)

    figure = plt.figure(figsize=(8.05, 3.45))
    grid = figure.add_gridspec(
        3, 2, width_ratios=(1.55, 1.0), left=0.07, right=0.985,
        bottom=0.13, top=0.91, wspace=0.22, hspace=0.18,
    )
    profile = figure.add_subplot(grid[:, 0])
    profile.plot(x, analytical[-1], color="black", linewidth=1.5, label=r"Analytical ($t=0.4$)")
    profile.plot(x, numerical[-1], color="#E66101", linewidth=1.3, label=r"Numerical ($t=0.4$)")
    profile.plot(x, initial_condition(np, x), color="#0072B2", linestyle="--", linewidth=1.2, label=r"Initial ($t=0$)")
    profile.set_xlabel(r"$x$")
    profile.set_ylabel(r"$\rho$")
    profile.set_ylim(-0.05, 1.0)
    profile.grid(color="#D9D9D9", linewidth=0.55)
    profile.legend(loc="upper right", frameon=True)
    profile.text(-0.12, 1.05, "(a)", transform=profile.transAxes, fontstyle="italic")

    fields = (
        (analytical, "Analytical", "viridis", (0.0, 0.9)),
        (numerical, "Numerical", "viridis", (0.0, 0.9)),
        (numerical - analytical, "Signed error", "RdBu_r", (-0.05, 0.05)),
    )
    panels = ("(b)", "(c)", "(d)")
    for row, ((field, title, cmap, limits), panel) in enumerate(zip(fields, panels)):
        axis = figure.add_subplot(grid[row, 1])
        levels = np.linspace(limits[0], limits[1], 61)
        image = axis.contourf(x, times, field, levels=levels, cmap=cmap, extend="both")
        axis.contour(
            x, times, field, levels=levels[::4], colors="white",
            linewidths=0.2, linestyles=":", alpha=0.28,
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 0.4)
        axis.set_ylabel(r"$t$", rotation=0, labelpad=10)
        axis.set_title(title, pad=1.5)
        axis.text(-0.18, 1.05, panel, transform=axis.transAxes, fontstyle="italic")
        if row < 2:
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel(r"$x$")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
        colorbar.ax.tick_params(labelsize=6.5, pad=1.0)
    return save_figure(
        plt, figure, output_dir, "one_d_observations",
        "One-dimensional observations and analytical reference",
    )


def one_d_weight_selection(plt, np, output_dir: Path) -> list[Path]:
    root = REPRO_ROOT / "figure_data" / "one_d"
    candidates = read_tsv(root / "strong_weight_candidates.tsv")
    pareto = read_tsv(root / "strong_weight_selection.tsv")
    if len(candidates) != 26 or len(pareto) != 13:
        raise ValueError("unexpected 1-D selection-table size")
    selected_candidates = [row for row in candidates if float(row["beta"]) == 3.0e-5]
    selected_pareto = [row for row in pareto if parse_bool(row["selected"])]
    if len(selected_candidates) != 1 or len(selected_pareto) != 1:
        raise ValueError("the selected 1-D beta is not unique")

    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.05))
    left, right = axes
    positive = [row for row in candidates if float(row["beta"]) > 0.0]
    beta = np.asarray([float(row["beta"]) for row in positive])
    data_ratio = np.asarray([float(row["data_ratio"]) for row in positive])
    pde_ratio = np.asarray([float(row["pde_ratio"]) for row in positive])
    left.loglog(beta, data_ratio, color="#4C78A8", marker="s", markersize=3.4,
                linewidth=1.2, label=r"$D_\beta/D_0$")
    left.loglog(beta, pde_ratio, color="#E66101", marker="o", markersize=3.4,
                linewidth=1.2, label=r"$R_\beta/R_0$")
    chosen = selected_candidates[0]
    left.scatter(float(chosen["beta"]), float(chosen["pde_ratio"]), s=38,
                 color="#19D325", edgecolor="white", linewidth=0.6,
                 zorder=5, label="selected")
    left.annotate(r"$\beta=3\times10^{-5}$",
                  (float(chosen["beta"]), float(chosen["pde_ratio"])),
                  xytext=(18, 7), textcoords="offset points", fontsize=7.2)
    left.set_xlabel(r"strong-form weight $\beta$")
    left.set_ylabel("Data-only-normalized MSE")
    left.grid(True, which="both", color="#D9D9D9", linewidth=0.5)
    left.legend(frameon=True, loc="upper left")
    left.set_title("(a)", loc="left", fontstyle="italic")

    px = np.asarray([float(row["normalized_log_data"]) for row in pareto])
    py = np.asarray([float(row["normalized_log_pde"]) for row in pareto])
    right.plot(px, py, color="#4D4D4D", marker="o", markersize=3.2,
               markerfacecolor="white", linewidth=1.5, label="Pareto front")
    right.plot((0.0, 1.0), (1.0, 0.0), color="#A7A7A7",
               linestyle=(0, (4, 3)), linewidth=1.1, label="endpoint chord")
    point = selected_pareto[0]
    selected_x = float(point["normalized_log_data"])
    selected_y = float(point["normalized_log_pde"])
    right.scatter(selected_x, selected_y, s=42, color="#19D325",
                  edgecolor="white", linewidth=0.6, zorder=5, label="selected knee")
    right.annotate("", xy=(selected_x, selected_y), xytext=(selected_x - 0.09, selected_y - 0.13),
                   arrowprops={"arrowstyle": "-", "color": "red", "linewidth": 1.0})
    right.text(0.10, 0.92, "data-dominated", color="#666666", fontsize=7.0)
    right.text(0.80, 0.24, "physics-\ndominated", color="#666666", fontsize=7.0)
    right.set_xlabel(r"$\hat d_\beta$")
    right.set_ylabel(r"$\hat s_\beta$")
    right.set_xlim(-0.05, 1.05)
    right.set_ylim(-0.05, 1.05)
    right.grid(color="#D9D9D9", linewidth=0.5)
    right.legend(frameon=True, loc="lower left")
    right.set_title("(b)", loc="left", fontstyle="italic")
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.91, wspace=0.35)
    return save_figure(
        plt, figure, output_dir, "one_d_weight_selection",
        "One-dimensional strong-form weight selection",
    )


def grouped_curves(rows: list[dict[str, str]], series_key: str, index_key: str):
    result: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        result.setdefault(row[series_key], []).append(row)
    for values in result.values():
        values.sort(key=lambda row: int(row[index_key]))
    return result


def one_d_balance_histories(plt, np, output_dir: Path) -> list[Path]:
    rows = read_tsv(REPRO_ROOT / "figure_data" / "one_d" / "balance_histories.tsv")
    curves = grouped_curves(rows, "series_id", "time_index")
    q1_series = (
        "analytical_exact", "released_numerical_data", "data_only_mfa",
        "declared_subset_pinn", "strong_only_gamma0", "selected_gamma_0.1",
        "gamma10_reporting_only",
    )
    if set(curves) != set(q1_series) or any(len(values) != 400 for values in curves.values()):
        raise ValueError("unexpected 1-D balance-curve contract")

    styles = {
        "analytical_exact": ("Analytical", "black", (0, (5, 3, 1, 3)), 1.2),
        "released_numerical_data": ("Numerical data (FDM)", "#999999", "-", 0.9),
        "data_only_mfa": (r"Data-only MFA ($\beta=0$)", "#CC79A7", (0, (6, 2, 1, 2)), 1.25),
        "declared_subset_pinn": ("PINN", "#009E73", (0, (1, 1.5)), 1.4),
        "strong_only_gamma0": (r"PI-BSR (strong-form, $\gamma=0$)", "#0072B2", (0, (5, 4)), 1.55),
        "selected_gamma_0.1": (r"PI-BSR (hybrid, $\gamma=0.1$)", "#D55E00", "-", 2.1),
        "gamma10_reporting_only": (r"PI-BSR (hybrid, $\gamma=10$)", "#7570B3", (0, (4, 2)), 1.55),
    }
    q4_series = (
        "data_only_mfa", "strong_only_gamma0", "selected_gamma_0.1",
        "gamma10_reporting_only",
    )
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 5.15), sharex=True)
    top, bottom = axes
    handle_map = {}
    for name in q4_series:
        entries = curves[name]
        label, color, linestyle, width = styles[name]
        line = top.plot(
            [float(row["t"]) for row in entries],
            [float(row["q4_gl8_regional_rms"]) for row in entries],
            label=label, color=color, linestyle=linestyle, linewidth=width,
            marker="D" if name == "gamma10_reporting_only" else None,
            markevery=(20, 40), markersize=3.5, markerfacecolor="white",
        )[0]
        handle_map[name] = line
    top.set_yscale("log")
    q4_max = max(float(row["q4_gl8_regional_rms"]) for name in q4_series for row in curves[name])
    q4_min = min(float(row["q4_gl8_regional_rms"]) for name in q4_series for row in curves[name])
    if q4_min <= 0:
        raise ValueError("Regional RMS must be positive for logarithmic plotting")
    top.set_ylim(min(3.0e-4, q4_min / 1.1), max(1.5e-1, q4_max * 1.1))
    top.set_ylabel(r"regional RMS $R_{\rho}^{(4)}(t)$")
    top.set_title(r"(a) Regional $q=4$, GL8 balance residual")

    for name in q1_series:
        entries = curves[name]
        label, color, linestyle, width = styles[name]
        line = bottom.plot(
            [float(row["t"]) for row in entries],
            [float(row["q1_gl8_signed"]) for row in entries],
            label=label, color=color, linestyle=linestyle, linewidth=width,
            marker="D" if name == "gamma10_reporting_only" else None,
            markevery=(20, 40), markersize=3.5, markerfacecolor="white",
        )[0]
        handle_map[name] = line
    bottom.set_ylim(-0.28, 0.14)
    bottom.set_ylabel(r"$b_{\rho,\Omega}(t)$")
    bottom.set_xlabel(r"$t$")
    bottom.set_title(r"(b) Whole-domain $q=1$, GL8 balance residual")
    for axis in axes:
        axis.set_xlim(0.0, 0.4)
        axis.set_xticks(np.arange(0.0, 0.401, 0.05))
        axis.grid(color="#D9D9D9", linewidth=0.55)
    legend_order = q1_series
    figure.legend(
        [handle_map[name] for name in legend_order],
        [styles[name][0] for name in legend_order],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4,
        frameon=False, handlelength=2.7, columnspacing=0.9,
    )
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.835, hspace=0.34)
    return save_figure(
        plt, figure, output_dir, "one_d_balance_histories",
        "One-dimensional regional and whole-domain balance histories",
    )


def load_burgers_frame(np, directory: Path, index: int):
    values = np.loadtxt(directory / f"uv_times_{index:04d}.txt")
    if values.shape != (1681, 4):
        raise ValueError(f"unexpected Burgers frame shape: {values.shape}")
    x = np.unique(values[:, 0])
    y = np.unique(values[:, 1])
    if x.size != 41 or y.size != 41:
        raise ValueError("unexpected Burgers grid")
    return x, y, values[:, 2].reshape(41, 41), values[:, 3].reshape(41, 41)


def burgers_discrepancy(plt, np, output_dir: Path) -> list[Path]:
    base = REPRO_ROOT / "data" / "two_d_burgers"
    indices = (0, 50, 100, 150, 200)
    times = (0.0, 0.5, 1.0, 1.5, 2.0)
    figure = plt.figure(figsize=(9.6, 6.8))
    grid = figure.add_gridspec(
        5, 6, left=0.06, right=0.985, bottom=0.07, top=0.92,
        wspace=0.42, hspace=0.08,
    )
    error_limits = (1.0e-5, 0.07, 0.26, 0.30, 0.30)
    axes = []
    for row, (index, time) in enumerate(zip(indices, times)):
        x_ref, y_ref, u_ref, v_ref = load_burgers_frame(np, base / "reference", index)
        x_obs, y_obs, u_obs, v_obs = load_burgers_frame(np, base / "observed", index)
        if not np.array_equal(x_ref, x_obs) or not np.array_equal(y_ref, y_obs):
            raise ValueError(f"Burgers grids differ at time {time}")
        values = (
            (u_ref, "turbo", 0.0, 1.0),
            (u_obs, "turbo", 0.0, 1.0),
            (np.abs(u_obs - u_ref), "RdBu_r", 0.0, error_limits[row]),
            (v_ref, "turbo", 0.0, 0.8),
            (v_obs, "turbo", 0.0, 0.8),
            (np.abs(v_obs - v_ref), "RdBu_r", 0.0, error_limits[row]),
        )
        row_axes = []
        for column, (field, cmap, low, high) in enumerate(values):
            axis = figure.add_subplot(grid[row, column])
            image = axis.imshow(
                field, origin="lower", extent=(0.0, 2.0, 0.0, 2.0),
                cmap=cmap, vmin=low, vmax=high, interpolation="bilinear",
                rasterized=True, aspect="equal",
            )
            colorbar = figure.colorbar(image, ax=axis, fraction=0.038, pad=0.012)
            colorbar.ax.tick_params(labelsize=5.6, length=2, pad=1)
            colorbar.set_ticks((low, high))
            axis.set_xticks((0.0, 1.0, 2.0))
            axis.set_yticks((0.0, 1.0, 2.0))
            if row < 4:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(r"$x$")
            if column not in (0, 3):
                axis.tick_params(labelleft=False)
            else:
                axis.set_ylabel(r"$y$", rotation=0, labelpad=8)
            if column == 0:
                axis.text(-0.38, 0.5, rf"$t={time:.1f}$", rotation=90,
                          va="center", transform=axis.transAxes, fontsize=8.5)
            row_axes.append(axis)
        axes.append(row_axes)
    titles = ("Reference\n(FD/RK4)", "Forced\nobservations", "Absolute\nerror")
    for offset, field in ((0, r"$u$"), (3, r"$v$")):
        for column, title in enumerate(titles):
            axes[0][offset + column].set_title(title, pad=5)
        axes[0][offset + 1].text(0.5, 1.31, field, transform=axes[0][offset + 1].transAxes,
                                ha="center", fontweight="bold", fontsize=10)
    return save_figure(
        plt, figure, output_dir, "burgers_discrepancy",
        "Forced Burgers observations and unforced reference",
    )


def burgers_weight_selection(plt, _np, output_dir: Path) -> list[Path]:
    root = REPRO_ROOT / "figure_data" / "two_d_burgers"
    beta_rows = read_tsv(root / "beta_selection.tsv")
    kappa_rows = read_tsv(root / "kappa_selection.tsv")
    gamma_rows = read_tsv(root / "gamma_selection.tsv")
    if (len(beta_rows), len(kappa_rows), len(gamma_rows)) != (10, 5, 4):
        raise ValueError("unexpected Burgers selection-table size")

    front = [row for row in beta_rows if parse_bool(row["pareto"])]
    x_min, x_max = min(math.log(float(row["data_ratio"])) for row in front), max(
        math.log(float(row["data_ratio"])) for row in front
    )
    y_min, y_max = min(math.log(float(row["physics_ratio"])) for row in front), max(
        math.log(float(row["physics_ratio"])) for row in front
    )

    def coordinates(row):
        x = (math.log(float(row["data_ratio"])) - x_min) / (x_max - x_min)
        y = (math.log(float(row["physics_ratio"])) - y_min) / (y_max - y_min)
        return x, y

    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.72))
    first, second, third = axes
    front_xy = [coordinates(row) for row in front]
    first.plot([p[0] for p in front_xy], [p[1] for p in front_xy],
               color="#0072B2", marker="o", markerfacecolor="white",
               markersize=3.6, linewidth=1.4, label="Pareto front")
    first.plot((0.0, 1.0), (1.0, 0.0), color="#999999",
               linestyle=(0, (4, 3)), linewidth=1.0, label="endpoint chord")
    rejected = [row for row in beta_rows if float(row["beta"]) > 0 and not parse_bool(row["eligible"])]
    first.scatter(*zip(*(coordinates(row) for row in rejected)), marker="x", s=26,
                  color="#999999", linewidth=1.0, label="not converged")
    baseline = next(row for row in beta_rows if float(row["beta"]) == 0.0)
    first.scatter(*coordinates(baseline), marker="D", s=34, facecolor="white",
                  edgecolor="#666666", linewidth=1.0, label="data-only baseline")
    chosen = next(row for row in beta_rows if parse_bool(row["selected"]))
    chosen_xy = coordinates(chosen)
    first.scatter(*chosen_xy, s=38, color="#19D325", edgecolor="white", zorder=6)
    first.annotate(r"selected $\beta=0.01$", chosen_xy, xytext=(16, 9),
                   textcoords="offset points", fontsize=6.5)
    first.set_xlabel(r"log-normalized data coordinate $\hat d_\beta$")
    first.set_ylabel(r"log-normalized strong-form coordinate $\hat s_\beta$")
    first.set_xlim(-0.08, 1.72)
    first.set_ylim(-1.05, 1.10)
    first.set_title(r"(a) Strong-form weight $\beta$")
    first.legend(frameon=False, loc="lower left", fontsize=5.6)

    strong = [row for row in kappa_rows if row["literal"] != "10_hybrid_gamma_0.03"]
    second.loglog(
        [float(row["kappa"]) for row in strong],
        [float(row["trace_rms"]) for row in strong],
        color="#0072B2", marker="o", markersize=4.0, linewidth=1.4,
        label=r"strong form, $\gamma=0$",
    )
    hybrid = next(row for row in kappa_rows if row["literal"] == "10_hybrid_gamma_0.03")
    second.scatter(float(hybrid["kappa"]), float(hybrid["trace_rms"]), marker="D", s=45,
                   facecolor="white", edgecolor="#E66101", linewidth=1.2,
                   label=r"hybrid, $\gamma=0.03$")
    chosen_kappa = next(row for row in strong if parse_bool(row["selected"]))
    second.scatter(float(chosen_kappa["kappa"]), float(chosen_kappa["trace_rms"]),
                   s=38, color="#19D325", edgecolor="white", zorder=6)
    second.axhline(1.0e-3, color="#C51B32", linestyle="--", linewidth=1.0)
    second.text(8.0, 1.12e-3, r"tolerance $10^{-3}$", color="#C51B32", fontsize=6.1)
    second.set_xticks((1, 3, 10, 30), ("1", "3", "10", "30"))
    second.set_xlabel(r"trace multiplier $\kappa$")
    second.set_ylabel("maximum normalized IC/BC trace RMS")
    second.set_title(r"(b) Trace multiplier $\kappa$")
    second.legend(frameon=False, loc="lower left", fontsize=5.7)

    gamma = [float(row["gamma"]) for row in gamma_rows]
    design_a = [float(row["design_a_ratio"]) for row in gamma_rows]
    third.plot(gamma, design_a, color="#0072B2", marker="o", markersize=4.0,
               linewidth=1.4, label="selection times")
    selected_gamma = next(row for row in gamma_rows if parse_bool(row["selected"]))
    gx = float(selected_gamma["gamma"])
    gy = float(selected_gamma["design_a_ratio"])
    third.scatter(gx, gy, s=40, color="#19D325", edgecolor="white", zorder=6)
    third.scatter(gx, float(selected_gamma["design_b_ratio"]), marker="s", s=70,
                  facecolor="none", edgecolor="#E66101", linewidth=1.2,
                  label="separate evaluation times")
    third.axhline(0.99, color="#C51B32", linestyle="--", linewidth=0.9)
    third.axhspan(0.92, 0.99, color="#009E73", alpha=0.08)
    third.text(0.02, 0.992, "1% reduction threshold", color="#C51B32", fontsize=5.9)
    third.annotate(r"selected $\gamma=0.03$", (gx, gy), xytext=(-29, -22),
                   textcoords="offset points", fontsize=6.2)
    third.set_xlabel(r"balance weight $\gamma$")
    third.set_ylabel(r"geometric $q=4$ balance ratio $G(\gamma)$")
    third.set_ylim(0.92, 1.005)
    third.set_xticks(gamma, ("0", "0.01", "0.03", "0.10"))
    third.set_title(r"(c) Balance weight $\gamma$")
    third.legend(frameon=False, loc="lower left", fontsize=5.7)
    for axis in axes:
        axis.grid(color="#D9D9D9", linewidth=0.5)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.21, top=0.88, wspace=0.40)
    return save_figure(
        plt, figure, output_dir, "burgers_weight_selection",
        "Burgers three-stage weight selection",
    )


def burgers_balance_histories(plt, np, output_dir: Path) -> list[Path]:
    rows = read_tsv(REPRO_ROOT / "figure_data" / "two_d_burgers" / "balance_histories.tsv")
    curves = grouped_curves(rows, "series_id", "time_index")
    all_series = (
        "numerical_observations", "data_only_mfa", "pinn_mean", "matched_gamma0",
        "selected_gamma0p03", "higher_gamma0p10", "stress_gamma1",
    )
    q4_series = (
        "data_only_mfa", "matched_gamma0", "selected_gamma0p03",
        "higher_gamma0p10", "stress_gamma1",
    )
    if set(curves) != set(all_series) or len(rows) != 1402:
        raise ValueError("unexpected Burgers balance-curve contract")
    styles = {
        "numerical_observations": ("Numerical data (FDM; contextual)", "#999999", "-", None),
        "data_only_mfa": (r"Data-only MFA ($\beta=0$)", "#CC79A7", (0, (6, 2, 1, 2)), None),
        "pinn_mean": ("PINN", "#009E73", (0, (1, 1.5)), None),
        "matched_gamma0": (r"PI-BSR (strong-form, $\gamma=0$; matched)", "#0072B2", (0, (5, 4)), None),
        "selected_gamma0p03": (r"PI-BSR (hybrid, $\gamma=0.03$; selected)", "#D55E00", "-", None),
        "higher_gamma0p10": (r"PI-BSR (hybrid, $\gamma=0.10$)", "#E69F00", (0, (4, 2)), None),
        "stress_gamma1": (r"PI-BSR (hybrid, $\gamma=1$)", "#7570B3", (0, (4, 2)), "D"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), sharex=True)
    handles = {}
    for column, component in enumerate(("u", "v")):
        top = axes[0, column]
        for series in q4_series:
            values = curves[series]
            label, color, linestyle, marker = styles[series]
            line = top.plot(
                [float(row["time"]) for row in values],
                [float(row[f"q4_scaled_rms_{component}"]) for row in values],
                label=label, color=color, linestyle=linestyle,
                linewidth=2.0 if series == "selected_gamma0p03" else 1.25,
                marker=marker, markevery=(10, 20), markersize=3.3,
                markerfacecolor="white",
            )[0]
            handles[series] = line
        top.set_yscale("log")
        top.set_ylim(2.0e-4, 1.2e-1)
        top.set_ylabel(rf"$R_{{{component}}}^{{(4)}}(t)$")
        top.set_title(rf"({chr(97 + column)}) Regional GL8 $q=4$, ${component}$ momentum")

        bottom = axes[1, column]
        bottom.axhline(0.0, color="black", linestyle=(0, (5, 3, 1, 3)),
                       linewidth=1.1, label=r"Zero balance ($q=1$ only)")
        pinn = curves["pinn_mean"]
        times = np.asarray([float(row["time"]) for row in pinn])
        center = np.asarray([float(row[f"signed_q1_{component}"]) for row in pinn])
        spread = np.asarray([float(row[f"sample_sd_{component}"]) for row in pinn])
        bottom.fill_between(times, center - spread, center + spread,
                            color="#009E73", alpha=0.12, linewidth=0.0)
        for series in all_series:
            values = curves[series]
            label, color, linestyle, marker = styles[series]
            line = bottom.plot(
                [float(row["time"]) for row in values],
                [float(row[f"signed_q1_{component}"]) for row in values],
                label=label, color=color, linestyle=linestyle,
                linewidth=2.0 if series == "selected_gamma0p03" else 1.15,
                marker=marker, markevery=(10, 20), markersize=3.3,
                markerfacecolor="white",
            )[0]
            handles[series] = line
        bottom.set_ylim((-0.72, 0.72) if component == "u" else (-0.62, 0.62))
        bottom.set_ylabel(rf"$b_{{{component},\Omega}}(t)$")
        bottom.set_xlabel(r"$t$")
        bottom.set_title(rf"({chr(99 + column)}) Whole-domain $q=1$, ${component}$ momentum")
    for axis in axes.ravel():
        axis.set_xlim(0.0, 2.0)
        axis.set_xticks((0.0, 0.5, 1.0, 1.5, 2.0))
        axis.grid(color="#D9D9D9", linewidth=0.5)
    zero_handle = axes[1, 0].get_lines()[0]
    legend_order = all_series + ("zero",)
    handles["zero"] = zero_handle
    legend_labels = [styles[name][0] if name != "zero" else r"Zero balance ($q=1$ only)" for name in legend_order]
    figure.legend(
        [handles[name] for name in legend_order], legend_labels,
        frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        fontsize=5.75, columnspacing=0.62, handlelength=2.15, handletextpad=0.38,
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.105, top=0.82, wspace=0.27, hspace=0.31)
    return save_figure(
        plt, figure, output_dir, "burgers_balance_histories",
        "Burgers regional and whole-domain balance histories",
    )


def cavity_weight_selection(plt, _np, output_dir: Path) -> list[Path]:
    path = (
        REPRO_ROOT / "results" / "lid_driven_cavity" / "postselection"
        / "ramped_cavity_weight_selection_figure_v2"
        / "CAVITY_FAMILY_SELECTION_COORDINATES.tsv"
    )
    rows = read_tsv(path)
    if len(rows) != 16 or sum(parse_bool(row["admissible"]) for row in rows) != 9:
        raise ValueError("unexpected cavity selection-table contract")
    selected_ids = {row["candidate_id"] for row in rows if parse_bool(row["within_family_selected"])}
    if selected_ids != {
        "strong_only__gamma_0p03",
        "gl4_conservative_only__gamma_0p02",
        "gl4_hybrid__gamma_0p04",
    }:
        raise ValueError("cavity selected endpoints changed")
    order = ("strong_only", "trapezoid_hybrid", "gl4_conservative_only", "gl4_hybrid")
    styles = {
        "strong_only": ("Strong form only", "#0072B2", "o", "-"),
        "trapezoid_hybrid": ("Trapezoidal hybrid", "#777777", "^", (0, (3, 2))),
        "gl4_conservative_only": ("GL4 integral-balance only", "#CC79A7", "s", (0, (5, 2, 1, 2))),
        "gl4_hybrid": ("GL4 hybrid", "#D55E00", "D", "-"),
    }
    panels = (
        ("worst_aligned_validation_ratio", r"worst-case ratio $R_{\max}$", (0.925, 1.004)),
        ("aligned_geometric_mean_validation_ratio", r"geometric-mean overall ratio $R_{\rm geo}$", (0.10, 0.40)),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.18))
    for axis, (field, ylabel, limits) in zip(axes, panels):
        for family in order:
            label, color, marker, linestyle = styles[family]
            family_rows = sorted((row for row in rows if row["family"] == family),
                                 key=lambda row: float(row["gamma"]))
            admissible = [row for row in family_rows if parse_bool(row["admissible"])]
            if admissible:
                axis.plot(
                    [float(row["gamma"]) for row in admissible],
                    [float(row[field]) for row in admissible],
                    label=label, color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.4, markersize=4.0, markerfacecolor="white",
                    markeredgewidth=1.0,
                )
            rejected = [row for row in family_rows if not parse_bool(row["admissible"])]
            if rejected:
                axis.scatter([float(row["gamma"]) for row in rejected],
                             [float(row[field]) for row in rejected], marker="x",
                             s=31, color=color, linewidth=1.2, zorder=5)
            chosen = [row for row in family_rows if parse_bool(row["within_family_selected"])]
            if chosen:
                axis.scatter(float(chosen[0]["gamma"]), float(chosen[0][field]),
                             marker="*", s=105, facecolor=color, edgecolor="black",
                             linewidth=0.65, zorder=8)
        axis.set_xlim(0.0075, 0.0425)
        axis.set_ylim(*limits)
        axis.set_xticks((0.01, 0.02, 0.03, 0.04))
        axis.set_xlabel(r"physics budget $\gamma_{\rm tot}=\gamma_S+\gamma_I$")
        axis.set_ylabel(ylabel)
        axis.grid(color="#D9D9D9", linewidth=0.5)
    axes[0].axhline(1.0, color="#B2182B", linestyle="--", linewidth=0.9)
    axes[0].text(0.0418, 1.0005, r"no change from Stage A ($=1$)",
                 color="#9B1C31", ha="right", va="bottom", fontsize=5.8)
    axes[0].set_title("(a)", loc="left", fontstyle="italic")
    axes[1].set_title("(b)", loc="left", fontstyle="italic")
    from matplotlib.lines import Line2D
    family_proxies = [
        Line2D(
            [], [], color=styles[name][1], marker=styles[name][2],
            linestyle=styles[name][3], markersize=4.0,
            markerfacecolor="white", label=styles[name][0],
        )
        for name in order
    ]
    selected_proxy = Line2D([], [], marker="*", linestyle="none", markersize=8,
                            markerfacecolor="white", markeredgecolor="black",
                            label="selected within family")
    rejected_proxy = Line2D([], [], marker="x", linestyle="none", markersize=5,
                            color="#555555", label="inadmissible")
    figure.legend(
        family_proxies + [selected_proxy, rejected_proxy],
        [styles[name][0] for name in order] + ["selected within family", "inadmissible"],
        frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        fontsize=6.1, columnspacing=0.75, handlelength=2.0, handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.73, wspace=0.30)
    return save_figure(
        plt, figure, output_dir, "cavity_weight_selection",
        "Cavity within-family weight selection",
    )


def cavity_balance_histories(plt, np, output_dir: Path) -> list[Path]:
    root = (
        REPRO_ROOT / "results" / "lid_driven_cavity" / "postselection"
        / "ramped_cavity_whole_domain_q1_figure_v13"
    )
    rows = read_tsv(root / "CAVITY_DETERMINISTIC_Q4_SIGNED_Q1_CURVES.tsv")
    aggregate = read_tsv(root / "PINN_AGGREGATE_HISTORIES.tsv")
    curves = grouped_curves(rows, "role", "time_index")
    if len(rows) != 800 or len(aggregate) != 200 or any(len(values) != 200 for values in curves.values()):
        raise ValueError("unexpected cavity balance-curve contract")
    visible = ("stage_a", "matched_strong_gamma_i0", "gl4_hybrid")
    styles = {
        "stage_a": ("Data/trace-only MFA (Stage A)", "#CC79A7", (0, (6, 2, 1, 2)), 1.2),
        "matched_strong_gamma_i0": (r"PI-BSR (matched strong form, $\gamma_I=0$)", "#0072B2", (0, (5, 4)), 1.5),
        "gl4_hybrid": (r"PI-BSR (GL4 hybrid, $\gamma_{\rm tot}=0.04$)", "#D55E00", "-", 2.1),
        "pinn": ("PINN (5-seed median [IQR]; contextual)", "#009E73", (0, (1, 1.5)), 1.35),
    }
    components = ("mass", "x_momentum", "y_momentum")
    labels = {"mass": "mass", "x_momentum": r"$x$ momentum", "y_momentum": r"$y$ momentum"}
    q4_limits = {"mass": (2.0e-4, 1.5e-2), "x_momentum": (8.0e-4, 3.0e-2), "y_momentum": (1.0e-4, 3.0e-2)}
    q1_limits = {"mass": (-0.015, 0.015), "x_momentum": (-0.05, 0.05), "y_momentum": (-0.015, 0.015)}
    aggregate.sort(key=lambda row: int(row["time_index"]))
    times = np.asarray([float(row["time"]) for row in aggregate])
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.35), sharex="col")
    handles = {}
    for column, component in enumerate(components):
        q4_field = f"q4_spatial_rms_{component}"
        top = axes[0, column]
        low = np.asarray([float(row[f"{q4_field}_q25"]) for row in aggregate])
        center = np.asarray([float(row[f"{q4_field}_median"]) for row in aggregate])
        high = np.asarray([float(row[f"{q4_field}_q75"]) for row in aggregate])
        top.fill_between(times, low, high, color="#009E73", alpha=0.12, linewidth=0.0)
        for role in visible:
            label, color, linestyle, width = styles[role]
            values = curves[role]
            line = top.plot([float(row["time"]) for row in values],
                            [float(row[q4_field]) for row in values],
                            label=label, color=color, linestyle=linestyle,
                            linewidth=width)[0]
            handles[role] = line
        handles["pinn"] = top.plot(times, center, label=styles["pinn"][0],
                                   color=styles["pinn"][1], linestyle=styles["pinn"][2],
                                   linewidth=styles["pinn"][3])[0]
        top.set_yscale("log")
        top.set_ylim(*q4_limits[component])
        top.set_ylabel(rf"$R_{{{component[0] if component != 'mass' else 'm'}}}^{{(4)}}(t)$")
        top.set_title(rf"({chr(97 + column)}) $q=4$ RMS: {labels[component]}", fontsize=7.0)

        q1_field = f"signed_q1_{component}"
        bottom = axes[1, column]
        zero = bottom.axhline(0.0, color="black", linewidth=1.1,
                             linestyle=(0, (5, 3, 1, 3)), label=r"Zero balance ($q=1$ only)")
        handles["zero"] = zero
        low = np.asarray([float(row[f"{q1_field}_q25"]) for row in aggregate])
        center = np.asarray([float(row[f"{q1_field}_median"]) for row in aggregate])
        high = np.asarray([float(row[f"{q1_field}_q75"]) for row in aggregate])
        bottom.fill_between(times, low, high, color="#009E73", alpha=0.12, linewidth=0.0)
        for role in visible:
            label, color, linestyle, width = styles[role]
            values = curves[role]
            bottom.plot([float(row["time"]) for row in values],
                        [float(row[q1_field]) for row in values],
                        label=label, color=color, linestyle=linestyle,
                        linewidth=width)
        bottom.plot(times, center, label=styles["pinn"][0], color=styles["pinn"][1],
                    linestyle=styles["pinn"][2], linewidth=styles["pinn"][3])
        bottom.set_ylim(*q1_limits[component])
        subscript = component[0] if component != "mass" else "m"
        bottom.set_ylabel(rf"$b_{{{subscript},\Omega}}(t)$")
        bottom.set_xlabel(r"$t$")
        bottom.set_title(rf"({chr(100 + column)}) $q=1$: {labels[component]}", fontsize=7.0)
    for axis in axes.ravel():
        axis.axvline(0.5, color="#999999", linestyle=(0, (1, 2)), linewidth=0.75)
        axis.set_xlim(0.0, 10.0)
        axis.set_xticks((0, 2, 4, 6, 8, 10))
        axis.grid(color="#D9D9D9", linewidth=0.48)
    legend_order = ("zero", "stage_a", "pinn", "matched_strong_gamma_i0", "gl4_hybrid")
    legend_labels = [
        r"Zero balance ($q=1$ only)", styles["stage_a"][0], styles["pinn"][0],
        styles["matched_strong_gamma_i0"][0], styles["gl4_hybrid"][0],
    ]
    figure.legend(
        [handles[name] for name in legend_order], legend_labels,
        frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995),
        fontsize=5.7, columnspacing=0.75, handlelength=2.55, handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.078, right=0.995, bottom=0.105, top=0.80, wspace=0.42, hspace=0.34)
    return save_figure(
        plt, figure, output_dir, "cavity_balance_histories",
        "Cavity regional and whole-domain balance histories",
    )


def requested_figures(target: str) -> tuple[str, ...]:
    if target == "all":
        return FIGURE_NAMES
    if target in FIGURE_GROUPS:
        return FIGURE_GROUPS[target]
    return (target,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", nargs="?", default="all",
        choices=("all",) + tuple(FIGURE_GROUPS) + FIGURE_NAMES,
        help="case group or individual figure to create",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--list", action="store_true", help="list individual figure names and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print("\n".join(FIGURE_NAMES))
        return
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("Figure generation requires NumPy and Matplotlib.") from error

    output_dir = args.output_dir.resolve()
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1788220800")
    with tempfile.TemporaryDirectory(prefix="pibsr_matplotlib_") as config_dir:
        os.environ["MPLCONFIGDIR"] = config_dir
        os.environ["XDG_CACHE_HOME"] = config_dir
        try:
            plt = configure_matplotlib()
        except ImportError as error:
            raise SystemExit("Figure generation requires NumPy and Matplotlib.") from error
        generators = {
            "one_d_observations": one_d_observations,
            "one_d_weight_selection": one_d_weight_selection,
            "one_d_balance_histories": one_d_balance_histories,
            "burgers_discrepancy": burgers_discrepancy,
            "burgers_weight_selection": burgers_weight_selection,
            "burgers_balance_histories": burgers_balance_histories,
            "cavity_weight_selection": cavity_weight_selection,
            "cavity_balance_histories": cavity_balance_histories,
        }
        generated = []
        for name in requested_figures(args.target):
            print(f"Creating {name} ...")
            generated.extend(generators[name](plt, np, output_dir))

    manifest = {
        "schema": "pibsr.reproducibility_figures.v1",
        "target": args.target,
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in generated
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Created {len(generated) // 2} figure(s) in {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
