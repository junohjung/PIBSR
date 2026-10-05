#!/usr/bin/env python3
"""Compose Figure 11 after rendering both selected volume panels."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = ROOT / "figure_data/selected_volumes"
    expected = json.loads((data / "manifest.json").read_text())["one_d.npz"]
    if hashlib.sha256((data / "one_d.npz").read_bytes()).hexdigest() != expected:
        raise ValueError("1D field checksum mismatch")
    with np.load(data / "one_d.npz", allow_pickle=False) as archive:
        x, t, rho = archive["x"], archive["t"], archive["rho"]
    rho_min, rho_max = float(rho.min()), float(rho.max())
    paths = {case + "_png": args.output_dir / (case + "_volume.png") for case in ("burgers", "cavity")}
    output_pdf = args.output_dir / "Figure11.pdf"
    output_png = args.output_dir / "Figure11.png"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(6.48, 2.55), dpi=500, facecolor="white")
    grid_spec = fig.add_gridspec(
        1,
        3,
        left=0.055,
        right=0.995,
        bottom=0.225,
        top=0.835,
        wspace=0.075,
        width_ratios=(0.93, 1.035, 1.035),
    )
    axes = [fig.add_subplot(grid_spec[0, i]) for i in range(3)]

    image_1d = axes[0].imshow(
        rho,
        origin="lower",
        extent=(float(x[0]), float(x[-1]), float(t[0]), float(t[-1])),
        aspect="auto",
        interpolation="bilinear",
        cmap="turbo",
        vmin=rho_min,
        vmax=rho_max,
        rasterized=True,
    )
    axes[0].set_xlabel("space $x$", fontsize=7.4, labelpad=1.0)
    axes[0].set_ylabel("time $t$", fontsize=7.4, labelpad=1.5)
    axes[0].set_xticks([0.0, 0.5, 1.0])
    axes[0].set_yticks([0.0, 0.2, 0.4])
    axes[0].tick_params(axis="both", labelsize=6.8, width=0.55, length=2.4, pad=1.2)
    for spine in axes[0].spines.values():
        spine.set_linewidth(0.55)
    axes[0].set_title(
        "1D convection--diffusion\nselected hybrid",
        fontsize=8.2,
        pad=2.0,
        linespacing=1.0,
    )
    axes[0].text(
        0.025,
        0.965,
        "(a)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color="white",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.62, "pad": 1.0},
    )

    panel_paths = [paths["burgers_png"], paths["cavity_png"]]
    titles = ["Burgers\nselected hybrid", "Smooth-ramped cavity\nselected GL4 hybrid"]
    time_labels = ["time $t$", "$0.1t$"]
    for panel_index, (axis, panel_path, title, time_label) in enumerate(
        zip(axes[1:], panel_paths, titles, time_labels), start=1
    ):
        panel = Image.open(panel_path).convert("RGB")
        if panel.size != (2800, 2200):
            raise ValueError(f"unexpected ParaView panel dimensions: {panel_path} {panel.size}")
        axis.imshow(panel)
        axis.set_axis_off()
        axis.set_title(title, fontsize=8.2, pad=2.0, linespacing=1.0)
        axis.text(
            0.018,
            0.965,
            f"({chr(97 + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            fontweight="bold",
        )
        axis.text(0.69, 0.80, "$x$", transform=axis.transAxes,
                  ha="center", va="center", fontsize=7.7, fontweight="bold")
        axis.text(0.31, 0.80, "$y$", transform=axis.transAxes,
                  ha="center", va="center", fontsize=7.7, fontweight="bold")
        axis.text(0.105, 0.50, time_label, transform=axis.transAxes,
                  ha="center", va="center", rotation=90, fontsize=7.7, fontweight="bold")

    colorbar_specs = [
        (image_1d, [rho_min, 0.30, 0.60, rho_max], r"field $\rho_h$"),
        (ScalarMappable(norm=Normalize(0.0, 1.0093387365341187), cmap="turbo"),
         [0.0, 0.50, 1.00], r"speed $\Vert\mathbf{u}_h\Vert_2$"),
        (ScalarMappable(norm=Normalize(0.0, 0.7514054179191589), cmap="turbo"),
         [0.0, 0.375, 0.75], r"speed $\Vert\mathbf{u}_h\Vert_2$"),
    ]
    fig.canvas.draw()
    for axis, (mappable, ticks, label) in zip(axes, colorbar_specs):
        if mappable is not image_1d:
            mappable.set_array([])
        box = axis.get_position()
        cax = fig.add_axes([box.x0 + 0.10 * box.width, 0.105, 0.80 * box.width, 0.027])
        colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
        colorbar.ax.tick_params(labelsize=6.2, width=0.5, length=2.1, pad=1.0)
        colorbar.outline.set_linewidth(0.5)
        colorbar.set_label(label, fontsize=6.8, labelpad=0.8)

    fig.savefig(output_png, dpi=500, facecolor="white")
    fig.savefig(output_pdf, dpi=500, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
