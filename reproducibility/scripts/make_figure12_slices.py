#!/usr/bin/env python3
"""Recreate the eight Figure 12 slices, excluding the volume-rendered panels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
BURGERS_TIMES = np.array([0.405, 0.805, 1.205, 1.605])
CAVITY_TIMES = np.array([0.5, 2.5, 5.005, 9.5])
BURGERS_RANGE = (0., 1.0093387365341187)
CAVITY_RANGE = (0., 0.7514054179191589)

def sha256_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def exact_slices(speed: np.ndarray, times: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    out = []
    records: list[dict] = []
    for target in targets:
        hits = np.flatnonzero(times == target)
        if hits.size != 1:
            raise RuntimeError(f"expected one exact stored plane for t={target}, found {hits.tolist()}")
        index = int(hits[0])
        plane = np.asarray(speed[index], dtype=np.float32)
        out.append(plane)
        records.append(
            {
                "physical_time": float(target),
                "method": "exact_stored_plane",
                "index": index,
                "stored_time": float(times[index]),
                "array_sha256": sha256_array(plane),
            }
        )
    return np.stack(out), records


def linear_time_slices(
    speed: np.ndarray,
    times: np.ndarray,
    targets: np.ndarray,
    *,
    display_time_scale: float,
) -> tuple[np.ndarray, list[dict]]:
    out = []
    records: list[dict] = []
    source = np.asarray(speed, dtype=np.float32)
    for target in targets:
        upper = int(np.searchsorted(times, target, side="right"))
        lower = upper - 1
        if lower < 0 or upper >= times.size:
            raise RuntimeError(f"target outside decoded time range: {target}")
        t0 = float(times[lower])
        t1 = float(times[upper])
        # Match the ParaView cut exactly: the cavity VTK stores
        # display z = 0.1 t, and interpolation weights are formed in that
        # displayed coordinate before the float32 scalar is interpolated.
        z_target = float(target) * display_time_scale
        z0 = t0 * display_time_scale
        z1 = t1 * display_time_scale
        weight = (z_target - z0) / (z1 - z0)
        if not (0.0 < weight < 1.0):
            raise RuntimeError(f"unexpected interpolation weight for t={target}: {weight}")
        # ParaView interpolates the stored float32 speed scalar itself.
        plane = np.asarray(
            source[lower].astype(np.float64)
            + weight * (source[upper].astype(np.float64) - source[lower].astype(np.float64)),
            dtype=np.float32,
        )
        out.append(plane)
        records.append(
            {
                "physical_time": float(target),
                "method": "linear_time_interpolation_of_float32_speed",
                "lower_index": lower,
                "upper_index": upper,
                "lower_time": t0,
                "upper_time": t1,
                "upper_weight": float(weight),
                "display_time_scale": float(display_time_scale),
                "display_coordinate": z_target,
                "array_sha256": sha256_array(plane),
            }
        )
    return np.stack(out), records


def stats(records: list[dict], planes: np.ndarray) -> list[dict]:
    enriched = []
    for record, plane in zip(records, planes, strict=True):
        enriched.append(
            {
                **record,
                "minimum": float(np.min(plane)),
                "maximum": float(np.max(plane)),
                "mean": float(np.mean(plane, dtype=np.float64)),
                "rms": float(np.sqrt(np.mean(np.square(plane.astype(np.float64))))),
            }
        )
    return enriched


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.2,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )


def draw_figure(
    burgers: np.ndarray,
    cavity: np.ndarray,
    output_pdf: Path,
    output_png: Path,
) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(10.7, 5.15), facecolor="white")
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.050],
        left=0.074,
        right=0.925,
        bottom=0.115,
        top=0.935,
        wspace=0.18,
        hspace=0.37,
    )
    axes = [[fig.add_subplot(grid[row, col]) for col in range(4)] for row in range(2)]
    caxes = [fig.add_subplot(grid[row, 4]) for row in range(2)]

    cases = [
        {
            "planes": burgers,
            "times": BURGERS_TIMES,
            "extent": (0.0, 2.0, 0.0, 2.0),
            "range": BURGERS_RANGE,
            "ticks": [0.0, 1.0, 2.0],
            "color_ticks": [0.0, 0.25, 0.50, 0.75, 1.00],
            "row_label": "Burgers selected hybrid",
        },
        {
            "planes": cavity,
            "times": CAVITY_TIMES,
            "extent": (0.0, 1.0, 0.0, 1.0),
            "range": CAVITY_RANGE,
            "ticks": [0.0, 0.5, 1.0],
            "color_ticks": [0.0, 0.20, 0.40, 0.60, 0.75],
            "row_label": "Smooth-ramped cavity selected GL4 hybrid",
        },
    ]
    letters = "abcdefgh"

    for row, case in enumerate(cases):
        norm = Normalize(*case["range"], clip=True)
        image = None
        for col, (axis, plane, time) in enumerate(
            zip(axes[row], case["planes"], case["times"], strict=True)
        ):
            image = axis.imshow(
                plane,
                origin="lower",
                extent=case["extent"],
                cmap="turbo",
                norm=norm,
                interpolation="bilinear",
                rasterized=True,
                aspect="equal",
            )
            axis.set_title(rf"({letters[row * 4 + col]})  $t={time:g}$", pad=4.0)
            axis.set_xlim(case["extent"][0], case["extent"][1])
            axis.set_ylim(case["extent"][2], case["extent"][3])
            axis.set_xticks(case["ticks"])
            axis.set_yticks(case["ticks"])
            axis.set_xlabel(r"$x$", labelpad=1.0)
            if col == 0:
                axis.set_ylabel(r"$y$", labelpad=1.5)
            else:
                axis.tick_params(labelleft=False)
            for spine in axis.spines.values():
                spine.set_color("0.20")
                spine.set_linewidth(0.65)
        assert image is not None
        colorbar = fig.colorbar(image, cax=caxes[row], ticks=case["color_ticks"])
        colorbar.set_label(r"speed $\|\mathbf{u}_h\|_2$", rotation=90, labelpad=4.5)
        colorbar.outline.set_linewidth(0.65)
        y_center = 0.712 if row == 0 else 0.300
        fig.text(
            0.018,
            y_center,
            case["row_label"],
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.4,
            fontweight="semibold",
        )

    fig.savefig(
        output_pdf,
        format="pdf",
        dpi=600,
        facecolor="white",
        metadata={
            "Title": "Burgers and cavity selected-reconstruction speed slices",
            "Author": "PI-BSR authors",
            "Subject": "Two-by-four decoded x-y speed slices",
            "Creator": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output_png, format="png", dpi=500, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "work/figures")
    args = parser.parse_args()
    data = ROOT / "figure_data/selected_volumes"
    manifest = json.loads((data / "manifest.json").read_text())
    for name in ("burgers.npz", "cavity.npz"):
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError("Checksum mismatch: " + name)
    with np.load(data / "burgers.npz", allow_pickle=False) as archive:
        burgers, br = exact_slices(archive["speed"], archive["t"], BURGERS_TIMES)
    with np.load(data / "cavity.npz", allow_pickle=False) as archive:
        cavity, cr = linear_time_slices(archive["speed"], archive["t"], CAVITY_TIMES, display_time_scale=0.1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_figure(burgers, cavity, args.output_dir / "Figure12_slices.pdf", args.output_dir / "Figure12_slices.png")
    (args.output_dir / "Figure12_slices_metrics.json").write_text(json.dumps({"burgers": stats(br, burgers), "cavity": stats(cr, cavity)}, indent=2) + "\n")

if __name__ == "__main__":
    main()
