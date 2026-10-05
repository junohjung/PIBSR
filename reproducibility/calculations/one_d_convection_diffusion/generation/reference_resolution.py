#!/usr/bin/env python3
"""Recompute the doubled-resolution check for the 1-D analytical reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[2]
DEFAULT_ARCHIVE = REPRO_ROOT / "data/one_d_convection_diffusion/selection_archive.npz"
DEFAULT_REPORTED = HERE.parent / "reported_metrics.json"


def initial_condition(np, x):
    return 0.45 * (1.0 + np.tanh((0.65 - x) / 0.03)) * 4.0 * x * (1.0 - x)


def analytical_solution(np, x, t, modes: int, quadrature_nodes: int):
    advection = 1.0
    diffusivity = 0.05
    quadrature_x = np.linspace(0.0, 1.0, quadrature_nodes, dtype=np.float64)
    mode = np.arange(1, modes + 1, dtype=np.float64)
    alpha = advection / (2.0 * diffusivity)
    integrand = (
        np.exp(-alpha * quadrature_x)[:, None]
        * initial_condition(np, quadrature_x)[:, None]
        * np.sin(np.pi * np.outer(quadrature_x, mode))
    )
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    coefficients = 2.0 * integrate(integrand, quadrature_x, axis=0)
    decay = diffusivity * (np.pi * mode) ** 2 + advection**2 / (4.0 * diffusivity)
    temporal = np.exp(-np.outer(t, decay)) * coefficients[None, :]
    spatial = np.exp(alpha * x)[:, None] * np.sin(np.pi * np.outer(x, mode))
    return temporal @ spatial.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--reported", type=Path, default=DEFAULT_REPORTED)
    parser.add_argument("--dry-run", action="store_true", help="Check paths without importing NumPy.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    args = parser.parse_args()

    missing = [str(path) for path in (args.archive, args.reported) if not path.is_file()]
    if missing:
        raise SystemExit("Missing required input(s):\n  " + "\n  ".join(missing))
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "archive": str(args.archive)}, indent=2))
        return

    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("NumPy is required for the full resolution calculation") from exc

    with np.load(args.archive, allow_pickle=False) as archive:
        x = np.asarray(archive["x"], dtype=np.float64)
        t = np.asarray(archive["t"], dtype=np.float64)
    low = analytical_solution(np, x, t, modes=900, quadrature_nodes=4097)
    high = analytical_solution(np, x, t, modes=1800, quadrature_nodes=8193)
    delta = high - low
    result = {
        "base_modes": 900,
        "base_quadrature_nodes": 4097,
        "doubled_modes": 1800,
        "doubled_quadrature_nodes": 8193,
        "evaluation_shape_t_x": [int(t.size), int(x.size)],
        "doubled_resolution_linf_change": float(np.max(np.abs(delta))),
        "doubled_resolution_relative_l2_change": float(
            np.linalg.norm(delta.ravel()) / np.linalg.norm(high.ravel())
        ),
    }
    expected = json.loads(args.reported.read_text(encoding="utf-8"))["metrics"][
        "analytical_reference_resolution"
    ]
    failures = [
        key
        for key in ("doubled_resolution_linf_change", "doubled_resolution_relative_l2_change")
        if not math.isclose(result[key], expected[key], rel_tol=5e-13, abs_tol=5e-15)
    ]
    result["status"] = "PASS" if not failures else "FAIL"
    result["failures"] = failures
    print(json.dumps(result, indent=2) if args.json or failures else "1-D reference-resolution check: PASS")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
