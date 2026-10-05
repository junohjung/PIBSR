#!/usr/bin/env python3
"""Reproduce the unforced Burgers space/time refinement table."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

np = None


HERE = Path(__file__).resolve().parent
CHECKPOINTS = (0.5, 1.0, 1.5, 2.0)
REPORT_N = 40
CONFIGURATIONS = {
    "space_n40_dt2.5e-4": (40, 2.5e-4),
    "space_n80_dt2.5e-4": (80, 2.5e-4),
    "space_n160_dt2.5e-4": (160, 2.5e-4),
    "time_n160_dt1e-3": (160, 1.0e-3),
    "time_n160_dt5e-4": (160, 5.0e-4),
    "reported_n200_dt5e-4": (200, 5.0e-4),
}


def load_core():
    path = HERE / "generate_burgers_observed.py"
    spec = importlib.util.spec_from_file_location("burgers_generation_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def solve(core, n: int, dt: float) -> dict[float, np.ndarray]:
    params = dataclasses.replace(core.Params(), nx=n, ny=n, dt=dt)
    _, _, xx, yy, dx, dy = core.make_grid(params.Lx, params.Ly, n, n)
    scale_u, scale_v = core.compute_common_ic_scaling(params)
    u, v = core.make_initial_condition(params, xx, yy, scale_u, scale_v)
    solver = core.HighOrderBurgers2D(n, n, dx, dy, params.nu)
    zeros = np.zeros_like(u)
    stride = n // REPORT_N
    if n % REPORT_N:
        raise ValueError(f"n={n} is not nested with report grid n={REPORT_N}")
    result: dict[float, np.ndarray] = {}
    time = 0.0
    for target in CHECKPOINTS:
        while time < target - 1.0e-15:
            step = min(dt, target - time)
            solver.step_rk4(u, v, zeros, zeros, step)
            time += step
        result[target] = np.stack((u[::stride, ::stride], v[::stride, ::stride]), axis=-1)
    return result


def metrics(coarse: np.ndarray, fine: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, field in enumerate(("u", "v")):
        difference = coarse[..., index] - fine[..., index]
        values[f"relative_l2_{field}"] = float(
            np.linalg.norm(difference.ravel()) / np.linalg.norm(fine[..., index].ravel())
        )
        values[f"linf_{field}"] = float(np.max(np.abs(difference)))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "scientific_solve_executed": False,
                    "configurations": CONFIGURATIONS,
                    "checkpoints": CHECKPOINTS,
                    "common_reporting_grid": "41 x 41 nested nodes",
                },
                indent=2,
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is used")

    global np
    try:
        import numpy as numpy_module
    except ImportError as error:
        raise SystemExit("Full refinement requires NumPy; --dry-run does not.") from error
    np = numpy_module

    output_dir = args.output_dir.resolve()
    json_path = output_dir / "burgers_reference_refinement.json"
    tsv_path = output_dir / "burgers_reference_refinement.tsv"
    if json_path.exists() or tsv_path.exists():
        raise FileExistsError("Refusing to overwrite an existing refinement result")

    core = load_core()
    solutions = {
        name: solve(core, n=n, dt=dt) for name, (n, dt) in CONFIGURATIONS.items()
    }
    pairs = (
        ("spatial", "space_n40_dt2.5e-4", "space_n80_dt2.5e-4"),
        ("spatial", "space_n80_dt2.5e-4", "space_n160_dt2.5e-4"),
        ("temporal", "time_n160_dt1e-3", "time_n160_dt5e-4"),
        ("temporal", "time_n160_dt5e-4", "space_n160_dt2.5e-4"),
        ("reported_reference", "time_n160_dt5e-4", "reported_n200_dt5e-4"),
    )
    rows = []
    for study, coarse, fine in pairs:
        for time in CHECKPOINTS:
            row = {"study": study, "coarser": coarse, "finer": fine, "time": time}
            row.update(metrics(solutions[coarse][time], solutions[fine][time]))
            rows.append(row)

    result = {
        "solver": "fourth-order centered finite differences plus RK4",
        "domain": [0.0, 2.0, 0.0, 2.0],
        "nu": 0.01,
        "common_reporting_grid": "41 x 41 nested nodes",
        "checkpoints": list(CHECKPOINTS),
        "configurations": {
            name: {"cells_x": n, "cells_y": n, "dt": dt}
            for name, (n, dt) in CONFIGURATIONS.items()
        },
        "comparisons": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    columns = list(rows[0])
    tsv_path.write_text(
        "\t".join(columns)
        + "\n"
        + "\n".join("\t".join(str(row[column]) for column in columns) for row in rows)
        + "\n"
    )
    print(json.dumps({"status": "COMPLETE", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
