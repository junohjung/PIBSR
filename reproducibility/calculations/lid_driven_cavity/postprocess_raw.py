#!/usr/bin/env python3
"""Recompute cavity field, form, and pressure metrics from packaged raw arrays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
DEFAULT_RESULTS = REPRO_ROOT / "results" / "lid_driven_cavity"


def raw_paths(results: Path) -> dict[str, Path]:
    base = results / "postselection"
    return {
        "field": base
        / "ramped_cavity_matched_strong_field_diagnostics_v1/RAW_FIELD_DIAGNOSTICS.npz",
        "forms": base
        / "ramped_cavity_advective_conservative_consistency_v1/RAW_FORM_CONSISTENCY.npz",
        "pressure": base
        / "ramped_cavity_pressure_reference_diagnostics_v1/RAW_PRESSURE_DIAGNOSTICS.npz",
    }


def rmse(np: Any, values: Any) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values, dtype=np.float64)))


def relative_l2(np: Any, error: Any, reference: Any) -> float:
    numerator = np.sum(np.asarray(error, dtype=np.float64) ** 2, dtype=np.float64)
    denominator = np.sum(np.asarray(reference, dtype=np.float64) ** 2, dtype=np.float64)
    return float(np.sqrt(numerator / denominator))


def vector_rmse(np: Any, error_x: Any, error_y: Any) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.asarray(error_x, dtype=np.float64) ** 2
                + np.asarray(error_y, dtype=np.float64) ** 2,
                dtype=np.float64,
            )
        )
    )


def collect(np: Any, paths: dict[str, Path]) -> dict[str, Any]:
    with np.load(paths["field"], allow_pickle=False) as field:
        field_metrics: dict[str, Any] = {}
        target = field["final_test_target_uv"]
        for role, prefix in (
            ("matched_strong_gamma_i0", "matched"),
            ("selected_gl4_hybrid", "hybrid"),
        ):
            error = field[f"{prefix}_predicted_uv"] - target
            field_metrics[role] = {
                "offnode_strong_rmse": {
                    "divergence": rmse(np, field[f"{prefix}_strong_divergence"]),
                    "x_momentum": rmse(np, field[f"{prefix}_strong_x_momentum"]),
                    "y_momentum": rmse(np, field[f"{prefix}_strong_y_momentum"]),
                },
                "velocity_rmse": {"u": rmse(np, error[:, 0]), "v": rmse(np, error[:, 1])},
            }

    with np.load(paths["forms"], allow_pickle=False) as forms:
        form_metrics: dict[str, Any] = {}
        for role, prefix in (
            ("matched_strong_gamma_i0", "matched"),
            ("selected_gl4_hybrid", "hybrid"),
        ):
            continuity = forms[f"{prefix}_continuity"]
            correction_x = forms[f"{prefix}_correction_x"]
            correction_y = forms[f"{prefix}_correction_y"]
            if not np.allclose(correction_x, forms[f"{prefix}_u"] * continuity, rtol=0.0, atol=1e-15):
                raise ValueError(f"{role}: u*continuity correction identity failed")
            if not np.allclose(correction_y, forms[f"{prefix}_v"] * continuity, rtol=0.0, atol=1e-15):
                raise ValueError(f"{role}: v*continuity correction identity failed")
            form_metrics[role] = {
                "continuity_rmse": rmse(np, continuity),
                "advective_x_rmse": rmse(np, forms[f"{prefix}_advective_x"]),
                "advective_y_rmse": rmse(np, forms[f"{prefix}_advective_y"]),
                "u_rc_rmse": rmse(np, correction_x),
                "v_rc_rmse": rmse(np, correction_y),
                "conservative_x_rmse": rmse(np, forms[f"{prefix}_conservative_direct_x"]),
                "conservative_y_rmse": rmse(np, forms[f"{prefix}_conservative_direct_y"]),
                "identity_x_max_abs": float(np.max(np.abs(forms[f"{prefix}_identity_error_x"]))),
                "identity_y_max_abs": float(np.max(np.abs(forms[f"{prefix}_identity_error_y"]))),
            }

    with np.load(paths["pressure"], allow_pickle=False) as pressure:
        pressure_metrics: dict[str, Any] = {}
        for role, prefix in (
            ("matched_strong_gamma_i0", "matched"),
            ("selected_gl4_hybrid", "hybrid"),
        ):
            p_error = pressure[f"{prefix}_midpoint_p_aligned"] - pressure[
                "midpoint_reference_p_aligned"
            ]
            midpoint_x = pressure[f"{prefix}_midpoint_fd_grad_x"] - pressure[
                "midpoint_reference_fd_grad_x"
            ]
            midpoint_y = pressure[f"{prefix}_midpoint_fd_grad_y"] - pressure[
                "midpoint_reference_fd_grad_y"
            ]
            fine_x = pressure[f"{prefix}_fine_fd_grad_x"] - pressure[
                "fine_reference_fd_grad_x"
            ]
            fine_y = pressure[f"{prefix}_fine_fd_grad_y"] - pressure[
                "fine_reference_fd_grad_y"
            ]
            pressure_metrics[role] = {
                "pressure_rmse": rmse(np, p_error),
                "pressure_relative_l2": relative_l2(
                    np, p_error, pressure["midpoint_reference_p_aligned"]
                ),
                "midpoint_gradient_vector_rmse": vector_rmse(np, midpoint_x, midpoint_y),
                "midpoint_gradient_relative_l2": relative_l2(
                    np,
                    np.column_stack((midpoint_x, midpoint_y)),
                    np.column_stack(
                        (
                            pressure["midpoint_reference_fd_grad_x"],
                            pressure["midpoint_reference_fd_grad_y"],
                        )
                    ),
                ),
                "fine_gradient_vector_rmse": vector_rmse(np, fine_x, fine_y),
                "fine_gradient_relative_l2": relative_l2(
                    np,
                    np.column_stack((fine_x, fine_y)),
                    np.column_stack(
                        (
                            pressure["fine_reference_fd_grad_x"],
                            pressure["fine_reference_fd_grad_y"],
                        )
                    ),
                ),
            }
    return {
        "field_diagnostics": field_metrics,
        "advective_conservative": form_metrics,
        "pressure": pressure_metrics,
    }


def compare(expected: Any, observed: Any, path: str = "metrics") -> list[str]:
    failures: list[str] = []
    if isinstance(observed, dict):
        for key, value in observed.items():
            if key not in expected:
                failures.append(f"{path}.{key}: absent from reported metrics")
            else:
                failures.extend(compare(expected[key], value, f"{path}.{key}"))
    elif not math.isclose(float(observed), float(expected), rel_tol=5e-13, abs_tol=5e-15):
        failures.append(f"{path}: recomputed {observed!r} != reported {expected!r}")
    return failures


def leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(leaf_count(item) for item in value.values())
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reported", type=Path, default=HERE / "reported_metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without loading NumPy.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON.")
    args = parser.parse_args()

    paths = raw_paths(args.results_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if not args.reported.is_file():
        missing.append(str(args.reported))
    if missing:
        raise SystemExit("Missing required input(s):\n  " + "\n  ".join(missing))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "scientific_evaluation_executed": False,
                    "numpy_required_for_full_run": True,
                    "raw_inputs": {name: str(path) for name, path in paths.items()},
                },
                indent=2,
            )
        )
        return

    import numpy as np

    expected = json.loads(args.reported.read_text(encoding="utf-8"))["metrics"]
    observed = collect(np, paths)
    failures = compare(expected, observed)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "recomputed_metric_count": leaf_count(observed),
        "failures": failures,
    }
    if args.json or failures:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"Cavity raw-array postprocessing: {report['status']} "
            f"({report['recomputed_metric_count']} checks)"
        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
