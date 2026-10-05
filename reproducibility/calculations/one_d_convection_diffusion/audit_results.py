#!/usr/bin/env python3
"""Audit the reported 1-D values against the archived final calculations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
DEFAULT_RESULTS = REPRO_ROOT / "results" / "one_d_convection_diffusion"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def candidate(records: list[dict[str, Any]], gamma: float) -> dict[str, Any]:
    matches = [row for row in records if float(row["gamma"]) == gamma]
    if len(matches) != 1:
        raise ValueError(f"Expected one gamma={gamma} selector record, found {len(matches)}")
    return matches[0]


def collect(results: Path) -> dict[str, Any]:
    strong_dir = results / "final_postselection" / "strong_selection_v2"
    balance_dir = results / "final_postselection" / "balance_addendum_v1"
    gamma10_dir = results / "final_postselection" / "gamma10_reporting_v1"

    strong_selection = load_json(strong_dir / "selection.json")
    strong_rows = load_json(strong_dir / "postselection_summary.json")
    data_only = next(row for row in strong_rows if row["label"] == "data_only")
    integral_selection = load_json(results / "selection" / "selection.json")
    selector_rows = load_json(results / "selection" / "selector_candidate_metrics.json")
    certificate_summary = load_json(results / "augmented_ls_certificate" / "certificate_summary.json")
    reference_resolution = load_json(results / "reference_resolution" / "reference_resolution.json")
    gamma0 = candidate(selector_rows, 0.0)
    gamma01 = candidate(selector_rows, 0.1)
    tradeoffs = load_json(balance_dir / "canonical_full_metric_tradeoffs.json")
    gamma10 = load_json(gamma10_dir / "GAMMA10_TABLE_ROW_AUDIT.json")
    selected_front = next(
        row for row in integral_selection["front"]
        if float(row["gamma"]) == float(integral_selection["selected_gamma"])
    )
    certificates = {float(row["gamma"]): row for row in certificate_summary["records"]}

    def certificate(gamma: float) -> dict[str, Any]:
        record = certificates[gamma]
        values = record["certificate"]
        return {
            "primary_objective": values["primary_objective"],
            "primary_raw_gradient_inf": values["primary_raw_gradient_inf"],
            "rows": values["rows"],
            "columns": values["cols"],
            "nonzeros": values["nnz"],
            "numerical_rank": values["numerical_rank"],
            "coefficient_relative_l2": values["coefficient_relative_l2"],
            "objective_gap_relative": values["objective_gap_relative"],
            "qr_kkt_backward_error": values["qr_kkt_backward_error"],
            "status": record["status"],
        }

    def tradeoff(name: str, role: str) -> float:
        return float(tradeoffs[name][role])

    return {
        "selection": {
            "stage_a_beta": float(strong_selection["selected_beta"]),
            "integral_balance_gamma": float(integral_selection["selected_gamma"]),
        },
        "integral_weight_selection": {
            "candidate_gamma": [float(row["gamma"]) for row in selector_rows],
            "eligible_gamma": [float(value) for value in integral_selection["eligible_gamma"]],
            "pareto_gamma": [float(value) for value in integral_selection["pareto_gamma"]],
            "selected_gamma": float(integral_selection["selected_gamma"]),
            "selected_data_ratio": selected_front["data_ratio"],
            "selected_heldout_balance_ratio": selected_front["heldout_balance_ratio"],
            "selected_ideal_distance": selected_front["signed_chord_toward_ideal_or_ideal_distance"],
            "trace_guard_rms": integral_selection["selector"]["trace_guard_rms"],
            "midpoint_pde_mse_max_ratio_to_gamma0": integral_selection["selector"][
                "midpoint_pde_mse_max_ratio_to_gamma0"
            ],
            "operator_balance_relative_tolerance": integral_selection["selector"][
                "operator_vs_independent_balance_relative_tolerance"
            ],
            "all_candidates_converged": all(bool(row["converged"]) for row in selector_rows),
            "all_operator_checks_passed": all(
                bool(row["operator_diagnostic_agreement_passed"]) for row in selector_rows
            ),
        },
        "sparse_qr_certificate": {
            "all_reference_excluded_gates_pass": bool(
                certificate_summary["all_reference_excluded_certificate_gates_pass"]
            ),
            "reference_boundary_remained_closed": bool(
                certificate_summary["reference_boundary_remained_closed"]
            ),
            "certified_gamma": [float(value) for value in certificate_summary["certified_gamma"]],
            "gamma_0": certificate(0.0),
            "gamma_0p1": certificate(0.1),
        },
        "analytical_reference_resolution": {
            "doubled_resolution_linf_change": reference_resolution[
                "doubled_resolution_linf_change"
            ],
            "doubled_resolution_relative_l2_change": reference_resolution[
                "doubled_resolution_relative_l2_change"
            ],
        },
        "models": {
            "data_only": {
                "midpoint_data_mse": data_only["midpoint"]["low_fidelity_data_mse"],
                "midpoint_pde_mse": data_only["midpoint"]["pde_mse"],
                "midpoint_analytic_relative_l2": data_only["midpoint"]["analytic_relative_l2"],
                "whole_domain_q1_gl8_time_rms": data_only["global_balance"]["gl8_time_rms"],
            },
            "matched_strong_gamma_0": {
                "midpoint_data_mse": tradeoff("midpoint_low_fidelity_data_mse", "gamma0"),
                "midpoint_pde_mse": tradeoff("midpoint_pde_mse", "gamma0"),
                "midpoint_analytic_relative_l2": tradeoff("midpoint_analytic_relative_l2", "gamma0"),
                "balance_validation_q4_gl4_mse": gamma0["q4_balance_validation_mse"],
                "whole_domain_q1_gl8_time_rms": tradeoff("global_balance_gl8_time_rms", "gamma0"),
            },
            "selected_hybrid_gamma_0p1": {
                "midpoint_data_mse": tradeoff("midpoint_low_fidelity_data_mse", "selected"),
                "midpoint_pde_mse": tradeoff("midpoint_pde_mse", "selected"),
                "midpoint_analytic_relative_l2": tradeoff("midpoint_analytic_relative_l2", "selected"),
                "balance_validation_q4_gl4_mse": gamma01["q4_balance_validation_mse"],
                "whole_domain_q1_gl8_time_rms": tradeoff("global_balance_gl8_time_rms", "selected"),
            },
            "reporting_only_gamma_10": {
                "eligible": bool(gamma10["model"]["eligible"]),
                "midpoint_data_mse": gamma10["values"]["midpoint_data_mse"],
                "midpoint_pde_mse": gamma10["values"]["midpoint_pde_mse"],
                "midpoint_analytic_relative_l2": gamma10["values"]["midpoint_analytical_relative_l2"],
                "balance_validation_q4_gl4_mse": gamma10["values"]["q4_gl4_balance_validation_mse"],
                "whole_domain_q1_gl8_time_rms": gamma10["values"]["q1_gl8_whole_domain_time_rms"],
            },
        },
    }


def compare(expected: Any, observed: Any, path: str = "metrics") -> list[str]:
    failures: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [f"{path}: expected an object"]
        for key, value in expected.items():
            if key not in observed:
                failures.append(f"{path}.{key}: missing")
            else:
                failures.extend(compare(value, observed[key], f"{path}.{key}"))
    elif isinstance(expected, bool):
        if observed is not expected:
            failures.append(f"{path}: {observed!r} != {expected!r}")
    elif isinstance(expected, (int, float)):
        if not math.isclose(float(observed), float(expected), rel_tol=5e-13, abs_tol=5e-15):
            failures.append(f"{path}: {observed!r} != {expected!r}")
    elif observed != expected:
        failures.append(f"{path}: {observed!r} != {expected!r}")
    return failures


def required_files(results: Path) -> list[Path]:
    return [
        results / "final_postselection/strong_selection_v2/selection.json",
        results / "final_postselection/strong_selection_v2/postselection_summary.json",
        results / "selection/selection.json",
        results / "selection/selector_candidate_metrics.json",
        results / "augmented_ls_certificate/certificate_summary.json",
        results / "reference_resolution/reference_resolution.json",
        results / "final_postselection/balance_addendum_v1/canonical_full_metric_tradeoffs.json",
        results / "final_postselection/gamma10_reporting_v1/GAMMA10_TABLE_ROW_AUDIT.json",
    ]


def leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(leaf_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(leaf_count(item) for item in value)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reported", type=Path, default=HERE / "reported_metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without comparing values.")
    parser.add_argument("--json", action="store_true", help="Print the audit report as JSON.")
    args = parser.parse_args()

    missing = [str(path) for path in required_files(args.results_root) if not path.is_file()]
    if not args.reported.is_file():
        missing.append(str(args.reported))
    if missing:
        raise SystemExit("Missing required input(s):\n  " + "\n  ".join(missing))

    if args.dry_run:
        report = {
            "status": "DRY_RUN_PASS",
            "scientific_evaluation_executed": False,
            "required_files": [str(path) for path in required_files(args.results_root)],
        }
    else:
        expected = load_json(args.reported)["metrics"]
        observed = collect(args.results_root)
        failures = compare(expected, observed)
        report = {
            "status": "PASS" if not failures else "FAIL",
            "metric_leaf_count": leaf_count(expected),
            "failures": failures,
        }
        if failures:
            print(json.dumps(report, indent=2))
            raise SystemExit(1)

    if args.json or args.dry_run:
        print(json.dumps(report, indent=2))
    else:
        print(f"1-D reported-metric audit: {report['status']} ({report['metric_leaf_count']} checks)")


if __name__ == "__main__":
    main()
