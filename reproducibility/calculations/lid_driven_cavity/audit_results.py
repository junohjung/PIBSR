#!/usr/bin/env python3
"""Audit reported cavity values against the packaged final result records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
DEFAULT_RESULTS = REPRO_ROOT / "results" / "lid_driven_cavity"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(rows: list[dict[str, str]], **keys: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one TSV row for {keys}, found {len(matches)}")
    return matches[0]


def gl8_metrics(path: Path) -> dict[str, Any]:
    rows = load_tsv(path)
    partitions = {
        "canonical_q4": "canonical_q4",
        "whole_domain_q1": "aggregate_q1",
        "shifted_q4": "shifted_q4",
    }
    result: dict[str, Any] = {}
    for label, partition in partitions.items():
        result[label] = {}
        expected_count = "200" if partition == "aggregate_q1" else "3200"
        for component in ("mass", "x_momentum", "y_momentum"):
            row = one(rows, partition=partition, component=component)
            if row["sample_count"] != expected_count:
                raise ValueError(f"Unexpected GL8 sample count for {partition}/{component}")
            result[label][component] = float(row["rms"])
    return result


def collect(results: Path) -> dict[str, Any]:
    selection = load_json(results / "family_selection_v4.json")
    selected = selection["selected"]
    matched_summary = load_json(
        results
        / "postselection/ramped_cavity_matched_strong_control_gl8_reporting_v1/SUMMARY.json"
    )
    field_summary = load_json(
        results
        / "postselection/ramped_cavity_matched_strong_field_diagnostics_v1/SUMMARY.json"
    )
    stage_a_comparison_path = results / "stage_a_comparison_v4/post_failure_diagnostic_v4.json"
    stage_a_metrics_path = results / "stage_a_normalizers_v3/stage_a_metrics_v3.json"
    expected_hashes = {
        stage_a_comparison_path: "3acdaae4aa69c77c26addc7daccc682065409561292b9e3558ca9d7518f9b9a0",
        stage_a_metrics_path: "ea9596191fe730d3b2f87ce7163a9061297d5415df000c6ffc89e5be34c84daa",
    }
    for path, expected_hash in expected_hashes.items():
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Archived source hash mismatch: {path}")
    stage_a_comparison = load_json(stage_a_comparison_path)
    stage_a_metrics = load_json(stage_a_metrics_path)
    normalization = load_json(results / "normalization_v4.json")
    pressure_summary = load_json(
        results
        / "postselection/ramped_cavity_pressure_reference_diagnostics_v1/SUMMARY.json"
    )
    forms = load_tsv(
        results / "postselection/ramped_cavity_advective_conservative_consistency_v1/METRICS.tsv"
    )
    pressure = load_tsv(
        results / "postselection/ramped_cavity_pressure_reference_diagnostics_v1/PRESSURE_METRICS_POOLED.tsv"
    )

    def selected_family(family: str) -> dict[str, Any]:
        record = selected[family]
        return {
            "candidate_id": record["candidate_id"],
            "gamma": record["gamma"],
            "worst_aligned_physics_ratio": record["worst_aligned_physics_ratio"],
            "geometric_mean_aligned_physics_ratio": record[
                "geometric_mean_aligned_physics_ratio"
            ],
        }

    audit_by_id = {row["candidate_id"]: row for row in selection["audit"]}
    gamma_grid = sorted({float(row["gamma"]) for row in selection["audit"]})
    if len(selection["audit"]) != 16 or gamma_grid != [0.01, 0.02, 0.03, 0.04]:
        raise ValueError("Cavity selection grid is not the frozen 4-family by 4-gamma grid")

    gl8_paths = {
        "stage_a": results / "protected_gl8_v4/stage_a/gl8_summary.tsv",
        "matched_strong_gamma_i0": results
        / "postselection/ramped_cavity_matched_strong_control_gl8_reporting_v1/independent_python/gl8_summary.tsv",
        "selected_gl4_hybrid": results / "protected_gl8_v4/gl4_hybrid/gl8_summary.tsv",
        "selected_strong_only": results / "protected_gl8_v4/strong_only/gl8_summary.tsv",
    }

    def form_metrics(role: str) -> dict[str, float]:
        row = one(forms, model=role)
        return {
            key: float(row[key])
            for key in (
                "continuity_rmse",
                "advective_x_rmse",
                "advective_y_rmse",
                "u_rc_rmse",
                "v_rc_rmse",
                "conservative_x_rmse",
                "conservative_y_rmse",
                "identity_x_max_abs",
                "identity_y_max_abs",
            )
        }

    def pressure_metrics(role: str) -> dict[str, float]:
        row = one(pressure, role=role)
        return {
            key: float(row[key])
            for key in (
                "pressure_rmse",
                "pressure_relative_l2",
                "midpoint_gradient_vector_rmse",
                "midpoint_gradient_relative_l2",
                "fine_gradient_vector_rmse",
                "fine_gradient_relative_l2",
            )
        }

    pointwise = stage_a_comparison["diagnostic_metrics"]["pointwise_metrics"]
    stage_a_pointwise = pointwise["stage_a"]
    hybrid_pointwise = pointwise["gl4_hybrid"]
    archived_ratios = stage_a_comparison["diagnostic_metrics"]["values"]

    def ratio(family: str, component: str) -> float:
        numerator = hybrid_pointwise[family][component]
        denominator = stage_a_pointwise[family][component]
        return float(numerator / denominator)

    offnode_ratio = {
        component: ratio("offnode_strong_rmse", component)
        for component in ("divergence", "x_momentum", "y_momentum")
    }
    velocity_ratio = {
        component: ratio("sealed_velocity_rmse", component) for component in ("u", "v")
    }
    ratio_keys = {
        "offnode_strong": "offnode_strong_ratios_to_stage_a",
        "sealed_velocity": "velocity_sealed_ratios_to_stage_a",
    }
    for family, recomputed in (("offnode_strong", offnode_ratio), ("sealed_velocity", velocity_ratio)):
        for component, value in recomputed.items():
            if not math.isclose(
                value,
                archived_ratios[ratio_keys[family]][component],
                rel_tol=5e-13,
                abs_tol=5e-15,
            ):
                raise ValueError(f"Stage-A comparison ratio mismatch for {family}/{component}")

    field_metrics = field_summary["absolute_metrics"]
    for family in ("offnode_strong_rmse", "sealed_velocity_rmse"):
        packaged_family = (
            field_metrics["selected_gl4_hybrid"][
                "velocity_rmse" if family == "sealed_velocity_rmse" else family
            ]
        )
        for component, value in hybrid_pointwise[family].items():
            if component not in packaged_family:
                continue
            if not math.isclose(
                float(value), float(packaged_family[component]), rel_tol=5e-13, abs_tol=5e-15
            ):
                raise ValueError(f"V8/V14 hybrid field mismatch for {family}/{component}")

    if normalization["source_stage_a_metrics_sha256"] != "8cf33433020274d246048e032ccb50419288d4e32e10462c3578c375c7448304":
        raise ValueError("Stage-A metric artifact is not bound to the normalization source")
    hybrid_weights = normalization["candidate_weights"]["gl4_hybrid__gamma_0p04"]
    if not hybrid_weights["equal_total_budget_verified"]:
        raise ValueError("Selected hybrid does not satisfy the equal-budget normalization")

    return {
        "within_family_selection": {
            "candidate_count": len(selection["audit"]),
            "gamma_grid": gamma_grid,
            "gl4_conservative_only": selected_family("gl4_conservative_only"),
            "gl4_hybrid": selected_family("gl4_hybrid"),
            "strong_only": selected_family("strong_only"),
            "trapezoid_hybrid_selected": selected["trapezoid_hybrid"],
            "conservative_gamma_0p03_reasons": audit_by_id[
                "gl4_conservative_only__gamma_0p03"
            ]["reasons"],
            "strong_only_gamma_0p04_reasons": audit_by_id[
                "strong_only__gamma_0p04"
            ]["reasons"],
        },
        "matched_reporting_control": {
            "candidate_id": matched_summary["contract"]["matched_control_candidate_id"],
            "gamma_s": matched_summary["contract"]["scientific_parameterization"][
                "matched_strong_control"
            ]["gamma_S"],
            "gamma_i": matched_summary["contract"]["scientific_parameterization"][
                "matched_strong_control"
            ]["gamma_I"],
            "strong_weights_exactly_equal": matched_summary["contract"][
                "strong_weights_exactly_equal"
            ],
        },
        "gl8_balance": {name: gl8_metrics(path) for name, path in gl8_paths.items()},
        "field_diagnostics": {
            "stage_a": {
                "offnode_strong_rmse": stage_a_pointwise["offnode_strong_rmse"],
                "velocity_rmse": stage_a_pointwise["sealed_velocity_rmse"],
            },
            "matched_strong_gamma_i0": field_metrics["matched_strong_gamma_i0"],
            "selected_gl4_hybrid": field_metrics["selected_gl4_hybrid"],
            "selected_hybrid_vs_stage_a": {
                "offnode_strong_ratio": offnode_ratio,
                "offnode_strong_percent_change": {
                    component: 100.0 * (value - 1.0)
                    for component, value in offnode_ratio.items()
                },
                "velocity_ratio": velocity_ratio,
            },
        },
        "stage_a_normalizers_and_selected_weights": {
            "gamma_s": 0.5 * float(hybrid_weights["gamma"]),
            "gamma_i": 0.5 * float(hybrid_weights["gamma"]),
            "stage_a_objective": stage_a_metrics["stage_a_objective"],
            "stage_a_strong_mse": {
                "x_momentum": stage_a_metrics["strong_component_mse"]["xmom"],
                "y_momentum": stage_a_metrics["strong_component_mse"]["ymom"],
                "divergence": stage_a_metrics["strong_component_mse"]["div"],
            },
            "stage_a_gl4_mse": {
                "mass": stage_a_metrics["gl4_component_mse"]["mass"],
                "x_momentum": stage_a_metrics["gl4_component_mse"]["xmom"],
                "y_momentum": stage_a_metrics["gl4_component_mse"]["ymom"],
            },
            "stage_a_trapezoid_mse": {
                "mass": stage_a_metrics["trapezoid_component_mse"]["mass"],
                "x_momentum": stage_a_metrics["trapezoid_component_mse"]["xmom"],
                "y_momentum": stage_a_metrics["trapezoid_component_mse"]["ymom"],
            },
            "selected_strong_weights": {
                "x_momentum": hybrid_weights["strong_weights"]["xmom"],
                "y_momentum": hybrid_weights["strong_weights"]["ymom"],
                "divergence": hybrid_weights["strong_weights"]["div"],
            },
            "selected_gl4_weights": {
                "mass": hybrid_weights["integral_weights"]["mass"],
                "x_momentum": hybrid_weights["integral_weights"]["xmom"],
                "y_momentum": hybrid_weights["integral_weights"]["ymom"],
            },
        },
        "advective_conservative": {
            "matched_strong_gamma_i0": form_metrics("matched_strong_gamma_i0"),
            "selected_gl4_hybrid": form_metrics("selected_gl4_hybrid"),
        },
        "pressure": {
            "matched_strong_gamma_i0": pressure_metrics("matched_strong_gamma_i0"),
            "selected_gl4_hybrid": pressure_metrics("selected_gl4_hybrid"),
            "directional_counts": pressure_summary["hybrid_vs_matched"][
                "directional_consistency"
            ],
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
    elif isinstance(expected, list):
        if observed != expected:
            failures.append(f"{path}: {observed!r} != {expected!r}")
    elif isinstance(expected, bool):
        if observed is not expected:
            failures.append(f"{path}: {observed!r} != {expected!r}")
    elif isinstance(expected, (int, float)):
        if not math.isclose(float(observed), float(expected), rel_tol=5e-13, abs_tol=5e-15):
            failures.append(f"{path}: {observed!r} != {expected!r}")
    elif observed != expected:
        failures.append(f"{path}: {observed!r} != {expected!r}")
    return failures


def leaf_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(leaf_count(item) for item in value.values())
    return 1


def required_files(results: Path) -> list[Path]:
    return [
        results / "family_selection_v4.json",
        results / "protected_gl8_v4/stage_a/gl8_summary.tsv",
        results / "protected_gl8_v4/gl4_hybrid/gl8_summary.tsv",
        results / "protected_gl8_v4/strong_only/gl8_summary.tsv",
        results
        / "postselection/ramped_cavity_matched_strong_control_gl8_reporting_v1/independent_python/gl8_summary.tsv",
        results
        / "postselection/ramped_cavity_matched_strong_control_gl8_reporting_v1/SUMMARY.json",
        results
        / "postselection/ramped_cavity_matched_strong_field_diagnostics_v1/SUMMARY.json",
        results / "stage_a_comparison_v4/post_failure_diagnostic_v4.json",
        results / "stage_a_normalizers_v3/stage_a_metrics_v3.json",
        results / "normalization_v4.json",
        results
        / "postselection/ramped_cavity_advective_conservative_consistency_v1/METRICS.tsv",
        results
        / "postselection/ramped_cavity_pressure_reference_diagnostics_v1/PRESSURE_METRICS_POOLED.tsv",
        results
        / "postselection/ramped_cavity_pressure_reference_diagnostics_v1/SUMMARY.json",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reported", type=Path, default=HERE / "reported_metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without comparing values.")
    parser.add_argument("--json", action="store_true", help="Print the audit report as JSON.")
    args = parser.parse_args()

    inputs = required_files(args.results_root)
    missing = [str(path) for path in inputs if not path.is_file()]
    if not args.reported.is_file():
        missing.append(str(args.reported))
    if missing:
        raise SystemExit("Missing required input(s):\n  " + "\n  ".join(missing))

    if args.dry_run:
        report = {
            "status": "DRY_RUN_PASS",
            "scientific_evaluation_executed": False,
            "required_files": [str(path) for path in inputs],
        }
    else:
        expected = load_json(args.reported)["metrics"]
        failures = compare(expected, collect(args.results_root))
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
        print(f"Cavity reported-metric audit: {report['status']} ({report['metric_leaf_count']} checks)")


if __name__ == "__main__":
    main()
