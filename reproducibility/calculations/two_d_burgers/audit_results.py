#!/usr/bin/env python3
"""Audit the reported Burgers values against the archived final calculations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import hashlib
import struct
from typing import Any


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
DEFAULT_RESULTS = REPRO_ROOT / "results" / "two_d_burgers"
DEFAULT_POINT_SETS = REPRO_ROOT / "point_sets" / "two_d_burgers"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def one(rows: list[dict[str, str]], **keys: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one TSV row for {keys}, found {len(matches)}")
    return matches[0]


def trajectory_rms(
    rows: list[dict[str, str]], model: str, order: str, column: str
) -> float:
    values = [
        float(row[column])
        for row in rows
        if row["model_id"] == model and row["quadrature_order"] == order
    ]
    if len(values) != 200:
        raise ValueError(f"Expected 200 trajectory rows for {model}/GL{order}, found {len(values)}")
    return math.sqrt(math.fsum(value * value for value in values) / len(values))


def time_design_hash(rows: list[dict[str, str]]) -> str:
    first_model = rows[0]["model_id"]
    first_order = rows[0]["quadrature_order"]
    times = [
        float(row["time"])
        for row in rows
        if row["model_id"] == first_model and row["quadrature_order"] == first_order
    ]
    digest = hashlib.sha256()
    digest.update(b"float64")
    digest.update(json.dumps((len(times),)).encode("ascii"))
    digest.update(struct.pack(f"<{len(times)}d", *times))
    return digest.hexdigest()


def index_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def collect(results: Path, point_sets: Path) -> dict[str, Any]:
    beta = load_json(results / "selection/BETA_SELECTION_SEAL.json")
    kappa = load_json(results / "kappa_screen/KAPPA_SELECTION_SEAL.json")
    kappa_records = {
        value: load_json(results / f"kappa_screen/kappa_{value}/SCREEN_RECORD.json")
        for value in (1, 3, 10, 30)
    }
    stage_a = load_json(results / "selection/STAGE_A_SELECTION_SEAL.json")
    stage_a_candidates = load_json(results / "selection/STAGE_A_CLASSIFIED_CANDIDATES.json")
    stage_a_summary = load_json(results / "selection/stage_a_design_a/GL_BALANCE_SUMMARY.json")
    stage_a_trajectories = load_tsv(
        results / "selection/stage_a_design_a/GL_BALANCE_TRAJECTORIES.tsv"
    )
    trajectories = load_tsv(results / "stage_b_interface_compatibility/GL_BALANCE_TRAJECTORIES.tsv")
    stage_b_gate = load_json(results / "stage_b_interface_compatibility/STAGE_B_GATE.json")
    reference = load_tsv(results / "postselection/reference_metrics.tsv")
    final = load_tsv(results / "postselection/final_test_metrics.tsv")
    postselection_summary = load_json(results / "postselection/POSTSELECTION_SUMMARY.json")
    refinement = load_json(results / "reference_refinement/burgers_reference_refinement.json")

    def balance(model: str) -> dict[str, float]:
        return {
            "gl4_q4_u_time_rms": trajectory_rms(trajectories, model, "4", "q4_scaled_rms_u"),
            "gl4_q4_v_time_rms": trajectory_rms(trajectories, model, "4", "q4_scaled_rms_v"),
            "gl8_q4_u_time_rms": trajectory_rms(trajectories, model, "8", "q4_scaled_rms_u"),
            "gl8_q4_v_time_rms": trajectory_rms(trajectories, model, "8", "q4_scaled_rms_v"),
            "gl8_q1_u_raw_time_rms": trajectory_rms(trajectories, model, "8", "q1_raw_abs_u"),
            "gl8_q1_v_raw_time_rms": trajectory_rms(trajectories, model, "8", "q1_raw_abs_v"),
        }

    def reference_value(model: str, family: str, metric: str) -> float:
        return float(one(reference, model_id=model, family=family, metric=metric)["value"])

    def final_value(model: str, metric: str) -> float:
        return float(one(final, model_id=model, metric=metric)["value"])

    def postselection(model: str) -> dict[str, float]:
        return {
            "offnode_strong_residual_u_rms": reference_value(model, "test_a_field", "strong_residual_u_rms"),
            "offnode_strong_residual_v_rms": reference_value(model, "test_a_field", "strong_residual_v_rms"),
            "offnode_velocity_relative_l2": reference_value(model, "test_a_field", "relative_l2_velocity"),
            "offnode_vorticity_relative_l2": reference_value(model, "test_a_field", "relative_l2_vorticity"),
            "kinetic_energy_trajectory_relative_l2": reference_value(
                model, "test_b_mechanics", "kinetic_energy_trajectory_relative_l2"
            ),
            "enstrophy_trajectory_relative_l2": reference_value(
                model, "test_b_mechanics", "enstrophy_trajectory_relative_l2"
            ),
            "legacy_q1_u_time_rms": reference_value(
                model, "test_b_mechanics", "legacy_global_integrated_equation_defect_u_time_rms"
            ),
            "legacy_q1_v_time_rms": reference_value(
                model, "test_b_mechanics", "legacy_global_integrated_equation_defect_v_time_rms"
            ),
            "final_velocity_relative_l2": final_value(model, "relative_l2_velocity"),
            "final_vorticity_relative_l2": final_value(model, "relative_l2_vorticity"),
        }

    comparisons = refinement["comparisons"]

    def comparison(study: str, coarse: str, fine: str, time: float) -> dict[str, Any]:
        matches = [
            row
            for row in comparisons
            if row["study"] == study
            and row["coarser"] == coarse
            and row["finer"] == fine
            and float(row["time"]) == time
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one refinement comparison for {study}/{coarse}/{fine}/t={time}")
        return matches[0]

    n40 = comparison("spatial", "space_n40_dt2.5e-4", "space_n80_dt2.5e-4", 2.0)
    n80 = comparison("spatial", "space_n80_dt2.5e-4", "space_n160_dt2.5e-4", 2.0)
    n200 = comparison("reported_reference", "time_n160_dt5e-4", "reported_n200_dt5e-4", 2.0)
    tight_time = [
        row
        for row in comparisons
        if row["study"] == "temporal"
        and row["coarser"] == "time_n160_dt5e-4"
        and row["finer"] == "space_n160_dt2.5e-4"
    ]
    if len(tight_time) != 4:
        raise ValueError("Expected four fine time-step comparisons")

    candidate_by_gamma = {float(row["gamma"]): row for row in stage_a_candidates}
    if set(candidate_by_gamma) != {0.01, 0.03, 0.1}:
        raise ValueError("Stage-A candidate set is not exactly {0.01, 0.03, 0.1}")
    if stage_a_summary["time_count"] != 200 or len(stage_a_trajectories) != 1000:
        raise ValueError("Stage-A does not contain five complete 200-time trajectories")

    sealed_kappa = {int(float(row["kappa"])): row for row in kappa["records"]}
    if set(sealed_kappa) != set(kappa_records):
        raise ValueError("Kappa seal does not contain exactly {1, 3, 10, 30}")
    for value, record in kappa_records.items():
        if record != sealed_kappa[value]:
            raise ValueError(f"Kappa {value} screen record differs from the selection seal")

    def kappa_screen(value: int) -> dict[str, Any]:
        guard = kappa_records[value]["absolute_trace_guard"]
        return {
            "maximum_normalized_trace_rms": max(
                float(check["normalized_rms"]) for check in guard["checks"].values()
            ),
            "passed": bool(guard["passed"]),
        }

    # Independently recompute the two reported Stage-A q4 values for each candidate.
    for model in ("gamma_0p01", "gamma_0p03", "gamma_0p1", "matched_gamma0"):
        for component in ("u", "v"):
            observed = trajectory_rms(
                stage_a_trajectories, model, "4", f"q4_scaled_rms_{component}"
            )
            frozen = stage_a_summary["summary"][model]["gl4"][f"q4_{component}_time_rms"]
            if not math.isclose(observed, frozen, rel_tol=5e-13, abs_tol=5e-15):
                raise ValueError(f"Stage-A summary mismatch for {model}/q4_{component}")

    return {
        "selection": {
            "beta": float(beta["selected"]["beta"]),
            "kappa": float(kappa["selected_kappa_literal"]),
            "gamma": float(stage_a["selected"]["gamma"]),
            "beta_selector_endpoint_chord_distance": beta["selected"][
                "selector_endpoint_chord_distance"
            ],
            "kappa_trace_screen": {
                "normalized_rms_limit": 0.001,
                "kappa_1": kappa_screen(1),
                "kappa_3": kappa_screen(3),
                "kappa_10": kappa_screen(10),
                "kappa_30": kappa_screen(30),
            },
        },
        "time_roles": {
            "optimizer_integral_balance": {
                "fit_time_count": index_count(point_sets / "integral_balance/fit_time_indices.txt"),
                "validation_time_count": index_count(
                    point_sets / "integral_balance/validation_time_indices.txt"
                ),
            },
            "weight_selection_design_a": {
                "time_count": int(stage_a_summary["time_count"]),
                "times_sha256": time_design_hash(stage_a_trajectories),
                "gamma_0p01_geometric_ratio": candidate_by_gamma[0.01]["eligibility_guard"]["geometric_ratio"],
                "gamma_0p01_eligible": candidate_by_gamma[0.01]["eligible"],
                "gamma_0p03_geometric_ratio": candidate_by_gamma[0.03]["eligibility_guard"]["geometric_ratio"],
                "gamma_0p03_eligible": candidate_by_gamma[0.03]["eligible"],
                "gamma_0p1_geometric_ratio": candidate_by_gamma[0.1]["eligibility_guard"]["geometric_ratio"],
                "gamma_0p1_eligible": candidate_by_gamma[0.1]["eligible"],
            },
            "confirmation_design_b": {
                "time_count": int(stage_b_gate["evaluation"]["time_count"]),
                "times_sha256": time_design_hash(trajectories),
                "gl4_selected_to_matched_geometric_ratio": stage_b_gate[
                    "matched_q4_final_confirmation"
                ]["gl4"]["geometric_ratio"],
                "gl8_selected_to_matched_geometric_ratio": stage_b_gate[
                    "matched_q4_final_confirmation"
                ]["gl8"]["geometric_ratio"],
            },
        },
        "balance": {
            "matched_strong_gamma_0": balance("matched_gamma0"),
            "selected_hybrid_gamma_0p03": balance("selected"),
        },
        "postselection": {
            "matched_strong_gamma_0": postselection("matched_gamma0"),
            "selected_hybrid_gamma_0p03": postselection("selected"),
        },
        "reference_refinement_at_t2": {
            "n40_to_n80_relative_l2_u": n40["relative_l2_u"],
            "n40_to_n80_relative_l2_v": n40["relative_l2_v"],
            "n80_to_n160_relative_l2_u": n80["relative_l2_u"],
            "n80_to_n160_relative_l2_v": n80["relative_l2_v"],
            "n160_to_n200_relative_l2_u": n200["relative_l2_u"],
            "n160_to_n200_relative_l2_v": n200["relative_l2_v"],
            "dt_5e4_to_2p5e4_max_relative_l2": max(
                float(row[key]) for row in tight_time for key in ("relative_l2_u", "relative_l2_v")
            ),
        },
        "reference_interpolation_sensitivity": postselection_summary["evaluation"][
            "reference_interpolation_sensitivity"
        ],
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
    elif isinstance(expected, (int, float)):
        if not math.isclose(float(observed), float(expected), rel_tol=5e-13, abs_tol=5e-15):
            failures.append(f"{path}: {observed!r} != {expected!r}")
    elif observed != expected:
        failures.append(f"{path}: {observed!r} != {expected!r}")
    return failures


def required_files(results: Path) -> list[Path]:
    return [
        results / "selection/BETA_SELECTION_SEAL.json",
        results / "kappa_screen/KAPPA_SELECTION_SEAL.json",
        results / "kappa_screen/kappa_1/SCREEN_RECORD.json",
        results / "kappa_screen/kappa_3/SCREEN_RECORD.json",
        results / "kappa_screen/kappa_10/SCREEN_RECORD.json",
        results / "kappa_screen/kappa_30/SCREEN_RECORD.json",
        results / "selection/STAGE_A_SELECTION_SEAL.json",
        results / "selection/STAGE_A_CLASSIFIED_CANDIDATES.json",
        results / "selection/stage_a_design_a/GL_BALANCE_SUMMARY.json",
        results / "selection/stage_a_design_a/GL_BALANCE_TRAJECTORIES.tsv",
        results / "stage_b_interface_compatibility/GL_BALANCE_TRAJECTORIES.tsv",
        results / "stage_b_interface_compatibility/STAGE_B_GATE.json",
        results / "postselection/reference_metrics.tsv",
        results / "postselection/final_test_metrics.tsv",
        results / "postselection/POSTSELECTION_SUMMARY.json",
        results / "reference_refinement/burgers_reference_refinement.json",
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
    parser.add_argument("--point-sets-root", type=Path, default=DEFAULT_POINT_SETS)
    parser.add_argument("--reported", type=Path, default=HERE / "reported_metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without comparing values.")
    parser.add_argument("--json", action="store_true", help="Print the audit report as JSON.")
    args = parser.parse_args()

    missing = [str(path) for path in required_files(args.results_root) if not path.is_file()]
    point_files = [
        args.point_sets_root / "integral_balance/fit_time_indices.txt",
        args.point_sets_root / "integral_balance/validation_time_indices.txt",
    ]
    missing.extend(str(path) for path in point_files if not path.is_file())
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
        failures = compare(expected, collect(args.results_root, args.point_sets_root))
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
        print(f"Burgers reported-metric audit: {report['status']} ({report['metric_leaf_count']} checks)")


if __name__ == "__main__":
    main()
