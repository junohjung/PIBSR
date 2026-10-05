#!/usr/bin/env python3
"""Audit the package-wide implementation checks against their archived records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[1]
DEFAULT_RESULTS = REPRO_ROOT / "results"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_errors(path: Path) -> list[float]:
    values = [
        float(value)
        for value in re.findall(
            r"(?:relative_error|relative error)\s*=\s*([0-9.eE+-]+)",
            path.read_text(encoding="utf-8"),
        )
    ]
    if not values:
        raise ValueError(f"No directional-gradient discrepancies found in {path}")
    return values


def source_paths(results: Path) -> dict[str, Path]:
    return {
        "results/verification/burgers_strong_directional_gradient.log":
            results / "verification/burgers_strong_directional_gradient.log",
        "results/verification/cavity_strong_directional_gradient_historical.log":
            results / "verification/cavity_strong_directional_gradient_historical.log",
        "results/two_d_burgers/stage_b_interface_compatibility/GL_BALANCE_SUMMARY.json":
            results / "two_d_burgers/stage_b_interface_compatibility/GL_BALANCE_SUMMARY.json",
        "results/lid_driven_cavity/protected_gl8_v4/strong_only/run_audit.json":
            results / "lid_driven_cavity/protected_gl8_v4/strong_only/run_audit.json",
        "results/two_d_burgers/postselection/POSTSELECTION_SUMMARY.json":
            results / "two_d_burgers/postselection/POSTSELECTION_SUMMARY.json",
    }


def collect(results: Path) -> dict[str, float]:
    paths = source_paths(results)
    burgers_log = paths["results/verification/burgers_strong_directional_gradient.log"]
    cavity_log = paths[
        "results/verification/cavity_strong_directional_gradient_historical.log"
    ]
    balance = load_json(
        paths[
            "results/two_d_burgers/stage_b_interface_compatibility/GL_BALANCE_SUMMARY.json"
        ]
    )
    cavity = load_json(
        paths[
            "results/lid_driven_cavity/protected_gl8_v4/strong_only/run_audit.json"
        ]
    )
    postselection = load_json(
        paths["results/two_d_burgers/postselection/POSTSELECTION_SUMMARY.json"]
    )
    cavity_components = cavity["production_comparison"]["components"].values()
    derivative_checks = postselection["evaluation"]["physical_derivative_self_checks"].values()
    return {
        "burgers_strong_max_relative_directional_discrepancy": max(
            relative_errors(burgers_log)
        ),
        "navier_stokes_strong_max_relative_directional_discrepancy": max(
            relative_errors(cavity_log)
        ),
        "burgers_balance_max_direct_flux_closure_discrepancy": balance[
            "flux_direct_closure"
        ]["maximum_flux_direct_mismatch"],
        "cavity_cross_implementation_max_absolute_difference": max(
            float(component["maximum_absolute_difference"])
            for component in cavity_components
        ),
        "physical_derivative_max_scaled_centered_difference_discrepancy": max(
            float(check["max_scaled_error"]) for check in derivative_checks
        ),
    }


def compare(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    failures = []
    for key, value in expected.items():
        if key not in observed:
            failures.append(f"metrics.{key}: missing")
        elif not math.isclose(
            float(observed[key]), float(value), rel_tol=5e-13, abs_tol=5e-18
        ):
            failures.append(f"metrics.{key}: {observed[key]!r} != {value!r}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reported", type=Path, default=HERE / "reported_metrics.json")
    parser.add_argument("--dry-run", action="store_true", help="Check paths and hashes only.")
    parser.add_argument("--json", action="store_true", help="Print the audit report as JSON.")
    args = parser.parse_args()

    if not args.reported.is_file():
        raise SystemExit(f"Missing reported metric record: {args.reported}")
    record = load_json(args.reported)
    paths = source_paths(args.results_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing required input(s):\n  " + "\n  ".join(missing))

    hash_failures = [
        f"{relative}: SHA-256 mismatch"
        for relative, path in paths.items()
        if sha256(path) != record["source_sha256"][relative]
    ]
    if args.dry_run:
        report = {
            "status": "DRY_RUN_PASS" if not hash_failures else "FAIL",
            "scientific_evaluation_executed": False,
            "source_hash_failures": hash_failures,
        }
    else:
        failures = hash_failures + compare(record["metrics"], collect(args.results_root))
        report = {
            "status": "PASS" if not failures else "FAIL",
            "metric_leaf_count": len(record["metrics"]),
            "failures": failures,
        }

    if report["status"] == "FAIL":
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    if args.json or args.dry_run:
        print(json.dumps(report, indent=2))
    else:
        print(
            "Package-wide verification audit: "
            f"{report['status']} ({report['metric_leaf_count']} checks)"
        )


if __name__ == "__main__":
    main()
