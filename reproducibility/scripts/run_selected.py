#!/usr/bin/env python3
"""Run one selected PI-BSR model from the packaged inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable


PACKAGE_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PACKAGE_DIR.parent


def one_d_spec(build_root: Path, output: Path) -> tuple[list[str], dict[str, str]]:
    points = PACKAGE_DIR / "point_sets" / "one_d_convection_diffusion"
    command = [
        str(build_root / "one_d" / "examples" / "pde" / "flowdata"),
        "-i", "ConDiff1D",
        "-f", str(PACKAGE_DIR / "data" / "one_d_convection_diffusion"),
        "-v", "40", "-v", "40", "-q", "3", "-x", "1",
    ]
    environment = {
        "MFA_LAMBDA_DATA": "1",
        "MFA_LAMBDA_PDE": "3e-5",
        "MFA_LAMBDA_IC": "3e-5",
        "MFA_LAMBDA_BC": "3e-5",
        "MFA_LAMBDA_INT_BALANCE": "0.00026372967032555535",
        "MFA_INITIAL_CTRL_PATH": str(
            PACKAGE_DIR
            / "checkpoints"
            / "one_d_convection_diffusion"
            / "stage_a_initializer.txt"
        ),
        "MFA_INT_FIT_TIME_INDEX_FILE": str(points / "fit_time_indices.txt"),
        "MFA_INT_VALID_TIME_INDEX_FILE": str(points / "validation_time_indices.txt"),
        "MFA_INT_PARTS_X": "4",
        "MFA_INT_QUADRATURE": "gl4",
        "MFA_INT_DIAGNOSTICS": "1",
        "MFA_INT_DIRECTIONAL_CHECK": "0",
        "MFA_INT_DIRECTIONAL_DIRECTIONS": "3",
        "MFA_INT_DIRECTIONAL_STEP": "1e-6",
        "MFA_INT_DIRECTIONAL_TOL": "1e-8",
        "MFA_QR_CERTIFICATE": "0",
        "MFA_QR_CERTIFICATE_PREFIX": "augmented_ls",
        "MFA_QR_PIVOT_THRESHOLD": "1e-12",
        "MFA_BFGS_REL_GRAD_REDUCTION": "1e-8",
        "MFA_BFGS_GRAD_ABS_FLOOR": "1e-10",
        "MFA_BFGS_MAX_UPDATES": "10000",
    }
    return command, environment


def burgers_spec(build_root: Path, output: Path) -> tuple[list[str], dict[str, str]]:
    points = PACKAGE_DIR / "point_sets" / "two_d_burgers"
    command = [
        str(build_root / "burgers" / "examples" / "pde" / "flowdata"),
        "-i", "Burger2D",
        "-f", str(PACKAGE_DIR / "data" / "two_d_burgers"),
        "-v", "30", "-v", "30", "-v", "30",
        "-k", "1", "-q", "3", "-x", "1", "-b", "0",
    ]
    environment = {
        "MFA_LAMBDA_DATA": "1",
        "MFA_LAMBDA_PDE": "0.01",
        "MFA_LAMBDA_IC": "0.1",
        "MFA_LAMBDA_BC": "0.1",
        "MFA_INITIALIZER_LOAD": str(
            PACKAGE_DIR / "checkpoints" / "two_d_burgers" / "stage_a_initializer.txt"
        ),
        "MFA_DATA_ACTIVE_INDEX_FILE": str(points / "data_fit_indices.txt"),
        "MFA_INITIALIZER_ACTIVE_INDEX_FILE": str(points / "initializer_fit_indices.txt"),
        "MFA_PDE_ACTIVE_INDEX_FILE": str(points / "collocation_fit_indices.txt"),
        "MFA_DATA_VALID_INDEX_FILE": str(points / "data_val_indices.txt"),
        "MFA_PDE_VALID_INDEX_FILE": str(points / "collocation_val_indices.txt"),
        "MFA_BURGERS_LAMBDA_INT_U": "0.0015215823165991675",
        "MFA_BURGERS_LAMBDA_INT_V": "0.0021535330926510761",
        "MFA_BURGERS_INT_PARTS_X": "4",
        "MFA_BURGERS_INT_PARTS_Y": "4",
        "MFA_BURGERS_INT_QUADRATURE": "gl4",
        "MFA_BURGERS_INT_FIT_TIME_INDEX_FILE": str(
            points / "integral_balance" / "fit_time_indices.txt"
        ),
        "MFA_BURGERS_INT_VALID_TIME_INDEX_FILE": str(
            points / "integral_balance" / "validation_time_indices.txt"
        ),
        "MFA_BURGERS_INT_DIAGNOSTICS": "1",
        "MFA_BURGERS_INT_CHECK_CLOSURE": "1",
        "MFA_BURGERS_INT_CLOSURE_TOL": "1e-12",
        "MFA_BURGERS_BC_DESIGN_AUDIT": "1",
        "MFA_OPTIMIZER_EPSILON": "1e-8",
        "MFA_OPTIMIZER_EPSILON_REL": "0",
        "MFA_OPTIMIZER_PAST": "0",
        "MFA_OPTIMIZER_DELTA": "0",
        "MFA_MAX_ITERATIONS": "300",
    }
    return command, environment


def cavity_spec(build_root: Path, output: Path) -> tuple[list[str], dict[str, str]]:
    points = PACKAGE_DIR / "point_sets" / "lid_driven_cavity"
    roles = points / "integral_time_roles"
    command = [
        str(build_root / "cavity" / "examples" / "pde" / "flowdata"),
        "-i", "NS2D",
        "-f", str(PACKAGE_DIR / "data" / "lid_driven_cavity"),
        "-v", "20", "-v", "20", "-v", "50",
        "-q", "3", "-x", "1", "-b", "0", "-k", "0",
    ]
    environment = {
        "MFA_LAMBDA_DATA_U": "1",
        "MFA_LAMBDA_DATA_V": "1",
        "MFA_LAMBDA_DATA_P": "0",
        "MFA_LAMBDA_IC_U": "1",
        "MFA_LAMBDA_IC_V": "1",
        "MFA_LAMBDA_IC_P": "0",
        "MFA_LAMBDA_BC_U": "1",
        "MFA_LAMBDA_BC_V": "1",
        "MFA_LAMBDA_BC_P": "0",
        "MFA_LAMBDA_GAUGE": "1",
        "MFA_LAMBDA_DIV": "5.1480131775711293e-7",
        "MFA_LAMBDA_XMOM": "2.1286949198240469e-6",
        "MFA_LAMBDA_YMOM": "2.1651546252130242e-6",
        "MFA_LAMBDA_INT_MASS": "0.0015364163107434725",
        "MFA_LAMBDA_INT_XMOM": "7.6767270897830052e-5",
        "MFA_LAMBDA_INT_YMOM": "7.8300338545488321e-5",
        "MFA_INIT_CHECKPOINT_IN": str(
            PACKAGE_DIR
            / "checkpoints"
            / "lid_driven_cavity"
            / "stage_a"
            / "shared_stage_a_initializer_v1.txt"
        ),
        "MFA_INIT_PRESSURE_MODE": "zero",
        "MFA_FINAL_COEFFICIENT_CHECKPOINT_OUT": str(
            output / "final_coefficients_v1.txt"
        ),
        "MFA_DATA_ACTIVE_INDEX_FILE": str(points / "data_fit_indices.txt"),
        "MFA_INITIALIZER_ACTIVE_INDEX_FILE": str(points / "initializer_fit_indices.txt"),
        "MFA_PDE_ACTIVE_INDEX_FILE": str(points / "collocation_fit_indices.txt"),
        "MFA_DATA_VALID_INDEX_FILE": str(points / "data_val_indices.txt"),
        "MFA_PDE_VALID_INDEX_FILE": str(points / "collocation_val_indices.txt"),
        "MFA_INT_FIT_TIME_INDEX_FILE": str(roles / "integral_time_fit_indices.txt"),
        "MFA_INT_VALID_TIME_INDEX_FILE": str(
            roles / "integral_time_validation_indices.txt"
        ),
        "MFA_INT_PARTS_X": "4",
        "MFA_INT_PARTS_Y": "4",
        "MFA_INT_QUADRATURE": "gl4",
        "MFA_INT_DIAGNOSTICS": "1",
        "MFA_INT_CHECK_CLOSURE": "1",
        "MFA_INT_CLOSURE_TOL": "1e-12",
        "MFA_NS_NU": "0.001",
        "MFA_NS_PHYSICS_T_MIN": "0.01",
        "MFA_NS_TIME_DEPENDENT_BC": "1",
        "MFA_REQUIRE_ZERO_PRESSURE_BC": "1",
        "MFA_FINAL_STATE_HANDSHAKE": "1",
        "MFA_OPTIMIZER_EPSILON": "1e-9",
        "MFA_OPTIMIZER_EPSILON_REL": "0",
        "MFA_OPTIMIZER_PAST": "0",
        "MFA_OPTIMIZER_DELTA": "0",
        "MFA_MAX_ITERATIONS": "2400",
    }
    return command, environment


SPECS: dict[str, Callable[[Path, Path], tuple[list[str], dict[str, str]]]] = {
    "one_d": one_d_spec,
    "burgers": burgers_spec,
    "cavity": cavity_spec,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(SPECS))
    parser.add_argument(
        "--build-root",
        type=Path,
        default=REPO_DIR / ".build",
        help="build directory created by build_and_test.sh",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the optimizer; without this flag, print the resolved contract",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_root = args.build_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else PACKAGE_DIR / "work" / f"{args.case}_selected"
    )
    command, overrides = SPECS[args.case](build_root, output)
    record = {
        "case": args.case,
        "command": command,
        "environment_overrides": dict(sorted(overrides.items())),
        "output": str(output),
    }

    if not args.execute:
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    required = [Path(command[0])]
    required.extend(
        Path(value)
        for key, value in overrides.items()
        if key.endswith("_FILE") or key.endswith("_PATH") or key.endswith("_LOAD") or key.endswith("_IN")
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("Missing required files:\n" + "\n".join(missing), file=sys.stderr)
        return 1
    if output.exists():
        print(f"Refusing to reuse output directory: {output}", file=sys.stderr)
        return 1

    output.mkdir(parents=True)
    (output / "run_contract.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MFA_")
    }
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMPI_MCA_btl": "self",
        }
    )
    environment.update(overrides)

    print("Running:", " ".join(command))
    print("Output:", output)
    with (output / "stdout.log").open("wb") as stdout, (
        output / "stderr.log"
    ).open("wb") as stderr:
        completed = subprocess.run(
            command,
            cwd=output,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode:
        print(
            f"Solver exited {completed.returncode}; inspect {output / 'stderr.log'}.",
            file=sys.stderr,
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
