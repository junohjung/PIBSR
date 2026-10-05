#!/usr/bin/env python3
"""Serialize the exact interval forcing used by the Burgers observations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

try:
    import numpy as np
except ImportError:
    np = None


HERE = Path(__file__).resolve().parent
LX = LY = 2.0
NX = NY = 40
DT_OUT = 0.01
N_INTERVALS = 200
MODES = ((1, 1), (2, 1), (1, 2), (2, 2))
RHO = 0.85
SEED = 1234
T_START = 0.10
T_END = 1.50
RAMP_FRACTION = 0.15
AMPLITUDE_U = 0.2
AMPLITUDE_V = 0.16

EXPECTED_HASHES = {
    "forcing_basis_coarse_rms": "bb3e5f0a2bb962bd3e7013ed8728111ba30ba145638c950673f05bdfdcd9f1fb",
    "forcing_combo_coarse_rms_u": "12a04b8a44e3ccf3315af056ffe7fde3e86360b88d9539d96add23f8846bda10",
    "forcing_combo_coarse_rms_v": "a720e91e83fb8800da4c4773b3f6b1fa434928469086f5b15cfc9bc2d4d71e11",
    "forcing_domain_integral_uv": "e56706e4ba31c0923a4b9fab94f14db8c567882cb9bcb88b69f127bbd5620417",
    "forcing_envelope": "8810392652c30bd5f3921ba5756a3afdff0b5368f14d8717deb4a6b8b1a36c17",
    "forcing_modal_coeff_u": "5ad35be3e04fdcfe5539a1bcd89bdff03c74a497506447ce6dfa56e3b6dba1c0",
    "forcing_modal_coeff_v": "a893dabee416d1c6436ea425f42dc500e0a8e0dbb26b8eb677eb381a035ca01b",
    "forcing_modes": "9f8b21d68bd14874c17a1d984a44ba93f5104e63dfed7941dc152d5ccc00b941",
    "forcing_raw_coeff_u": "178a3fcff8047fd1801b28690f058e91c47518e9aa9aa989784e6ea4c006008d",
    "forcing_raw_coeff_v": "025ca86e27e5821f0f1f982f362fa011993cfd24688f698490cdd2f4b49b6f98",
}


def rms_interior(field: np.ndarray) -> float:
    interior = field[1:-1, 1:-1]
    return float(np.sqrt(np.mean(interior * interior, dtype=np.float64)))


def envelope(time: float) -> float:
    if time <= T_START or time >= T_END:
        return 0.0
    ramp = RAMP_FRACTION * (T_END - T_START)
    if time < T_START + ramp:
        s = (time - T_START) / ramp
        return 0.5 * (1.0 - math.cos(math.pi * s))
    if time > T_END - ramp:
        s = (T_END - time) / ramp
        return 0.5 * (1.0 - math.cos(math.pi * s))
    return 1.0


def stable_mode_integral(modal: np.ndarray, integrals: np.ndarray) -> np.ndarray:
    """Evaluate this four-mode dot product with stable signed-zero semantics."""
    if integrals.shape != (4,) or np.any(integrals[1:] != 0.0):
        raise ValueError("The frozen forcing modes no longer have the expected integrals")
    result = modal[:, 0] * integrals[0]
    zero = result == 0.0
    negative_zero = zero & np.all(np.signbit(modal), axis=1)
    result[zero] = 0.0
    result[negative_zero] = -0.0
    return result


def build() -> dict[str, np.ndarray]:
    x = np.linspace(0.0, LX, NX + 1, dtype=np.float64)
    y = np.linspace(0.0, LY, NY + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    basis: list[np.ndarray] = []
    basis_rms: list[float] = []
    for mx, my in MODES:
        field = np.sin(mx * math.pi * xx / LX) * np.sin(my * math.pi * yy / LY)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
        norm = rms_interior(field)
        basis_rms.append(norm)
        basis.append(field / norm)

    rng = np.random.default_rng(SEED)
    innovation = math.sqrt(1.0 - RHO * RHO)
    coeff_u = rng.standard_normal(len(MODES))
    coeff_v = rng.standard_normal(len(MODES))
    raw_u = np.empty((N_INTERVALS, len(MODES)), dtype=np.float64)
    raw_v = np.empty_like(raw_u)
    combo_u = np.empty(N_INTERVALS, dtype=np.float64)
    combo_v = np.empty(N_INTERVALS, dtype=np.float64)
    envelopes = np.empty(N_INTERVALS, dtype=np.float64)

    for interval in range(N_INTERVALS):
        envelopes[interval] = envelope((interval + 0.5) * DT_OUT)
        coeff_u = RHO * coeff_u + innovation * rng.standard_normal(len(MODES))
        coeff_v = RHO * coeff_v + innovation * rng.standard_normal(len(MODES))
        raw_u[interval] = coeff_u
        raw_v[interval] = coeff_v
        combo_u[interval] = rms_interior(sum(c * b for c, b in zip(coeff_u, basis)))
        combo_v[interval] = rms_interior(sum(c * b for c, b in zip(coeff_v, basis)))

    norms = np.asarray(basis_rms, dtype=np.float64)
    modal_u = raw_u / norms / combo_u[:, None] * (AMPLITUDE_U * envelopes[:, None])
    modal_v = raw_v / norms / combo_v[:, None] * (AMPLITUDE_V * envelopes[:, None])
    mode_integrals = []
    for mx, my in MODES:
        integral_x = LX * (1.0 - (-1.0) ** mx) / (mx * math.pi)
        integral_y = LY * (1.0 - (-1.0) ** my) / (my * math.pi)
        mode_integrals.append(integral_x * integral_y)
    mode_integrals = np.asarray(mode_integrals, dtype=np.float64)
    return {
        "forcing_modes": np.asarray(MODES, dtype=np.int16),
        "forcing_basis_coarse_rms": norms,
        "forcing_raw_coeff_u": raw_u,
        "forcing_raw_coeff_v": raw_v,
        "forcing_combo_coarse_rms_u": combo_u,
        "forcing_combo_coarse_rms_v": combo_v,
        "forcing_envelope": envelopes,
        "forcing_modal_coeff_u": modal_u,
        "forcing_modal_coeff_v": modal_v,
        "forcing_domain_integral_uv": np.column_stack(
            (
                stable_mode_integral(modal_u, mode_integrals),
                stable_mode_integral(modal_v, mode_integrals),
            )
        ),
    }


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape), "order": "C"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(header + b"\n")
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def verify_arrays(arrays: dict[str, np.ndarray]) -> None:
    actual = {name: array_hash(array) for name, array in arrays.items()}
    if actual != EXPECTED_HASHES:
        mismatches = {
            name: {"expected": EXPECTED_HASHES.get(name), "actual": actual.get(name)}
            for name in sorted(set(actual) | set(EXPECTED_HASHES))
            if actual.get(name) != EXPECTED_HASHES.get(name)
        }
        raise RuntimeError(f"Forcing contract mismatch: {mismatches}")


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, buffer.getvalue())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(npz_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": "burgers_exact_coarse_defined_forcing_realization",
        "contract": "continuous forcing normalized only on the original 39x39 interior",
        "published_generator_sha256": "aca87d645aaa44577083437e617e36dc42d97a7d500e513b05da2d7d940e5fc3",
        "parameters": {
            "domain": [0.0, LX, 0.0, LY],
            "coarse_cells": [NX, NY],
            "interval_count": N_INTERVALS,
            "dt_out": DT_OUT,
            "modes": [list(mode) for mode in MODES],
            "rho": RHO,
            "seed": SEED,
            "active_times": [T_START, T_END],
            "ramp_fraction": RAMP_FRACTION,
            "rhs_rms_amplitude": [AMPLITUDE_U, AMPLITUDE_V],
        },
        "array_contract_sha256": EXPECTED_HASHES,
        "npz": {"path": npz_path.name, "sha256": sha256_file(npz_path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "forcing_realization.npz")
    parser.add_argument(
        "--manifest", type=Path, default=HERE / "forcing_realization_manifest.json"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Build and hash arrays without writing.")
    mode.add_argument("--check", action="store_true", help="Check the packaged NPZ and manifest.")
    args = parser.parse_args()

    if np is None:
        raise SystemExit("NumPy is required to build or check the forcing realization")

    arrays = build()
    verify_arrays(arrays)
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "array_count": len(arrays)}, indent=2))
        return
    if args.check:
        if not args.output.is_file() or not args.manifest.is_file():
            raise FileNotFoundError("Packaged forcing NPZ or manifest is missing")
        with np.load(args.output, allow_pickle=False) as archive:
            observed = {name: archive[name] for name in archive.files}
        verify_arrays(observed)
        frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
        if frozen["array_contract_sha256"] != EXPECTED_HASHES:
            raise RuntimeError("Manifest array hashes differ from the frozen contract")
        if frozen["npz"]["sha256"] != sha256_file(args.output):
            raise RuntimeError("Manifest NPZ hash mismatch")
        print(json.dumps({"status": "PASS", "npz_sha256": sha256_file(args.output)}, indent=2))
        return

    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("Output exists; use --check instead of overwriting it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_npz(args.output, arrays)
    args.manifest.write_text(json.dumps(manifest(args.output), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "CREATED", "npz_sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
