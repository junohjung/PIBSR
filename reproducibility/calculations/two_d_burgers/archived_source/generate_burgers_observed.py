#!/usr/bin/env python3
"""Generate the exact coarse observed trajectory used by the 2-D Burgers case.

The solver is fourth-order centered in space, RK4 in time, and imposes
homogeneous Dirichlet values after every RK stage.  The smooth discrepancy
forcing is piecewise constant over each 0.01 output interval.  With the
defaults below, the text snapshots reproduce the files in
``data/2d_burgers/observed`` byte for byte.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CENTERS_U_FRAC = [(0.325, 0.425), (0.525, 0.500)]
SIGMAS_U_FRAC = [0.08, 0.10]
CENTERS_V_FRAC = [(0.375, 0.600), (0.600, 0.400)]
SIGMAS_V_FRAC = [0.09, 0.09]
FORCING_MODES = [(1, 1), (2, 1), (1, 2), (2, 2)]


@dataclass(frozen=True)
class Params:
    Lx: float = 2.0
    Ly: float = 2.0
    nu: float = 0.01
    nx: int = 40
    ny: int = 40
    dt: float = 5.0e-4
    dt_out: float = 1.0e-2
    t_end: float = 2.0
    amp_u: float = 1.0
    amp_v: float = 0.8
    ic_scale_n: int = 1025
    force_delta_u: float = 2.0e-3
    force_delta_v: float = 2.0e-3
    force_num_basis: int = 4
    force_time_smooth: float = 0.85
    force_t_start: float = 0.10
    force_t_end: float = 1.50
    force_ramp_frac: float = 0.15
    force_seed: int = 1234
    obs_noise_std: float = 0.0
    perturb_first_frame: bool = False
    obs_noise_seed: int = 5678
    use_cfl: bool = False
    cfl: float = 0.35
    prefix: str = "uv_times"
    float_fmt: str = "%.8e"


def make_grid(Lx: float, Ly: float, nx: int, ny: int):
    if nx < 4 or ny < 4:
        raise ValueError("nx and ny must be at least 4")
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y, float(x[1] - x[0]), float(y[1] - y[0])


def apply_dirichlet_zero(u: np.ndarray, v: np.ndarray) -> None:
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
    v[0, :] = v[-1, :] = v[:, 0] = v[:, -1] = 0.0


def gaussian_sum(X: np.ndarray, Y: np.ndarray, centers, sigmas) -> np.ndarray:
    result = np.zeros_like(X, dtype=np.float64)
    for (cx, cy), sigma in zip(centers, sigmas):
        result += np.exp(
            -((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * float(sigma) ** 2)
        )
    return result


def compute_common_ic_scaling(P: Params):
    grid = np.linspace(0.0, P.Lx, max(17, int(P.ic_scale_n)))
    X, Y = np.meshgrid(grid, grid)
    length = min(P.Lx, P.Ly)
    centers_u = [(fx * P.Lx, fy * P.Ly) for fx, fy in CENTERS_U_FRAC]
    centers_v = [(fx * P.Lx, fy * P.Ly) for fx, fy in CENTERS_V_FRAC]
    u = gaussian_sum(X, Y, centers_u, [s * length for s in SIGMAS_U_FRAC])
    v = gaussian_sum(X, Y, centers_v, [s * length for s in SIGMAS_V_FRAC])
    apply_dirichlet_zero(u, v)
    return P.amp_u / np.max(np.abs(u)), P.amp_v / np.max(np.abs(v))


def make_initial_condition(P: Params, X, Y, Au: float, Av: float):
    length = min(P.Lx, P.Ly)
    centers_u = [(fx * P.Lx, fy * P.Ly) for fx, fy in CENTERS_U_FRAC]
    centers_v = [(fx * P.Lx, fy * P.Ly) for fx, fy in CENTERS_V_FRAC]
    u = Au * gaussian_sum(
        X, Y, centers_u, [s * length for s in SIGMAS_U_FRAC]
    )
    v = Av * gaussian_sum(
        X, Y, centers_v, [s * length for s in SIGMAS_V_FRAC]
    )
    apply_dirichlet_zero(u, v)
    return u.astype(np.float64), v.astype(np.float64)


class HighOrderBurgers2D:
    def __init__(self, nx: int, ny: int, dx: float, dy: float, nu: float):
        self.dx = float(dx)
        self.dy = float(dy)
        self.nu = float(nu)
        shape = (ny + 1, nx + 1)
        for name in (
            "ux",
            "uy",
            "vx",
            "vy",
            "lapu",
            "lapv",
            "k1u",
            "k1v",
            "k2u",
            "k2v",
            "k3u",
            "k3v",
            "k4u",
            "k4v",
            "ut",
            "vt",
        ):
            setattr(self, name, np.zeros(shape, dtype=np.float64))

    def _gradient(self, field, fx, fy) -> None:
        fx.fill(0.0)
        fy.fill(0.0)
        fx[:, 2:-2] = (
            -field[:, 4:]
            + 8.0 * field[:, 3:-1]
            - 8.0 * field[:, 1:-3]
            + field[:, :-4]
        ) / (12.0 * self.dx)
        fx[:, 1] = (field[:, 2] - field[:, 0]) / (2.0 * self.dx)
        fx[:, -2] = (field[:, -1] - field[:, -3]) / (2.0 * self.dx)
        fy[2:-2, :] = (
            -field[4:, :]
            + 8.0 * field[3:-1, :]
            - 8.0 * field[1:-3, :]
            + field[:-4, :]
        ) / (12.0 * self.dy)
        fy[1, :] = (field[2, :] - field[0, :]) / (2.0 * self.dy)
        fy[-2, :] = (field[-1, :] - field[-3, :]) / (2.0 * self.dy)

    def _laplacian(self, field, lap) -> None:
        lap.fill(0.0)
        lap[:, 2:-2] += (
            -field[:, 4:]
            + 16.0 * field[:, 3:-1]
            - 30.0 * field[:, 2:-2]
            + 16.0 * field[:, 1:-3]
            - field[:, :-4]
        ) / (12.0 * self.dx**2)
        lap[:, 1] += (
            field[:, 2] - 2.0 * field[:, 1] + field[:, 0]
        ) / self.dx**2
        lap[:, -2] += (
            field[:, -1] - 2.0 * field[:, -2] + field[:, -3]
        ) / self.dx**2
        lap[2:-2, :] += (
            -field[4:, :]
            + 16.0 * field[3:-1, :]
            - 30.0 * field[2:-2, :]
            + 16.0 * field[1:-3, :]
            - field[:-4, :]
        ) / (12.0 * self.dy**2)
        lap[1, :] += (
            field[2, :] - 2.0 * field[1, :] + field[0, :]
        ) / self.dy**2
        lap[-2, :] += (
            field[-1, :] - 2.0 * field[-2, :] + field[-3, :]
        ) / self.dy**2

    def rhs(self, u, v, fu, fv, du, dv) -> None:
        self._gradient(u, self.ux, self.uy)
        self._gradient(v, self.vx, self.vy)
        self._laplacian(u, self.lapu)
        self._laplacian(v, self.lapv)
        du[:] = -(u * self.ux + v * self.uy) + self.nu * self.lapu + fu
        dv[:] = -(u * self.vx + v * self.vy) + self.nu * self.lapv + fv
        apply_dirichlet_zero(du, dv)

    def step_rk4(self, u, v, fu, fv, dt: float) -> None:
        self.rhs(u, v, fu, fv, self.k1u, self.k1v)
        self.ut[:] = u + 0.5 * dt * self.k1u
        self.vt[:] = v + 0.5 * dt * self.k1v
        apply_dirichlet_zero(self.ut, self.vt)
        self.rhs(self.ut, self.vt, fu, fv, self.k2u, self.k2v)
        self.ut[:] = u + 0.5 * dt * self.k2u
        self.vt[:] = v + 0.5 * dt * self.k2v
        apply_dirichlet_zero(self.ut, self.vt)
        self.rhs(self.ut, self.vt, fu, fv, self.k3u, self.k3v)
        self.ut[:] = u + dt * self.k3u
        self.vt[:] = v + dt * self.k3v
        apply_dirichlet_zero(self.ut, self.vt)
        self.rhs(self.ut, self.vt, fu, fv, self.k4u, self.k4v)
        u[:] += (dt / 6.0) * (
            self.k1u + 2.0 * self.k2u + 2.0 * self.k3u + self.k4u
        )
        v[:] += (dt / 6.0) * (
            self.k1v + 2.0 * self.k2v + 2.0 * self.k3v + self.k4v
        )
        apply_dirichlet_zero(u, v)


def rms_interior(field: np.ndarray) -> float:
    interior = field[1:-1, 1:-1]
    return float(np.sqrt(np.mean(interior * interior)))


def raised_cosine_envelope(
    time_midpoint: float, start: float, end: float, ramp_fraction: float
) -> float:
    if time_midpoint <= start or time_midpoint >= end:
        return 0.0
    ramp = max(min(ramp_fraction, 0.5), 0.0) * (end - start)
    if ramp <= 0.0:
        return 1.0
    if time_midpoint < start + ramp:
        s = (time_midpoint - start) / ramp
        return 0.5 * (1.0 - math.cos(math.pi * s))
    if time_midpoint > end - ramp:
        s = (end - time_midpoint) / ramp
        return 0.5 * (1.0 - math.cos(math.pi * s))
    return 1.0


def make_interval_forcing_sequence(P: Params, X, Y):
    basis = []
    for mx, my in FORCING_MODES[: P.force_num_basis]:
        field = np.sin(mx * math.pi * X / P.Lx) * np.sin(
            my * math.pi * Y / P.Ly
        )
        apply_dirichlet_zero(field, field)
        field /= rms_interior(field)
        basis.append(field)

    rng = np.random.default_rng(P.force_seed)
    rho = P.force_time_smooth
    innovation_scale = math.sqrt(1.0 - rho * rho)
    coeff_u = rng.standard_normal(P.force_num_basis)
    coeff_v = rng.standard_normal(P.force_num_basis)
    amplitude_u = P.force_delta_u * P.amp_u / P.dt_out
    amplitude_v = P.force_delta_v * P.amp_v / P.dt_out
    fu_list, fv_list, envelopes = [], [], []
    for frame in range(int(round(P.t_end / P.dt_out))):
        midpoint = (frame + 0.5) * P.dt_out
        envelope = raised_cosine_envelope(
            midpoint, P.force_t_start, P.force_t_end, P.force_ramp_frac
        )
        coeff_u = (
            rho * coeff_u
            + innovation_scale * rng.standard_normal(P.force_num_basis)
        )
        coeff_v = (
            rho * coeff_v
            + innovation_scale * rng.standard_normal(P.force_num_basis)
        )
        fu = sum(coeff * field for coeff, field in zip(coeff_u, basis))
        fv = sum(coeff * field for coeff, field in zip(coeff_v, basis))
        fu = fu / rms_interior(fu) * amplitude_u * envelope
        fv = fv / rms_interior(fv) * amplitude_v * envelope
        apply_dirichlet_zero(fu, fv)
        fu_list.append(fu)
        fv_list.append(fv)
        envelopes.append(envelope)
    return fu_list, fv_list, np.asarray(envelopes)


def maybe_cfl_dt(P: Params, dx, dy, u, v, remaining: float) -> float:
    if not P.use_cfl:
        return float(remaining)
    max_speed = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))), 1e-12)
    dt_adv = P.cfl * min(dx, dy) / max_speed
    dt_diff = 0.25 * P.cfl * min(dx, dy) ** 2 / max(P.nu, 1e-12)
    return min(float(remaining), dt_adv, dt_diff)


def write_snapshot(path: Path, X, Y, u, v, fmt: str) -> None:
    np.savetxt(
        path,
        np.column_stack((X.ravel(), Y.ravel(), u.ravel(), v.ravel())),
        fmt=fmt,
    )


def generate(P: Params, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, X, Y, dx, dy = make_grid(P.Lx, P.Ly, P.nx, P.ny)
    Au, Av = compute_common_ic_scaling(P)
    u, v = make_initial_condition(P, X, Y, Au, Av)
    solver = HighOrderBurgers2D(P.nx, P.ny, dx, dy, P.nu)
    fu_list, fv_list, _ = make_interval_forcing_sequence(P, X, Y)
    n_out = int(round(P.t_end / P.dt_out))
    times = np.linspace(0.0, P.t_end, n_out + 1)
    write_snapshot(output_dir / f"{P.prefix}_0000.txt", X, Y, u, v, P.float_fmt)
    time = 0.0
    for frame in range(1, n_out + 1):
        target = float(times[frame])
        while time < target - 1e-15:
            dt = maybe_cfl_dt(P, dx, dy, u, v, min(P.dt, target - time))
            solver.step_rk4(u, v, fu_list[frame - 1], fv_list[frame - 1], dt)
            time += dt
        write_snapshot(
            output_dir / f"{P.prefix}_{frame:04d}.txt",
            X,
            Y,
            u,
            v,
            P.float_fmt,
        )
    (output_dir / "times.csv").write_text(
        "frame,time\n"
        + "\n".join(f"{k},{value:.16e}" for k, value in enumerate(times))
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(Params(), args.output_dir)
    print(f"Wrote exact coarse observed trajectory to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
