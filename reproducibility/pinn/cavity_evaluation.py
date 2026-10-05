"""Float64 conservative GL8 balance evaluation for cavity PINNs."""

from __future__ import annotations
import hashlib
import time
from pathlib import Path
from typing import Any
import numpy as np
from numpy.polynomial.legendre import leggauss
import torch

NU = 1e-3
COMPONENTS = ("mass", "x_momentum", "y_momentum")
SEALED_TIMES = Path(__file__).resolve().parent / "cavity/evaluation_times.txt"
UPSTREAM_SHA256 = {SEALED_TIMES: hashlib.sha256(SEALED_TIMES.read_bytes()).hexdigest()}


def _mapped_gl(lower: float, upper: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(order)
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    return (midpoint + half_width * nodes, half_width * weights)


def make_q4_quadrature(order: int = 8) -> tuple[np.ndarray, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    spatial_parts: list[np.ndarray] = []
    specifications: list[dict[str, Any]] = []
    cursor = 0

    def add(points: np.ndarray) -> slice:
        nonlocal cursor
        start = cursor
        spatial_parts.append(points)
        cursor += points.shape[0]
        return slice(start, cursor)

    for block_y in range(4):
        for block_x in range(4):
            xl, xu = (float(edges[block_x]), float(edges[block_x + 1]))
            yl, yu = (float(edges[block_y]), float(edges[block_y + 1]))
            yn, yw = _mapped_gl(yl, yu, order)
            xn, xw = _mapped_gl(xl, xu, order)
            right = add(np.column_stack((np.full(order, xu), yn)))
            left = add(np.column_stack((np.full(order, xl), yn)))
            top = add(np.column_stack((xn, np.full(order, yu))))
            bottom = add(np.column_stack((xn, np.full(order, yl))))
            xx, yy = np.meshgrid(xn, yn, indexing="xy")
            volume = add(np.column_stack((xx.ravel(), yy.ravel())))
            specifications.append(
                {
                    "block_x": block_x,
                    "block_y": block_y,
                    "x_lower": xl,
                    "x_upper": xu,
                    "y_lower": yl,
                    "y_upper": yu,
                    "right": right,
                    "left": left,
                    "top": top,
                    "bottom": bottom,
                    "volume": volume,
                    "x_weights": xw,
                    "y_weights": yw,
                    "volume_weights": (yw[:, None] * xw[None, :]).ravel(),
                }
            )
    spatial = np.vstack(spatial_parts).astype(np.float64)
    if spatial.shape != (16 * (4 * order + order**2), 2) or len(specifications) != 16:
        raise RuntimeError("Canonical q=4 quadrature construction changed")
    return (spatial, specifications)


def evaluate_network_jet(
    model: torch.nn.Module, spatial: np.ndarray, physical_time: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz_np = np.column_stack((spatial, np.full(spatial.shape[0], physical_time, dtype=np.float64)))
    xyz = torch.tensor(xyz_np, dtype=torch.float64, requires_grad=True)
    output = model(xyz)
    u, v, p = (output[:, 0], output[:, 1], output[:, 2])
    grad_u = torch.autograd.grad(u.sum(), xyz, retain_graph=True, create_graph=False)[0]
    grad_v = torch.autograd.grad(v.sum(), xyz, retain_graph=False, create_graph=False)[0]
    return (
        output.detach().cpu().numpy(),
        grad_u.detach().cpu().numpy(),
        grad_v.detach().cpu().numpy(),
    )


def face_flux(
    output: np.ndarray,
    grad_u: np.ndarray,
    grad_v: np.ndarray,
    selection: slice,
    weights: np.ndarray,
    normal_axis: int,
) -> np.ndarray:
    values = output[selection]
    gu = grad_u[selection]
    gv = grad_v[selection]
    u, v, pressure = (values[:, 0], values[:, 1], values[:, 2])
    normal_velocity = u if normal_axis == 0 else v
    mass = np.sum(weights * normal_velocity)
    x_flux = np.sum(
        weights
        * (u * normal_velocity + (pressure if normal_axis == 0 else 0.0) - NU * gu[:, normal_axis])
    )
    y_flux = np.sum(
        weights
        * (v * normal_velocity + (pressure if normal_axis == 1 else 0.0) - NU * gv[:, normal_axis])
    )
    return np.asarray((mass, x_flux, y_flux), dtype=np.float64)


def evaluate_gl8(
    model: torch.nn.Module, times: np.ndarray, seed_root: Path, order: int = 8
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = model.to(dtype=torch.float64, device=torch.device("cpu"))
    model.eval()
    spatial, specifications = make_q4_quadrature(order)
    block_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    maximum_closure = {component: 0.0 for component in COMPONENTS}
    start = time.perf_counter()
    for time_index, physical_time in enumerate(times):
        output, grad_u, grad_v = evaluate_network_jet(model, spatial, float(physical_time))
        raw_blocks = np.empty((4, 4, 3), dtype=np.float64)
        for specification in specifications:
            bx = int(specification["block_x"])
            by = int(specification["block_y"])
            surface = (
                face_flux(
                    output, grad_u, grad_v, specification["right"], specification["y_weights"], 0
                )
                - face_flux(
                    output, grad_u, grad_v, specification["left"], specification["y_weights"], 0
                )
                + face_flux(
                    output, grad_u, grad_v, specification["top"], specification["x_weights"], 1
                )
                - face_flux(
                    output, grad_u, grad_v, specification["bottom"], specification["x_weights"], 1
                )
            )
            volume_selection = specification["volume"]
            transient = np.asarray(
                (
                    0.0,
                    np.sum(specification["volume_weights"] * grad_u[volume_selection, 2]),
                    np.sum(specification["volume_weights"] * grad_v[volume_selection, 2]),
                ),
                dtype=np.float64,
            )
            raw = surface + transient
            if not np.all(np.isfinite(raw)):
                raise RuntimeError("PINN GL8 evaluation produced a nonfinite balance")
            raw_blocks[by, bx] = raw
            normalized = np.abs(raw) / 0.25
            block_rows.append(
                {
                    "time_index": time_index,
                    "time": format(float(physical_time), ".17e"),
                    "block_x": bx,
                    "block_y": by,
                    "x_lower": format(specification["x_lower"], ".17e"),
                    "x_upper": format(specification["x_upper"], ".17e"),
                    "y_lower": format(specification["y_lower"], ".17e"),
                    "y_upper": format(specification["y_upper"], ".17e"),
                    "mass": format(raw[0], ".17e"),
                    "x_momentum": format(raw[1], ".17e"),
                    "y_momentum": format(raw[2], ".17e"),
                    "normalized_abs_mass": format(normalized[0], ".17e"),
                    "normalized_abs_x_momentum": format(normalized[1], ".17e"),
                    "normalized_abs_y_momentum": format(normalized[2], ".17e"),
                }
            )
        q1 = np.sum(raw_blocks, axis=(0, 1))
        q4_rms = np.sqrt(np.mean((raw_blocks / 0.25) ** 2, axis=(0, 1)))
        curve_rows.append(
            {
                "time_index": time_index,
                "time": format(float(physical_time), ".17e"),
                "q4_spatial_rms_mass": format(q4_rms[0], ".17e"),
                "q4_spatial_rms_x_momentum": format(q4_rms[1], ".17e"),
                "q4_spatial_rms_y_momentum": format(q4_rms[2], ".17e"),
                "signed_q1_mass": format(q1[0], ".17e"),
                "signed_q1_x_momentum": format(q1[1], ".17e"),
                "signed_q1_y_momentum": format(q1[2], ".17e"),
                "evaluation_pipeline": "torch_float64_autograd_canonical_gl8_q4_aggregate_q1",
            }
        )
        reconstructed = np.asarray(
            [sum((float(row[component]) for row in block_rows[-16:])) for component in COMPONENTS],
            dtype=np.float64,
        )
        for component_index, component in enumerate(COMPONENTS):
            maximum_closure[component] = max(
                maximum_closure[component],
                abs(float(q1[component_index]) - reconstructed[component_index]),
            )
        if (time_index + 1) % 50 == 0:
            print(f"[seed evaluation {time_index + 1:3d}/200] t={physical_time:.6f}", flush=True)
    evaluation_seconds = time.perf_counter() - start
    if max(maximum_closure.values()) > 2e-15:
        raise RuntimeError(f"Serialized q4-to-q1 closure failed: {maximum_closure}")
    audit = {
        "schema": "pimfa.ramped_cavity.pinn.gl8_evaluation_audit.v1",
        "status": "PASS",
        "sealed_time_count": len(times),
        "sealed_time_sha256": UPSTREAM_SHA256[SEALED_TIMES],
        "canonical_q4_block_count_per_time": 16,
        "gl_order_per_spatial_dimension_per_block": 8,
        "q4_normalization": "abs(raw balance)/sqrt(block area); sqrt(area)=0.25",
        "q4_history_reduction": "RMS across the 16 canonical q=4 blocks",
        "q1_definition": "signed sum of the 16 canonical q=4 raw balances",
        "maximum_serialized_q4_to_q1_closure": maximum_closure,
        "autodiff_dtype": "torch.float64 after exact float32 checkpoint promotion",
        "viscosity": NU,
        "reference_or_validation_values_opened": False,
        "evaluation_seconds": evaluation_seconds,
    }
    audit["gl_order_per_spatial_dimension_per_block"] = order
    audit["schema"] = "pibsr.cavity.pinn.balance_evaluation.v1"
    for row in curve_rows:
        row["evaluation_pipeline"] = f"torch_float64_autograd_canonical_gl{order}_q4_aggregate_q1"
    return (block_rows, curve_rows, audit)
