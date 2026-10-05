"""Neural representation and diagnostic operators for checkpoint evaluation."""

from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

GEN_CENTERS_U_FRAC = ((0.325, 0.425), (0.525, 0.5))
GEN_SIGMAS_U_FRAC = (0.08, 0.1)
GEN_CENTERS_V_FRAC = ((0.375, 0.6), (0.6, 0.4))
GEN_SIGMAS_V_FRAC = (0.09, 0.09)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trapz_compat(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return fn(y, x, axis=axis)


def trapz2d(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Integrate a field of shape (..., ny, nx) over x and y using tensor-product trapz."""
    return trapz_compat(trapz_compat(field, x, axis=-1), y, axis=-1)


class MLP(nn.Module):

    def __init__(
        self,
        layers: Sequence[int],
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        ic_Au: Optional[float] = None,
        ic_Av: Optional[float] = None,
        hard_icbc: bool = False,
        hard_time_beta: float = 4.0,
    ):
        super().__init__()
        if len(layers) < 3:
            raise ValueError("Need at least input, one hidden, and output layers.")
        (self.x_min, self.x_max), (self.y_min, self.y_max), (self.t_min, self.t_max) = bounds
        self.hard_icbc = bool(hard_icbc)
        self.hard_time_beta = float(hard_time_beta)
        self.Lx = float(self.x_max - self.x_min)
        self.Ly = float(self.y_max - self.y_min)
        self.Lmin = float(min(self.Lx, self.Ly))
        self.ic_Au = None if ic_Au is None else float(ic_Au)
        self.ic_Av = None if ic_Av is None else float(ic_Av)
        mods: List[nn.Module] = []
        for i in range(len(layers) - 2):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            mods.append(nn.Tanh())
        mods.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*mods)
        self._init_weights()
        if self.hard_icbc and (self.ic_Au is None or self.ic_Av is None):
            raise ValueError("hard_icbc=True requires ic_Au and ic_Av.")

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)

    def _normalize(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        t = xyz[:, 2:3]
        x_hat = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_hat = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_hat = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        return torch.cat([x_hat, y_hat, t_hat], dim=1)

    def _gaussian_sum(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        centers_frac: Sequence[Tuple[float, float]],
        sigmas_frac: Sequence[float],
        amplitude: float,
    ) -> torch.Tensor:
        out = torch.zeros_like(x)
        for (fx, fy), sf in zip(centers_frac, sigmas_frac):
            cx = self.x_min + fx * self.Lx
            cy = self.y_min + fy * self.Ly
            sigma = sf * self.Lmin
            out = out + torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma * sigma))
        out = amplitude * out
        tol = 1e-07 * max(self.Lx, self.Ly, 1.0)
        on_boundary = (
            (torch.abs(x - self.x_min) <= tol)
            | (torch.abs(x - self.x_max) <= tol)
            | (torch.abs(y - self.y_min) <= tol)
            | (torch.abs(y - self.y_max) <= tol)
        )
        return torch.where(on_boundary, torch.zeros_like(out), out)

    def _analytic_ic(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        u0 = self._gaussian_sum(x, y, GEN_CENTERS_U_FRAC, GEN_SIGMAS_U_FRAC, self.ic_Au)
        v0 = self._gaussian_sum(x, y, GEN_CENTERS_V_FRAC, GEN_SIGMAS_V_FRAC, self.ic_Av)
        return (u0, v0)

    def _time_envelope(self, xyz: torch.Tensor) -> torch.Tensor:
        t = xyz[:, 2:3]
        t01 = (t - self.t_min) / (self.t_max - self.t_min + 1e-12)
        t01 = torch.clamp(t01, min=0.0)
        return 1.0 - torch.exp(-self.hard_time_beta * t01)

    def _space_envelope(self, xyz: torch.Tensor) -> torch.Tensor:
        """C2 polynomial gate that vanishes on all four spatial walls."""
        x01 = (xyz[:, 0:1] - self.x_min) / (self.x_max - self.x_min)
        y01 = (xyz[:, 1:2] - self.y_min) / (self.y_max - self.y_min)
        return 16.0 * x01 * (1.0 - x01) * y01 * (1.0 - y01)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        raw = self.net(self._normalize(xyz))
        if not self.hard_icbc:
            return raw
        u0, v0 = self._analytic_ic(xyz)
        gate = self._time_envelope(xyz) * self._space_envelope(xyz)
        u = u0 + gate * raw[:, 0:1]
        v = v0 + gate * raw[:, 1:2]
        return torch.cat([u, v], dim=1)


def to_tensor(a: np.ndarray, device: torch.device, requires_grad: bool = False) -> torch.Tensor:
    t = torch.as_tensor(a, dtype=torch.float32, device=device)
    t.requires_grad_(requires_grad)
    return t


def split_uv(out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return (out[:, 0:1], out[:, 1:2])


def model_and_derivatives(model: nn.Module, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
    uv = model(xyz)
    u, v = split_uv(uv)
    grad_u = torch.autograd.grad(
        u, xyz, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0]
    grad_v = torch.autograd.grad(
        v, xyz, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]
    v_x = grad_v[:, 0:1]
    v_y = grad_v[:, 1:2]
    v_t = grad_v[:, 2:3]
    grad_u_x = torch.autograd.grad(
        u_x, xyz, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0]
    grad_u_y = torch.autograd.grad(
        u_y, xyz, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True
    )[0]
    grad_v_x = torch.autograd.grad(
        v_x, xyz, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True
    )[0]
    grad_v_y = torch.autograd.grad(
        v_y, xyz, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True
    )[0]
    u_xx = grad_u_x[:, 0:1]
    u_yy = grad_u_y[:, 1:2]
    v_xx = grad_v_x[:, 0:1]
    v_yy = grad_v_y[:, 1:2]
    return {
        "u": u,
        "v": v,
        "u_x": u_x,
        "u_y": u_y,
        "u_t": u_t,
        "u_xx": u_xx,
        "u_yy": u_yy,
        "v_x": v_x,
        "v_y": v_y,
        "v_t": v_t,
        "v_xx": v_xx,
        "v_yy": v_yy,
    }


@dataclass
class EvalResults:
    u_pred: np.ndarray
    v_pred: np.ndarray
    u_data_abs: np.ndarray
    v_data_abs: np.ndarray
    u_ref_abs: np.ndarray
    v_ref_abs: np.ndarray
    ru_abs: np.ndarray
    rv_abs: np.ndarray
    ru_signed: np.ndarray
    rv_signed: np.ndarray
    ru_mass: np.ndarray
    rv_mass: np.ndarray
    metrics: Dict[str, float]


def eval_on_points(
    model: nn.Module,
    xyz_np: np.ndarray,
    device: torch.device,
    batch_size: int,
    need_derivatives: bool = False,
) -> Dict[str, np.ndarray]:
    n = xyz_np.shape[0]
    out: Dict[str, List[np.ndarray]] = {"u": [], "v": []}
    if need_derivatives:
        for key in ["u_x", "u_y", "u_t", "u_xx", "u_yy", "v_x", "v_y", "v_t", "v_xx", "v_yy"]:
            out[key] = []
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        xyz = to_tensor(xyz_np[start:stop], device, requires_grad=need_derivatives)
        if need_derivatives:
            vals = model_and_derivatives(model, xyz)
            for key, value in vals.items():
                out[key].append(value.detach().cpu().numpy())
        else:
            with torch.no_grad():
                uv = model(xyz)
            u, v = split_uv(uv)
            out["u"].append(u.detach().cpu().numpy())
            out["v"].append(v.detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).reshape(-1) for k, v in out.items()}


def compute_global_balance_residuals(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    u_t: np.ndarray,
    v_t: np.ndarray,
    u_x: np.ndarray,
    u_y: np.ndarray,
    v_x: np.ndarray,
    v_y: np.ndarray,
    nu: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the componentwise global integral-balance residuals for the 2D Burgers system,
    matching the manuscript definition

        r_{q,M}(t) = dM_q/dt - ∫_Ω q (∇·U) dΩ - ν ∫_{∂Ω} ∇q · n ds,  q ∈ {u,v}.

    with dM_q/dt evaluated as ∫_Ω q_t dΩ from the continuous PINN representation.
    """
    div_u = u_x + v_y
    dMu_dt = trapz2d(u_t, x=x, y=y)
    dMv_dt = trapz2d(v_t, x=x, y=y)
    interior_u = trapz2d(u * div_u, x=x, y=y)
    interior_v = trapz2d(v * div_u, x=x, y=y)
    qx_left_u = u_x[:, :, 0]
    qx_right_u = u_x[:, :, -1]
    qy_bottom_u = u_y[:, 0, :]
    qy_top_u = u_y[:, -1, :]
    qx_left_v = v_x[:, :, 0]
    qx_right_v = v_x[:, :, -1]
    qy_bottom_v = v_y[:, 0, :]
    qy_top_v = v_y[:, -1, :]
    bdry_u = trapz_compat(-qx_left_u + qx_right_u, y, axis=1) + trapz_compat(
        -qy_bottom_u + qy_top_u, x, axis=1
    )
    bdry_v = trapz_compat(-qx_left_v + qx_right_v, y, axis=1) + trapz_compat(
        -qy_bottom_v + qy_top_v, x, axis=1
    )
    ru_M = dMu_dt - interior_u - nu * bdry_u
    rv_M = dMv_dt - interior_v - nu * bdry_v
    return (ru_M.astype(np.float32), rv_M.astype(np.float32))


def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    u_obs: np.ndarray,
    v_obs: np.ndarray,
    u_ref: np.ndarray,
    v_ref: np.ndarray,
    nu: float,
    train_time_s: float,
    device: torch.device,
    eval_batch: int,
) -> EvalResults:
    X, Y, T = np.meshgrid(x, y, t, indexing="xy")
    xyz_full = np.stack(
        [
            np.transpose(X, (2, 0, 1)).reshape(-1),
            np.transpose(Y, (2, 0, 1)).reshape(-1),
            np.transpose(T, (2, 0, 1)).reshape(-1),
        ],
        axis=1,
    ).astype(np.float32)
    vals = eval_on_points(
        model, xyz_np=xyz_full, device=device, batch_size=eval_batch, need_derivatives=True
    )
    nt, ny, nx = u_obs.shape

    def rs(name: str) -> np.ndarray:
        return vals[name].reshape(nt, ny, nx).astype(np.float32)

    u_pred = rs("u")
    v_pred = rs("v")
    u_x = rs("u_x")
    u_y = rs("u_y")
    u_t = rs("u_t")
    u_xx = rs("u_xx")
    u_yy = rs("u_yy")
    v_x = rs("v_x")
    v_y = rs("v_y")
    v_t = rs("v_t")
    v_xx = rs("v_xx")
    v_yy = rs("v_yy")
    ru = u_t + u_pred * u_x + v_pred * u_y - nu * (u_xx + u_yy)
    rv = v_t + u_pred * v_x + v_pred * v_y - nu * (v_xx + v_yy)
    u_data_abs = np.abs(u_pred - u_obs).astype(np.float32)
    v_data_abs = np.abs(v_pred - v_obs).astype(np.float32)
    u_ref_abs = np.abs(u_pred - u_ref).astype(np.float32)
    v_ref_abs = np.abs(v_pred - v_ref).astype(np.float32)
    ru_abs = np.abs(ru).astype(np.float32)
    rv_abs = np.abs(rv).astype(np.float32)
    ru_mass, rv_mass = compute_global_balance_residuals(
        x=x, y=y, u=u_pred, v=v_pred, u_t=u_t, v_t=v_t, u_x=u_x, u_y=u_y, v_x=v_x, v_y=v_y, nu=nu
    )
    interior = np.ones_like(u_pred, dtype=bool)
    interior[0, :, :] = False
    interior[:, 0, :] = False
    interior[:, -1, :] = False
    interior[:, :, 0] = False
    interior[:, :, -1] = False
    bc_mask = np.zeros_like(u_pred, dtype=bool)
    bc_mask[:, 0, :] = True
    bc_mask[:, -1, :] = True
    bc_mask[:, :, 0] = True
    bc_mask[:, :, -1] = True
    Lu_data = float(np.mean((u_pred[interior] - u_obs[interior]) ** 2))
    Lv_data = float(np.mean((v_pred[interior] - v_obs[interior]) ** 2))
    Lu_pde = float(np.mean(ru[interior] ** 2))
    Lv_pde = float(np.mean(rv[interior] ** 2))
    Lu_ic = float(np.mean((u_pred[0] - u_obs[0]) ** 2))
    Lv_ic = float(np.mean((v_pred[0] - v_obs[0]) ** 2))
    Lu_bc = float(np.mean(u_pred[bc_mask] ** 2))
    Lv_bc = float(np.mean(v_pred[bc_mask] ** 2))
    Ltotal = Lu_data + Lv_data + Lu_pde + Lv_pde + Lu_ic + Lv_ic + Lu_bc + Lv_bc
    u_actual_sqnorm_over_N = float(np.linalg.norm(u_ref_abs.ravel()) ** 2 / u_ref_abs.size)
    v_actual_sqnorm_over_N = float(np.linalg.norm(v_ref_abs.ravel()) ** 2 / v_ref_abs.size)
    metrics = {
        "Lu_data": Lu_data,
        "Lv_data": Lv_data,
        "Lu_pde": Lu_pde,
        "Lv_pde": Lv_pde,
        "Lu_IC": Lu_ic,
        "Lv_IC": Lv_ic,
        "Lu_BC": Lu_bc,
        "Lv_BC": Lv_bc,
        "Ltotal": Ltotal,
        "MAE_ref_u": float(np.mean(u_ref_abs)),
        "MAE_ref_v": float(np.mean(v_ref_abs)),
        "MSE_ref_u": float(np.mean((u_pred - u_ref) ** 2)),
        "MSE_ref_v": float(np.mean((v_pred - v_ref) ** 2)),
        "actual_error_sqnorm_over_N_u": u_actual_sqnorm_over_N,
        "actual_error_sqnorm_over_N_v": v_actual_sqnorm_over_N,
        "RMS_ref_u": float(np.sqrt(u_actual_sqnorm_over_N)),
        "RMS_ref_v": float(np.sqrt(v_actual_sqnorm_over_N)),
        "RMS_balance_u": float(np.sqrt(np.mean(ru_mass**2))),
        "RMS_balance_v": float(np.sqrt(np.mean(rv_mass**2))),
        "max_abs_balance_u": float(np.max(np.abs(ru_mass))),
        "max_abs_balance_v": float(np.max(np.abs(rv_mass))),
        "train_time_seconds": float(train_time_s),
        "max_abs_IC_u": float(np.max(np.abs(u_pred[0] - u_obs[0]))),
        "max_abs_IC_v": float(np.max(np.abs(v_pred[0] - v_obs[0]))),
        "max_abs_BC_u": float(np.max(np.abs(u_pred[bc_mask]))),
        "max_abs_BC_v": float(np.max(np.abs(v_pred[bc_mask]))),
    }
    return EvalResults(
        u_pred=u_pred,
        v_pred=v_pred,
        u_data_abs=u_data_abs,
        v_data_abs=v_data_abs,
        u_ref_abs=u_ref_abs,
        v_ref_abs=v_ref_abs,
        ru_abs=ru_abs,
        rv_abs=rv_abs,
        ru_signed=ru.astype(np.float32),
        rv_signed=rv.astype(np.float32),
        ru_mass=ru_mass,
        rv_mass=rv_mass,
        metrics=metrics,
    )
